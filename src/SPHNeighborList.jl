module SPHNeighborList

export ConstructStencil, ExtractCells!, UpdateNeighbors!, BuildNeighborCellLists!, ComputeCellParticleCounts, ComputeCellNeighborCounts, UpdateΔx!, FindCellIndex, NeighborSortScratch, PackedNeighborCellLists

using StaticArrays

"""
    PackedNeighborCellLists(MaxCellCount)

Reusable contiguous neighbor-cell storage. Consecutive IDs are encoded as runs
with inclusive start/end IDs, using the smallest unsigned integer that can
represent `MaxCellCount` (including the sentinel cell); offsets use `Int`.
Indexing returns a read-only vector in the original stencil order. Rebuild with
`BuildNeighborCellLists!`; the declared ID width must accommodate every cell.
"""
struct PackedNeighborCellLists{I<:Unsigned}
    Offsets::Vector{Int} # First stored run for each owner cell, plus a terminator.
    Neighbors::Vector{I} # Inclusive first neighbor-cell ID of each run.
    RunEnds::Vector{I}   # Inclusive last neighbor-cell ID of each run.
end

# Preserve construction from explicit lists: every supplied ID is a singleton.
PackedNeighborCellLists(Offsets::Vector{Int}, Neighbors::Vector{I}) where {I<:Unsigned} =
    PackedNeighborCellLists(Offsets, Neighbors, copy(Neighbors))

function PackedNeighborCellLists(MaxCellCount::Integer)
    MaxCellCount >= 0 || throw(ArgumentError("MaxCellCount must be nonnegative"))
    IndexType = MaxCellCount <= typemax(UInt8) ? UInt8 :
                MaxCellCount <= typemax(UInt16) ? UInt16 :
                MaxCellCount <= typemax(UInt32) ? UInt32 : UInt64
    return PackedNeighborCellLists(Int[1], IndexType[], IndexType[])
end

struct NeighborCellRange{I<:Unsigned} <: AbstractVector{Int}
    Neighbors::Vector{I}
    RunEnds::Vector{I}
    First::Int
    Last::Int
end

function Base.size(Range::NeighborCellRange)
    Count = 0
    @inbounds for Index in Range.First:Range.Last
        Count += Int(Range.RunEnds[Index]) - Int(Range.Neighbors[Index]) + 1
    end
    return (Count,)
end
Base.IndexStyle(::Type{<:NeighborCellRange}) = IndexLinear()

Base.@propagate_inbounds function Base.getindex(Range::NeighborCellRange, Index::Int)
    @boundscheck checkbounds(Range, Index)
    @inbounds for Run in Range.First:Range.Last
        FirstCell = Int(Range.Neighbors[Run])
        Count = Int(Range.RunEnds[Run]) - FirstCell + 1
        Index <= Count && return FirstCell + Index - 1
        Index -= Count
    end
    throw(BoundsError(Range, Index))
end

# Decode directly to native indices without constructing a general array view.
@inline function Base.iterate(Range::NeighborCellRange)
    Range.First > Range.Last && return nothing
    @inbounds Cell = Int(Range.Neighbors[Range.First])
    return Cell, (Range.First, Cell)
end

@inline function Base.iterate(Range::NeighborCellRange, State::Tuple{Int,Int})
    Run, Cell = State
    @inbounds if Cell < Int(Range.RunEnds[Run])
        return Cell + 1, (Run, Cell + 1)
    end
    Run += 1
    Run > Range.Last && return nothing
    @inbounds NextCell = Int(Range.Neighbors[Run])
    return NextCell, (Run, NextCell)
end

Base.length(Lists::PackedNeighborCellLists) = length(Lists.Offsets) - 1

Base.@propagate_inbounds function Base.getindex(Lists::PackedNeighborCellLists, CellIndex::Integer)
    return NeighborCellRange(Lists.Neighbors, Lists.RunEnds, Lists.Offsets[CellIndex], Lists.Offsets[CellIndex + 1] - 1)
end

# Consecutive cell IDs own adjacent particle ranges. Joining only those cells
# removes inner-loop restarts without changing which particles are visited or
# their accumulation order. No temporary lists are allocated during traversal.
struct NeighborParticleRanges{C<:AbstractVector{Int},R<:AbstractVector{Int}}
    Cells::C
    Ranges::R
end

Base.IteratorSize(::Type{<:NeighborParticleRanges}) = Base.SizeUnknown()
Base.eltype(::Type{<:NeighborParticleRanges}) = UnitRange{Int}

# Packed lists already merged the cells at rebuild time. Each particle reads
# just two narrow IDs per run instead of scanning all neighbor cells again.
@inline function Base.iterate(Spans::NeighborParticleRanges{<:NeighborCellRange}, Index::Int=Spans.Cells.First)
    Index > Spans.Cells.Last && return nothing
    @inbounds begin
        FirstCell = Int(Spans.Cells.Neighbors[Index])
        LastCell = Int(Spans.Cells.RunEnds[Index])
        return Spans.Ranges[FirstCell]:(Spans.Ranges[LastCell + 1] - 1), Index + 1
    end
end

@inline function Base.iterate(Spans::NeighborParticleRanges, Index::Int=firstindex(Spans.Cells))
    Index > lastindex(Spans.Cells) && return nothing
    @inbounds begin
        FirstCell = Spans.Cells[Index]
        LastCell = FirstCell
        Index += 1
        while Index <= lastindex(Spans.Cells)
            NextCell = Spans.Cells[Index]
            NextCell == LastCell + 1 || break
            LastCell = NextCell
            Index += 1
        end
        return Spans.Ranges[FirstCell]:(Spans.Ranges[LastCell + 1] - 1), Index
    end
end

"""
    NeighborSortScratch(ParticleCount)

Reusable integer buffers for sorting particles by cell without repeatedly moving
complete particle records. Equal-cell particles retain their previous order.
"""
struct NeighborSortScratch
    Permutation::Vector{Int}
    Sorting::Vector{Int}
end

function NeighborSortScratch(ParticleCount::Integer)
    return NeighborSortScratch(Vector{Int}(undef, ParticleCount), Vector{Int}(undef, ParticleCount))
end

function SortParticlesByCell!(Particles, Scratch::NeighborSortScratch)
    issorted(Particles.Cells) && return nothing
    Permutation = Scratch.Permutation
    resize!(Permutation, length(Particles))
    # sortperm! resolves equal keys by their original index, preserving the
    # accumulation order of interactions within each cell.
    sortperm!(Permutation, Particles.Cells; scratch=Scratch.Sorting)

    # Apply each permutation cycle once to whole rows so every particle field,
    # including optional boundary and kernel data, remains aligned. Negative
    # indices mark visited entries; the next sortperm! reinitializes the buffer.
    @inbounds for StartIndex in eachindex(Permutation)
        NextIndex = Permutation[StartIndex]
        NextIndex <= 0 && continue
        if NextIndex == StartIndex
            Permutation[StartIndex] = -NextIndex
            continue
        end

        SavedParticle = Particles[StartIndex]
        CurrentIndex = StartIndex
        while NextIndex != StartIndex
            Particles[CurrentIndex] = Particles[NextIndex]
            Permutation[CurrentIndex] = -NextIndex
            CurrentIndex = NextIndex
            NextIndex = Permutation[CurrentIndex]
        end
        Particles[CurrentIndex] = SavedParticle
        Permutation[CurrentIndex] = -NextIndex
    end
    return nothing
end

function SortParticlesByCell!(Particles, SortingScratchSpace)
    # Retain support for callers supplying Base.Sort.make_scratch row buffers.
    sort!(Particles, by = p -> p.Cells; scratch=SortingScratchSpace)
    return nothing
end

function ConstructStencil(V::Val{d}) where d
    return CartesianIndices(ntuple(_ -> -1:1, V))
end

@inline function MinCell(::Type{CartesianIndex{D}}) where D
    return CartesianIndex(ntuple(_ -> typemin(Int), Val(D)))
end

@inline function FindCellIndex(UniqueCellsView, Cell)
    idx = searchsortedfirst(UniqueCellsView, Cell)
    if idx <= length(UniqueCellsView) && UniqueCellsView[idx] == Cell
        return idx
    end
    return 1
end

function BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView, ParticleRanges)
    CellIndexMap = Dict{eltype(UniqueCellsView), Int}()
    sizehint!(CellIndexMap, length(UniqueCellsView))
    BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView, ParticleRanges, CellIndexMap)
    return nothing
end

function BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView, ParticleRanges, CellIndexMap)
    TargetLen   = length(UniqueCellsView)
    OriginalLen = length(NeighborCellLists)
    resize!(NeighborCellLists, TargetLen)
    MaxNeighborCount = max(length(FullStencil) - 1, 0)

    if TargetLen > OriginalLen
        @inbounds for Index in (OriginalLen + 1):TargetLen
            NeighborCellLists[Index] = Int[]
            sizehint!(NeighborCellLists[Index], MaxNeighborCount)
        end
    end

    empty!(CellIndexMap)
    sizehint!(CellIndexMap, TargetLen)
    @inbounds for CellIndex in eachindex(UniqueCellsView)
        if ParticleRanges[CellIndex] < ParticleRanges[CellIndex + 1]
            CellIndexMap[UniqueCellsView[CellIndex]] = CellIndex
        end
    end

    @inbounds for CellIndex in eachindex(UniqueCellsView)
        Neighbors = NeighborCellLists[CellIndex]
        empty!(Neighbors)
        if ParticleRanges[CellIndex] >= ParticleRanges[CellIndex + 1]
            continue
        end
        Cell = UniqueCellsView[CellIndex]
        for Offset in FullStencil
            NeighborCell = Cell + Offset
            NeighborIndex = get(CellIndexMap, NeighborCell, 0)
            if !iszero(NeighborIndex) && NeighborIndex != CellIndex
                push!(Neighbors, NeighborIndex)
            end
        end
    end

    return nothing
end

function BuildNeighborCellLists!(Lists::PackedNeighborCellLists{I}, FullStencil,
                                 UniqueCellsView, ParticleRanges, CellIndexMap) where {I}
    CellCount = length(UniqueCellsView)
    CellCount <= typemax(I) || throw(ArgumentError("Cell count exceeds packed neighbor ID capacity"))
    resize!(Lists.Offsets, CellCount + 1)
    empty!(Lists.Neighbors)
    empty!(Lists.RunEnds)
    # A Cartesian stencil has one run per row, plus a split around the current
    # cell. This is a capacity hint only; arbitrary stencils can grow as needed.
    RunsPerCell = FullStencil isa CartesianIndices ?
        cld(length(FullStencil), max(size(FullStencil, 1), 1)) + 1 : length(FullStencil)
    sizehint!(Lists.Neighbors, CellCount * RunsPerCell)
    sizehint!(Lists.RunEnds, CellCount * RunsPerCell)
    empty!(CellIndexMap)
    sizehint!(CellIndexMap, CellCount)
    @inbounds for CellIndex in eachindex(UniqueCellsView)
        if ParticleRanges[CellIndex] < ParticleRanges[CellIndex + 1]
            CellIndexMap[UniqueCellsView[CellIndex]] = CellIndex
        end
    end
    @inbounds for CellIndex in eachindex(UniqueCellsView)
        Lists.Offsets[CellIndex] = length(Lists.Neighbors) + 1
        if ParticleRanges[CellIndex] >= ParticleRanges[CellIndex + 1]
            continue
        end
        Cell = UniqueCellsView[CellIndex]
        for Offset in FullStencil
            NeighborIndex = get(CellIndexMap, Cell + Offset, 0)
            if !iszero(NeighborIndex) && NeighborIndex != CellIndex
                if length(Lists.Neighbors) >= Lists.Offsets[CellIndex] &&
                   NeighborIndex == Int(Lists.RunEnds[end]) + 1
                    Lists.RunEnds[end] = I(NeighborIndex)
                else
                    push!(Lists.Neighbors, I(NeighborIndex))
                    push!(Lists.RunEnds, I(NeighborIndex))
                end
            end
        end
    end
    Lists.Offsets[end] = length(Lists.Neighbors) + 1
    return nothing
end

"""
Extracts the cells for each particle based on their positions and the inverse cutoff value.

# Arguments
- `Particles`: The particles whose cells are to be extracted.
- `::Val{InverseCutOff}`: The inverse cutoff value used for cell extraction.

# Returns
- `nothing`: This function modifies the `Particles` in place.
"""
# Replace unsafe_trunc with trunc if this ever errors
@inline function MapFloor(X, InverseCutOff)
    # This is different than just doing muladd(x,InverseCutOff,0.5) because it rounds towards zero.
    # Consider -1.7 + 0.5, this would give -1.2 and then truncated 1, but we want -2, therefore absolute addition beforehand
    # We add 0.5 instead of 1, to ensure proper rounding behavior when restoring the sign for negative numbers.
    Int(sign(X)) * unsafe_trunc(Int, muladd(abs(X), InverseCutOff, 0.5))
end

@inline function ExtractCells!(Particles, InverseCutOff)
    @inbounds @simd ivdep for Index ∈ eachindex(Particles.Cells)
        Particles.Cells[Index] = CartesianIndex(map(X -> MapFloor(X, InverseCutOff), Tuple(Particles.Position[Index])))
    end
    return nothing
end

"""
Updates the neighbor list and sorts particles by their cell indices.

# Arguments
- `Particles`: The particles whose neighbors are to be updated.
- `InverseCutOff`: The inverse cell width used for cell extraction.
- `SortingScratchSpace`: A `NeighborSortScratch` for reusable index sorting, or a legacy particle-row sorting buffer.
- `ParticleRanges`: Array to store the ranges of particles in each cell.
- `UniqueCells`: Array to store the unique cells.
- `CellListIndices`: Array mapping each sorted particle to its cell range.

# Returns
- `IndexCounter`: The number of unique cells identified.
"""
function UpdateNeighbors!(Particles, InverseCutOff, SortingScratchSpace,
                          ParticleRanges, UniqueCells, CellListIndices)
    ExtractCells!(Particles, InverseCutOff)

    SortParticlesByCell!(Particles, SortingScratchSpace)
    Cells = @views Particles.Cells
    UniqueCells[1] = MinCell(eltype(Cells))
    @. ParticleRanges             = zero(eltype(ParticleRanges))
    ParticleRanges[1] = 1
    IndexCounter                  = 2
    ParticleRanges[IndexCounter]  = 1
    UniqueCells[IndexCounter]     = Cells[1]
    CellListIndices[1]            = IndexCounter

    @inbounds for Index in 2:length(Cells)
        if Cells[Index] != Cells[Index - 1] # Equivalent to diff(Cells) != 0
            IndexCounter                 += 1
            ParticleRanges[IndexCounter]  = Index
            UniqueCells[IndexCounter]     = Cells[Index]
        end
        CellListIndices[Index]            = IndexCounter
    end
    ParticleRanges[IndexCounter + 1]  = length(Cells) + 1

    return IndexCounter
end

function ComputeCellParticleCounts(ParticleRanges, CellCount)
    Counts = Vector{Int}(undef, CellCount)
    @inbounds for Index in 1:CellCount
        Counts[Index] = ParticleRanges[Index + 1] - ParticleRanges[Index]
    end
    return Counts
end

function ComputeCellNeighborCounts(ParticleRanges, NeighborCellLists, CellCount)
    Counts = ComputeCellParticleCounts(ParticleRanges, CellCount)
    Neighbors = Vector{Int}(undef, CellCount)
    @inbounds for Index in 1:CellCount
        NeighborTotal = 0
        for NeighborIndex in NeighborCellLists[Index]
            NeighborTotal += Counts[NeighborIndex]
        end
        Neighbors[Index] = max(Counts[Index] - 1, 0) + NeighborTotal
    end
    return Neighbors
end

"""
    UpdateΔx!(Δx, posₙ⁺, pos)

Increment Δx by four times the maximum ‖posₙ⁺[i] – pos[i]‖, without allocating.
Here `posₙ⁺` is the predictor position, not the position at the last rebuild.
This accumulated motion estimate alone does not guarantee complete neighbor
coverage when the cached cell search has no extra margin beyond kernel support.
Returns the new Δx.
"""
# @inline function UpdateΔx!(Δx::T,
#                            posₙ⁺::AbstractVector{SVector{D, T}},
#                            pos   ::AbstractVector{SVector{D, T}}) where {D, T<:Real}
#     maxd = zero(T)
#     @inbounds for i in eachindex(posₙ⁺, pos)
#         # compute squared norm manually
#         sumsq = zero(T)
#         @inbounds for j in 1:D
#             d = posₙ⁺[i][j] - pos[i][j]
#             sumsq += d*d
#         end
#         # sqrt/T is allocation-free on scalars
#         nrm = sqrt(sumsq)
#         if nrm > maxd
#             maxd = nrm
#         end
#     end
#     return Δx + 4 * maxd
# end
@inline function UpdateΔx!(
    Δx::T,
    posₙ⁺::AbstractVector{SVector{D,T}},
    pos::AbstractVector{SVector{D,T}},
) where {D,T<:Real}

    maxd² = zero(T)

    @inbounds for i in eachindex(posₙ⁺, pos)
        sumsq = zero(T)

        @inbounds for j in 1:D
            d = posₙ⁺[i][j] - pos[i][j]
            sumsq += d * d
        end

        maxd² = max(maxd², sumsq)
    end

    return Δx + 4 * sqrt(maxd²)
end

end
