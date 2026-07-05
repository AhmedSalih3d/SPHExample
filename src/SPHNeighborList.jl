module SPHNeighborList

export ConstructStencil, ExtractCells!, UpdateNeighbors!, BuildNeighborCellLists!, ComputeCellParticleCounts, ComputeCellNeighborCounts, UpdateΔx!, FindCellIndex, CompressedNeighborCellLists

using StaticArrays

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

"""
    CompressedNeighborCellLists(FullStencil, CellCapacity)

Stores per-cell neighbor adjacency in one dense `(max_neighbors, n_cells)` matrix
plus a count per cell, avoiding one small `Vector{Int}` allocation per cell.
"""
mutable struct CompressedNeighborCellLists
    Indices::Matrix{Int}
    Counts::Vector{Int}
end

function CompressedNeighborCellLists(FullStencil, CellCapacity::Integer)
    MaxNeighborCount = length(FullStencil) - 1
    return CompressedNeighborCellLists(Matrix{Int}(undef, MaxNeighborCount, CellCapacity), zeros(Int, CellCapacity))
end

struct NeighborCellIndices
    Indices::Matrix{Int}
    CellIndex::Int
    Count::Int
end

@inline Base.length(Neighbors::NeighborCellIndices) = Neighbors.Count
@inline Base.eltype(::Type{NeighborCellIndices}) = Int
@inline Base.IteratorSize(::Type{NeighborCellIndices}) = Base.HasLength()

@inline function Base.iterate(Neighbors::NeighborCellIndices, State::Int = 1)
    State > Neighbors.Count && return nothing
    return (@inbounds(Neighbors.Indices[State, Neighbors.CellIndex]), State + 1)
end

@inline function Base.getindex(NeighborCellLists::CompressedNeighborCellLists, CellIndex::Integer)
    return NeighborCellIndices(NeighborCellLists.Indices, CellIndex, @inbounds(NeighborCellLists.Counts[CellIndex]))
end

function EnsureNeighborCellCapacity!(NeighborCellLists::CompressedNeighborCellLists, CellCount::Integer)
    if CellCount <= length(NeighborCellLists.Counts)
        return nothing
    end

    MaxNeighborCount = size(NeighborCellLists.Indices, 1)
    NewIndices = Matrix{Int}(undef, MaxNeighborCount, CellCount)
    NewCounts = zeros(Int, CellCount)
    OldCellCount = length(NeighborCellLists.Counts)
    @inbounds for CellIndex in 1:OldCellCount
        NewCounts[CellIndex] = NeighborCellLists.Counts[CellIndex]
        for NeighborOffset in 1:MaxNeighborCount
            NewIndices[NeighborOffset, CellIndex] = NeighborCellLists.Indices[NeighborOffset, CellIndex]
        end
    end
    NeighborCellLists.Indices = NewIndices
    NeighborCellLists.Counts = NewCounts
    return nothing
end

function BuildNeighborCellLists!(NeighborCellLists::CompressedNeighborCellLists, FullStencil, UniqueCellsView, ParticleRanges)
    TargetLen = length(UniqueCellsView)
    EnsureNeighborCellCapacity!(NeighborCellLists, TargetLen)

    @inbounds for CellIndex in eachindex(UniqueCellsView)
        NeighborCount = 0
        Cell = UniqueCellsView[CellIndex]
        for Offset in FullStencil
            NeighborCell = Cell + Offset
            NeighborIndex = FindCellIndex(UniqueCellsView, NeighborCell)
            StartIndex = ParticleRanges[NeighborIndex]
            EndIndex = ParticleRanges[NeighborIndex + 1] - 1
            if StartIndex <= EndIndex && NeighborIndex != CellIndex
                NeighborCount += 1
                NeighborCellLists.Indices[NeighborCount, CellIndex] = NeighborIndex
            end
        end
        NeighborCellLists.Counts[CellIndex] = NeighborCount
    end

    return NeighborCellLists
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
- `CutOff`: The cutoff value used for cell extraction.
- `SortingScratchSpace`: Scratch space for sorting.
- `ParticleRanges`: Array to store the ranges of particles in each cell.
- `UniqueCells`: Array to store the unique cells.

# Returns
- `IndexCounter`: The number of unique cells identified.
"""
function UpdateNeighbors!(Particles, InverseCutOff, SortingScratchSpace,
                          ParticleRanges, UniqueCells, CellListIndices)
    ExtractCells!(Particles, InverseCutOff)

    sort!(Particles, by = p -> p.Cells; scratch=SortingScratchSpace)
    Cells = @views Particles.Cells
    UniqueCells[1] = MinCell(eltype(Cells))
    @. ParticleRanges             = zero(eltype(ParticleRanges))
    ParticleRanges[1] = 1
    IndexCounter                  = 2
    ParticleRanges[IndexCounter]  = 1
    UniqueCells[IndexCounter]     = Cells[1]
    CellListIndices[1]            = IndexCounter

    @inbounds @simd ivdep for Index in eachindex(Cells)[2:end]
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

Increment Δx by twice the maximum ‖posₙ⁺[i] – pos[i]‖, without ever allocating.
Returns the new Δx.
"""
@inline function UpdateΔx!(Δx::T,
                           posₙ⁺::AbstractVector{SVector{D, T}},
                           pos   ::AbstractVector{SVector{D, T}}) where {D, T<:Real}
    maxd = zero(T)
    @inbounds for i in eachindex(posₙ⁺, pos)
        # compute squared norm manually
        sumsq = zero(T)
        @inbounds for j in 1:D
            d = posₙ⁺[i][j] - pos[i][j]
            sumsq += d*d
        end
        # sqrt/T is allocation-free on scalars
        nrm = sqrt(sumsq)
        if nrm > maxd
            maxd = nrm
        end
    end
    return Δx + 4 * maxd
end

end
