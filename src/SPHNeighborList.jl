module SPHNeighborList

export ConstructStencil, ExtractCells!, UpdateNeighbors!, BuildNeighborCellLists!,
       NeighborListScratch, UpdateCellCounts!, ComputeCellParticleCounts!,
       ComputeCellNeighborCounts!, ComputeCellParticleCounts, ComputeCellNeighborCounts,
       UpdateΔx!

using StaticArrays
using Base.Threads

mutable struct NeighborListScratch
    CellCounts::Vector{Int}
    CellOccupied::Vector{Bool}
    NeighborCounts::Vector{Int}

    function NeighborListScratch()
        new(Int[], Bool[], Int[])
    end
end

@inline function ResizeNeighborListScratch!(Scratch::NeighborListScratch, CellCount)
    resize!(Scratch.CellCounts, CellCount)
    resize!(Scratch.CellOccupied, CellCount)
    resize!(Scratch.NeighborCounts, CellCount)
    return nothing
end

function ConstructStencil(V::Val{d}) where d
    return CartesianIndices(ntuple(_ -> -1:1, V))
end

function BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView, CellDict, CellOccupied)
    TargetLen   = length(UniqueCellsView)
    OriginalLen = length(NeighborCellLists)
    resize!(NeighborCellLists, TargetLen)

    if TargetLen > OriginalLen
        @inbounds for Index in (OriginalLen + 1):TargetLen
            NeighborCellLists[Index] = Int[]
        end
    end

    if nthreads() > 1
        @inbounds Threads.@threads for CellIndex in eachindex(UniqueCellsView)
            Neighbors = NeighborCellLists[CellIndex]
            empty!(Neighbors)
            sizehint!(Neighbors, length(FullStencil))
            Cell = UniqueCellsView[CellIndex]
            for Offset in FullStencil
                NeighborCell = Cell + Offset
                NeighborIndex = get(CellDict, NeighborCell, 1)
                if NeighborIndex != CellIndex && CellOccupied[NeighborIndex]
                    push!(Neighbors, NeighborIndex)
                end
            end
        end
    else
        @inbounds for CellIndex in eachindex(UniqueCellsView)
            Neighbors = NeighborCellLists[CellIndex]
            empty!(Neighbors)
            sizehint!(Neighbors, length(FullStencil))
            Cell = UniqueCellsView[CellIndex]
            for Offset in FullStencil
                NeighborCell = Cell + Offset
                NeighborIndex = get(CellDict, NeighborCell, 1)
                if NeighborIndex != CellIndex && CellOccupied[NeighborIndex]
                    push!(Neighbors, NeighborIndex)
                end
            end
        end
    end

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
- `CutOff`: The cutoff value used for cell extraction.
- `SortingScratchSpace`: Scratch space for sorting.
- `ParticleRanges`: Array to store the ranges of particles in each cell.
- `UniqueCells`: Array to store the unique cells.

# Returns
- `IndexCounter`: The number of unique cells identified.
"""
function UpdateNeighbors!(Particles, InverseCutOff, SortingScratchSpace,
                          ParticleRanges, UniqueCells, CellDict)
    ExtractCells!(Particles, InverseCutOff)

    sort!(Particles, by = p -> p.Cells; scratch=SortingScratchSpace)
    Cells = @views Particles.Cells
    @. ParticleRanges             = zero(eltype(ParticleRanges))
    ParticleRanges[1] = 1
    IndexCounter                  = 2
    ParticleRanges[IndexCounter]  = 1
    UniqueCells[IndexCounter]     = Cells[1]
    empty!(CellDict)
    CellDict[Cells[1]] = IndexCounter

    @inbounds @simd ivdep for Index in eachindex(Cells)[2:end]
        if Cells[Index] != Cells[Index - 1] # Equivalent to diff(Cells) != 0
            IndexCounter                 += 1
            ParticleRanges[IndexCounter]  = Index
            UniqueCells[IndexCounter]     = Cells[Index]
            CellDict[Cells[Index]]       = IndexCounter
        end
    end
    ParticleRanges[IndexCounter + 1]  = length(ParticleRanges)

    return IndexCounter
end

function UpdateCellCounts!(Scratch::NeighborListScratch, ParticleRanges, CellCount)
    ResizeNeighborListScratch!(Scratch, CellCount)
    Counts = Scratch.CellCounts
    Occupied = Scratch.CellOccupied
    @inbounds for Index in 1:CellCount
        Count = ParticleRanges[Index + 1] - ParticleRanges[Index]
        Counts[Index] = Count
        Occupied[Index] = Count > 0
    end
    return nothing
end

function ComputeCellParticleCounts!(Counts, ParticleRanges, CellCount)
    resize!(Counts, CellCount)
    @inbounds for Index in 1:CellCount
        Counts[Index] = ParticleRanges[Index + 1] - ParticleRanges[Index]
    end
    return nothing
end

function ComputeCellParticleCounts(ParticleRanges, CellCount)
    Counts = Vector{Int}(undef, CellCount)
    ComputeCellParticleCounts!(Counts, ParticleRanges, CellCount)
    return Counts
end

function ComputeCellNeighborCounts!(Neighbors, Counts, NeighborCellLists, CellCount)
    resize!(Neighbors, CellCount)
    @inbounds for Index in 1:CellCount
        NeighborTotal = 0
        for NeighborIndex in NeighborCellLists[Index]
            NeighborTotal += Counts[NeighborIndex]
        end
        Neighbors[Index] = max(Counts[Index] - 1, 0) + NeighborTotal
    end
    return nothing
end

function ComputeCellNeighborCounts(ParticleRanges, NeighborCellLists, CellCount)
    Counts = ComputeCellParticleCounts(ParticleRanges, CellCount)
    Neighbors = Vector{Int}(undef, CellCount)
    ComputeCellNeighborCounts!(Neighbors, Counts, NeighborCellLists, CellCount)
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
