module SPHNeighborList

export ConstructStencil, ExtractCells!, UpdateNeighbors!,
       BuildNeighborCellLists!, ComputeCellParticleCounts, ComputeCellNeighborCounts, UpdateΔx!,
       CellIndexLookup, InitializeCellIndexLookup, CellLookupIndex

using StaticArrays

mutable struct CellIndexLookup{D}
    Grid::Vector{Int}
    MinCell::MVector{D, Int}
    MaxCell::MVector{D, Int}
    Strides::MVector{D, Int}
end

function InitializeCellIndexLookup(::Val{D}) where D
    zeros_tuple = ntuple(_ -> 0, D)
    return CellIndexLookup{D}(
        Int[],
        MVector{D, Int}(zeros_tuple),
        MVector{D, Int}(zeros_tuple),
        MVector{D, Int}(zeros_tuple),
    )
end

@inline function PrepareCellLookup!(CellLookup::CellIndexLookup{D}) where D
    total = 1
    @inbounds for d in 1:D
        dim = CellLookup.MaxCell[d] - CellLookup.MinCell[d] + 1
        CellLookup.Strides[d] = total
        total *= dim
    end
    resize!(CellLookup.Grid, total)
    fill!(CellLookup.Grid, 0)
    return nothing
end

@inline function CellLinearIndex(CellLookup::CellIndexLookup{D}, Cell::CartesianIndex{D}) where D
    idx = 1
    @inbounds for d in 1:D
        idx += (Cell[d] - CellLookup.MinCell[d]) * CellLookup.Strides[d]
    end
    return idx
end

@inline function CellLookupIndex(CellLookup::CellIndexLookup{D}, Cell::CartesianIndex{D}, Default) where D
    @inbounds for d in 1:D
        value = Cell[d]
        if value < CellLookup.MinCell[d] || value > CellLookup.MaxCell[d]
            return Default
        end
    end
    return CellLookup.Grid[CellLinearIndex(CellLookup, Cell)]
end

function ConstructStencil(V::Val{d}) where d
    return CartesianIndices(ntuple(_ -> -1:1, V))
end

function BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView, ParticleRanges, CellLookup)
    TargetLen   = length(UniqueCellsView)
    OriginalLen = length(NeighborCellLists)
    resize!(NeighborCellLists, TargetLen)

    if TargetLen > OriginalLen
        @inbounds for Index in (OriginalLen + 1):TargetLen
            NeighborCellLists[Index] = Int[]
        end
    end

    @inbounds for CellIndex in eachindex(UniqueCellsView)
        Neighbors = NeighborCellLists[CellIndex]
        empty!(Neighbors)
        sizehint!(Neighbors, length(FullStencil) - 1)
        Cell = UniqueCellsView[CellIndex]
        for Offset in FullStencil
            NeighborCell = Cell + Offset
            NeighborIndex = CellLookupIndex(CellLookup, NeighborCell, 0)
            if NeighborIndex != 0 && NeighborIndex != CellIndex
                StartIndex = ParticleRanges[NeighborIndex]
                EndIndex = ParticleRanges[NeighborIndex + 1] - 1
                if StartIndex <= EndIndex
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

@inline function ExtractCellsAndBounds!(Particles, InverseCutOff, MinCell, MaxCell)
    Cells = Particles.Cells
    Positions = Particles.Position
    if isempty(Cells)
        return false
    end

    FirstCell = CartesianIndex(map(X -> MapFloor(X, InverseCutOff), Tuple(Positions[1])))
    Cells[1] = FirstCell
    @inbounds for d in eachindex(MinCell)
        MinCell[d] = FirstCell[d]
        MaxCell[d] = FirstCell[d]
    end

    if length(Cells) > 1
        @inbounds @simd ivdep for Index in 2:length(Cells)
            Cell = CartesianIndex(map(X -> MapFloor(X, InverseCutOff), Tuple(Positions[Index])))
            Cells[Index] = Cell
            @inbounds for d in eachindex(MinCell)
                value = Cell[d]
                if value < MinCell[d]
                    MinCell[d] = value
                elseif value > MaxCell[d]
                    MaxCell[d] = value
                end
            end
        end
    end

    return true
end

"""
Updates the neighbor list without sorting particle storage.

This builds a per-cell particle ordering buffer so cell ranges can be iterated
without reordering the particle arrays.
"""
function UpdateNeighbors!(Particles, InverseCutOff, ParticleRanges,
                          UniqueCells, CellLookup, ParticleOrder, CellOffsets)
    HasParticles = ExtractCellsAndBounds!(Particles, InverseCutOff, CellLookup.MinCell, CellLookup.MaxCell)
    if !HasParticles
        return 0
    end
    PrepareCellLookup!(CellLookup)

    Cells = @views Particles.Cells
    ParticleRanges[1] = 1
    IndexCounter = 1
    fill!(CellOffsets, zero(eltype(CellOffsets)))

    @inbounds for Index in eachindex(Cells)
        Cell = Cells[Index]
        LinearIndex = CellLinearIndex(CellLookup, Cell)
        CellIndex = CellLookup.Grid[LinearIndex]
        if CellIndex == 0
            IndexCounter += 1
            CellIndex = IndexCounter
            CellLookup.Grid[LinearIndex] = CellIndex
            UniqueCells[CellIndex] = Cell
        end
        CellOffsets[CellIndex] += 1
    end

    RunningIndex = 1
    @inbounds for CellIndex in 2:IndexCounter
        Count = CellOffsets[CellIndex]
        ParticleRanges[CellIndex] = RunningIndex
        CellOffsets[CellIndex] = RunningIndex
        RunningIndex += Count
    end
    ParticleRanges[IndexCounter + 1] = RunningIndex

    @inbounds for Index in eachindex(Cells)
        CellIndex = CellLookup.Grid[CellLinearIndex(CellLookup, Cells[Index])]
        TargetIndex = CellOffsets[CellIndex]
        ParticleOrder[TargetIndex] = Index
        CellOffsets[CellIndex] = TargetIndex + 1
    end

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
