module SPHNeighborList

export ConstructStencil, ExtractCells!, UpdateNeighbors!,
       BuildNeighborCellLists!, ComputeCellParticleCounts, ComputeCellNeighborCounts, UpdateΔx!,
       CellLookup, InitializeCellLookup, ResetCellLookup!, GetCellIndex, SetCellIndex!

using StaticArrays

mutable struct CellLookup{D}
    Keys::Vector{CartesianIndex{D}}
    Values::Vector{Int}
    Filled::Vector{Bool}
    Mask::Int
end

@inline function ZeroCell(::Val{D}) where D
    return CartesianIndex(ntuple(_ -> 0, Val(D)))
end

@inline function NextPow2(n::Int)
    size = 1
    while size < n
        size <<= 1
    end
    return size
end

function InitializeCellLookup(::Val{D}, capacity::Int) where D
    size = NextPow2(max(16, capacity))
    zero_cell = ZeroCell(Val(D))
    return CellLookup{D}(fill(zero_cell, size), zeros(Int, size), fill(false, size), size - 1)
end

function ResetCellLookup!(Lookup::CellLookup{D}, capacity::Int) where D
    target = max(16, capacity)
    if length(Lookup.Keys) < target
        size = NextPow2(target)
        zero_cell = ZeroCell(Val(D))
        Lookup.Keys = fill(zero_cell, size)
        Lookup.Values = zeros(Int, size)
        Lookup.Filled = fill(false, size)
        Lookup.Mask = size - 1
    else
        fill!(Lookup.Filled, false)
    end
    return nothing
end

@inline function HashCellIndex(Cell::CartesianIndex{D}) where D
    h = UInt(0x9e3779b97f4a7c15)
    @inbounds for i in 1:D
        v = unsigned(Cell.I[i])
        h ⊻= v + 0x9e3779b97f4a7c15 + (h << 6) + (h >> 2)
    end
    return h
end

@inline function GetCellIndex(Lookup::CellLookup{D}, Cell::CartesianIndex{D}, default::Int) where D
    mask = Lookup.Mask
    index = Int(HashCellIndex(Cell) & UInt(mask))
    @inbounds for _ in 0:mask
        slot = index + 1
        if !Lookup.Filled[slot]
            return default
        elseif Lookup.Keys[slot] == Cell
            return Lookup.Values[slot]
        end
        index = (index + 1) & mask
    end
    return default
end

@inline function SetCellIndex!(Lookup::CellLookup{D}, Cell::CartesianIndex{D}, value::Int) where D
    mask = Lookup.Mask
    index = Int(HashCellIndex(Cell) & UInt(mask))
    @inbounds for _ in 0:mask
        slot = index + 1
        if !Lookup.Filled[slot]
            Lookup.Filled[slot] = true
            Lookup.Keys[slot] = Cell
            Lookup.Values[slot] = value
            return nothing
        elseif Lookup.Keys[slot] == Cell
            Lookup.Values[slot] = value
            return nothing
        end
        index = (index + 1) & mask
    end
    return nothing
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
            NeighborIndex = GetCellIndex(CellLookup, NeighborCell, 0)
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

"""
Updates the neighbor list without sorting particle storage.

This builds a per-cell particle ordering buffer so cell ranges can be iterated
without reordering the particle arrays.
"""
function UpdateNeighbors!(Particles, InverseCutOff, ParticleRanges,
                          UniqueCells, CellLookup, ParticleOrder, CellOffsets)
    ExtractCells!(Particles, InverseCutOff)

    Cells = @views Particles.Cells
    ParticleRanges[1] = 1
    IndexCounter = 1
    ResetCellLookup!(CellLookup, length(Cells) * 2)
    fill!(CellOffsets, zero(eltype(CellOffsets)))

    @inbounds for Index in eachindex(Cells)
        Cell = Cells[Index]
        CellIndex = GetCellIndex(CellLookup, Cell, 0)
        if CellIndex == 0
            IndexCounter += 1
            CellIndex = IndexCounter
            SetCellIndex!(CellLookup, Cell, CellIndex)
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
        CellIndex = GetCellIndex(CellLookup, Cells[Index], 0)
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
