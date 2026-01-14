module SPHCUDANeighborList

export CUDACellGrid, CUDANeighborList, AllocateCUDANeighborList,
       ConstructNeighborOffsets, BuildNeighborCellListsCUDA!,
       UpdateNeighborsCUDA!, NeighborLoopCUDA!

using CUDA
using StaticArrays

"""
    CUDACellGrid(Origin, CellSize, Dims)

Describe a uniform cell grid for CUDA neighbor list construction.
"""
struct CUDACellGrid{D, T}
    Origin::SVector{D, T}
    InvCellSize::T
    Dims::SVector{D, Int}
    Strides::SVector{D, Int}
    CellCount::Int
end

"""
    CUDACellGrid(Origin, CellSize, Dims)

Build a CUDA cell grid from an origin, cell size, and grid dimensions.
"""
function CUDACellGrid(Origin::SVector{D, T}, CellSize::T, Dims::SVector{D, Int}) where {D, T}
    Strides = SVector{D, Int}(ntuple(i -> i == 1 ? 1 : prod(Dims[1:(i - 1)]), D))
    CellCount = prod(Dims)
    return CUDACellGrid{D, T}(Origin, inv(CellSize), Dims, Strides, CellCount)
end

"""
    CUDANeighborList(Grid, ...)

CUDA storage for per-cell particle ranges and neighbor cell indices.
"""
struct CUDANeighborList{D, T}
    Grid::CUDACellGrid{D, T}
    CellIds::CuArray{Int}
    ParticleOrder::CuArray{Int}
    ParticleRanges::CuArray{Int}
    CellCounts::CuArray{Int}
    CellOffsets::CuArray{Int}
    NeighborCells::CuArray{Int}
    NeighborCounts::CuArray{Int}
    NeighborOffsets::CuArray{SVector{D, Int}}
end

"""
    ConstructNeighborOffsets(Val(D))

Return the non-zero stencil offsets for a D-dimensional Moore neighborhood.
"""
function ConstructNeighborOffsets(::Val{D}) where {D}
    Offsets = SVector{D, Int}[]
    for Offset in CartesianIndices(ntuple(_ -> -1:1, D))
        if !all(iszero, Tuple(Offset))
            push!(Offsets, SVector{D, Int}(Tuple(Offset)))
        end
    end
    return Offsets
end

"""
    AllocateCUDANeighborList(Grid, ParticleCount)

Allocate GPU buffers for neighbor cell lists and particle ordering.
"""
function AllocateCUDANeighborList(Grid::CUDACellGrid{D, T}, ParticleCount::Int) where {D, T}
    CellIds = CUDA.zeros(Int, ParticleCount)
    ParticleOrder = CUDA.zeros(Int, ParticleCount)
    ParticleRanges = CUDA.zeros(Int, Grid.CellCount + 1)
    CellCounts = CUDA.zeros(Int, Grid.CellCount)
    CellOffsets = CUDA.zeros(Int, Grid.CellCount)
    NeighborOffsets = CuArray(ConstructNeighborOffsets(Val(D)))
    NeighborCells = CUDA.zeros(Int, length(NeighborOffsets), Grid.CellCount)
    NeighborCounts = CUDA.zeros(Int, Grid.CellCount)
    NeighborList = CUDANeighborList{D, T}(
        Grid,
        CellIds,
        ParticleOrder,
        ParticleRanges,
        CellCounts,
        CellOffsets,
        NeighborCells,
        NeighborCounts,
        NeighborOffsets,
    )
    BuildNeighborCellListsCUDA!(NeighborList)
    return NeighborList
end

@inline function LinearCellId(CellCoords::SVector{D, Int}, Grid::CUDACellGrid{D, T}) where {D, T}
    CellId = 1
    @inbounds for DimIndex in 1:D
        CellId += CellCoords[DimIndex] * Grid.Strides[DimIndex]
    end
    return CellId
end

@generated function CellCoordsFromId(CellId::Int, Grid::CUDACellGrid{D, T}) where {D, T}
    coord_symbols = [Symbol(:Coord_, i) for i in 1:D]
    coord_defs = map(enumerate(coord_symbols)) do (i, sym)
        quote
            Dim = Grid.Dims[$i]
            $sym = Remaining % Dim
            Remaining = Remaining ÷ Dim
        end
    end
    return quote
        Remaining = CellId - 1
        $(coord_defs...)
        return SVector{$D, Int}($(coord_symbols...))
    end
end

@generated function CellCoordsFromPosition(Position, Grid::CUDACellGrid{D, T}) where {D, T}
    coord_exprs = map(1:D) do i
        quote
            Raw = (Position[$i] - Grid.Origin[$i]) * Grid.InvCellSize
            Cell = floor(Int, Raw)
            clamp(Cell, 0, Grid.Dims[$i] - 1)
        end
    end
    return quote
        return SVector{$D, Int}($(coord_exprs...))
    end
end

function ComputeCellIdsKernel!(CellIds, Position, Grid::CUDACellGrid{D, T}) where {D, T}
    Index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if Index <= length(CellIds)
        Pos = Position[Index]
        Coords = CellCoordsFromPosition(Pos, Grid)
        CellIds[Index] = LinearCellId(Coords, Grid)
    end
    return nothing
end

function ComputeCellCountsKernel!(CellCounts, CellIds)
    Index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if Index <= length(CellIds)
        CellId = CellIds[Index]
        CUDA.atomic_add!(CellCounts, CellId, 1)
    end
    return nothing
end

function BuildParticleRangesKernel!(ParticleRanges, CellPrefix)
    Index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if Index <= length(CellPrefix)
        if Index == 1
            ParticleRanges[1] = 1
        end
        ParticleRanges[Index + 1] = CellPrefix[Index] + 1
    end
    return nothing
end

function BuildParticleOrderKernel!(ParticleOrder, CellOffsets, CellIds)
    Index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if Index <= length(CellIds)
        CellId = CellIds[Index]
        TargetIndex = CUDA.atomic_add!(CellOffsets, CellId, 1)
        ParticleOrder[TargetIndex] = Index
    end
    return nothing
end

function BuildNeighborCellListsKernel!(NeighborCells, NeighborCounts, NeighborOffsets, Grid::CUDACellGrid{D, T}) where {D, T}
    CellId = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if CellId <= Grid.CellCount
        CellCoord = CellCoordsFromId(CellId, Grid)
        NeighborIndex = 0
        for OffsetIndex in 1:length(NeighborOffsets)
            Offset = NeighborOffsets[OffsetIndex]
            NeighborCoord = CellCoord + Offset
            IsValid = true
            @inbounds for DimIndex in 1:D
                if NeighborCoord[DimIndex] < 0 || NeighborCoord[DimIndex] >= Grid.Dims[DimIndex]
                    IsValid = false
                    break
                end
            end
            if IsValid
                NeighborIndex += 1
                NeighborCells[NeighborIndex, CellId] = LinearCellId(NeighborCoord, Grid)
            end
        end
        NeighborCounts[CellId] = NeighborIndex
    end
    return nothing
end

"""
    BuildNeighborCellListsCUDA!(NeighborList)

Precompute the neighbor cell indices for each grid cell on the GPU.
"""
function BuildNeighborCellListsCUDA!(NeighborList::CUDANeighborList)
    Threads = 256
    Blocks = cld(NeighborList.Grid.CellCount, Threads)
    CUDA.@cuda threads=Threads blocks=Blocks BuildNeighborCellListsKernel!(
        NeighborList.NeighborCells,
        NeighborList.NeighborCounts,
        NeighborList.NeighborOffsets,
        NeighborList.Grid,
    )
    return nothing
end

"""
    UpdateNeighborsCUDA!(NeighborList, Position)

Update cell ids, particle ranges, and particle order on the GPU.
"""
function UpdateNeighborsCUDA!(NeighborList::CUDANeighborList, Position)
    Threads = 256
    Blocks = cld(length(NeighborList.CellIds), Threads)
    CUDA.@cuda threads=Threads blocks=Blocks ComputeCellIdsKernel!(
        NeighborList.CellIds,
        Position,
        NeighborList.Grid,
    )
    fill!(NeighborList.CellCounts, 0)
    CUDA.@cuda threads=Threads blocks=Blocks ComputeCellCountsKernel!(
        NeighborList.CellCounts,
        NeighborList.CellIds,
    )
    CellPrefix = CUDA.cumsum(NeighborList.CellCounts)
    BlocksCells = cld(length(CellPrefix), Threads)
    CUDA.@cuda threads=Threads blocks=BlocksCells BuildParticleRangesKernel!(
        NeighborList.ParticleRanges,
        CellPrefix,
    )
    copyto!(NeighborList.CellOffsets, NeighborList.ParticleRanges[1:(end - 1)])
    CUDA.@cuda threads=Threads blocks=Blocks BuildParticleOrderKernel!(
        NeighborList.ParticleOrder,
        NeighborList.CellOffsets,
        NeighborList.CellIds,
    )
    return nothing
end

function NeighborLoopKernel!(InteractionKernel,
                             CellIds,
                             ParticleOrder,
                             ParticleRanges,
                             NeighborCells,
                             NeighborCounts,
                             Args...)
    Index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if Index <= length(CellIds)
        CellId = CellIds[Index]
        StartIndex = ParticleRanges[CellId]
        EndIndex = ParticleRanges[CellId + 1] - 1
        for j in StartIndex:EndIndex
            NeighborIndex = ParticleOrder[j]
            if NeighborIndex != Index
                InteractionKernel(Index, NeighborIndex, Args...)
            end
        end
        NeighborCount = NeighborCounts[CellId]
        for n in 1:NeighborCount
            NeighborCell = NeighborCells[n, CellId]
            StartIndex = ParticleRanges[NeighborCell]
            EndIndex = ParticleRanges[NeighborCell + 1] - 1
            for j in StartIndex:EndIndex
                NeighborIndex = ParticleOrder[j]
                InteractionKernel(Index, NeighborIndex, Args...)
            end
        end
    end
    return nothing
end

"""
    NeighborLoopCUDA!(InteractionKernel, NeighborList, Args...; Threads=256)

Launch a CUDA neighbor loop that calls `InteractionKernel(i, j, Args...)`.
"""
function NeighborLoopCUDA!(InteractionKernel,
                           NeighborList::CUDANeighborList,
                           Args...;
                           Threads::Int = 256)
    Blocks = cld(length(NeighborList.CellIds), Threads)
    CUDA.@cuda threads=Threads blocks=Blocks NeighborLoopKernel!(
        InteractionKernel,
        NeighborList.CellIds,
        NeighborList.ParticleOrder,
        NeighborList.ParticleRanges,
        NeighborList.NeighborCells,
        NeighborList.NeighborCounts,
        Args...,
    )
    return nothing
end

end
