module SPHCUDANeighborList

export CUDACellGrid, CUDANeighborList, AllocateCUDANeighborList,
       ConstructNeighborOffsets, BuildNeighborCellListsCUDA!,
       UpdateNeighborsCUDA!, NeighborLoopCUDA!,
       NeighborLoopPerParticleCUDA!

const CUDAIndex = Int32

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
    CellIds::CuArray{CUDAIndex}
    ParticleOrder::CuArray{CUDAIndex}
    ParticleRanges::CuArray{CUDAIndex}
    CellCounts::CuArray{CUDAIndex}
    CellOffsets::CuArray{CUDAIndex}
    NeighborCells::CuArray{CUDAIndex}
    NeighborCounts::CuArray{CUDAIndex}
    NeighborOffsets::CuArray{SVector{D, CUDAIndex}}
end

"""
    ConstructNeighborOffsets(Val(D))

Return the non-zero stencil offsets for a D-dimensional Moore neighborhood.
"""
function ConstructNeighborOffsets(::Val{D}) where {D}
    Offsets = SVector{D, CUDAIndex}[]
    for Offset in CartesianIndices(ntuple(_ -> -1:1, D))
        if !all(iszero, Tuple(Offset))
            push!(Offsets, SVector{D, CUDAIndex}(Tuple(Offset)))
        end
    end
    return Offsets
end

"""
    AllocateCUDANeighborList(Grid, ParticleCount)

Allocate GPU buffers for neighbor cell lists and particle ordering.
"""
function AllocateCUDANeighborList(Grid::CUDACellGrid{D, T}, ParticleCount::Int) where {D, T}
    CellIds = CUDA.zeros(CUDAIndex, ParticleCount)
    ParticleOrder = CUDA.zeros(CUDAIndex, ParticleCount)
    ParticleRanges = CUDA.zeros(CUDAIndex, Grid.CellCount + 1)
    CellCounts = CUDA.zeros(CUDAIndex, Grid.CellCount)
    CellOffsets = CUDA.zeros(CUDAIndex, Grid.CellCount)
    NeighborOffsets = CuArray(ConstructNeighborOffsets(Val(D)))
    NeighborCells = CUDA.zeros(CUDAIndex, length(NeighborOffsets), Grid.CellCount)
    NeighborCounts = CUDA.zeros(CUDAIndex, Grid.CellCount)
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

@inline function LinearCellId(CellCoords::SVector{D, CUDAIndex}, Grid::CUDACellGrid{D, T}) where {D, T}
    CellId = CUDAIndex(1)
    @inbounds for DimIndex in 1:D
        CellId += CellCoords[DimIndex] * CUDAIndex(Grid.Strides[DimIndex])
    end
    return CellId
end

@generated function CellCoordsFromId(CellId::CUDAIndex, Grid::CUDACellGrid{D, T}) where {D, T}
    coord_symbols = [Symbol(:Coord_, i) for i in 1:D]
    coord_defs = map(enumerate(coord_symbols)) do (i, sym)
        quote
            Dim = CUDAIndex(Grid.Dims[$i])
            $sym = Remaining % Dim
            Remaining = Remaining ÷ Dim
        end
    end
    return quote
        Remaining = CellId - CUDAIndex(1)
        $(coord_defs...)
        return SVector{$D, CUDAIndex}($(coord_symbols...))
    end
end

@generated function CellCoordsFromPosition(Position, Grid::CUDACellGrid{D, T}) where {D, T}
    coord_exprs = map(1:D) do i
        quote
            Raw = (Position[$i] - Grid.Origin[$i]) * Grid.InvCellSize
            Cell = floor(CUDAIndex, Raw)
            clamp(Cell, CUDAIndex(0), CUDAIndex(Grid.Dims[$i] - 1))
        end
    end
    return quote
        return SVector{$D, CUDAIndex}($(coord_exprs...))
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
        CUDA.atomic_add!(CellCounts, CellId, CUDAIndex(1))
    end
    return nothing
end

function BuildParticleRangesKernel!(ParticleRanges, CellPrefix)
    Index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if Index <= length(CellPrefix)
        if Index == 1
            ParticleRanges[1] = CUDAIndex(1)
        end
        ParticleRanges[Index + 1] = CellPrefix[Index] + CUDAIndex(1)
    end
    return nothing
end

function BuildParticleOrderKernel!(ParticleOrder, CellOffsets, CellIds)
    Index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if Index <= length(CellIds)
        CellId = CellIds[Index]
        TargetIndex = CUDA.atomic_add!(CellOffsets, CellId, CUDAIndex(1))
        ParticleOrder[Int(TargetIndex)] = CUDAIndex(Index)
    end
    return nothing
end

function BuildNeighborCellListsKernel!(NeighborCells, NeighborCounts, NeighborOffsets, Grid::CUDACellGrid{D, T}) where {D, T}
    CellId = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if CellId <= Grid.CellCount
        CellCoord = CellCoordsFromId(CUDAIndex(CellId), Grid)
        NeighborIndex = CUDAIndex(0)
        for OffsetIndex in 1:length(NeighborOffsets)
            Offset = NeighborOffsets[OffsetIndex]
            NeighborCoord = CellCoord + Offset
            IsValid = true
            @inbounds for DimIndex in 1:D
                if NeighborCoord[DimIndex] < CUDAIndex(0) || NeighborCoord[DimIndex] >= CUDAIndex(Grid.Dims[DimIndex])
                    IsValid = false
                    break
                end
            end
            if IsValid
                NeighborIndex += CUDAIndex(1)
                NeighborCells[Int(NeighborIndex), CellId] = LinearCellId(NeighborCoord, Grid)
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
        CellId = Int(CellIds[Index])
        StartIndex = Int(ParticleRanges[CellId])
        EndIndex = Int(ParticleRanges[CellId + 1]) - 1
        for j in StartIndex:EndIndex
            NeighborIndex = Int(ParticleOrder[j])
            if NeighborIndex != Index
                InteractionKernel(Index, NeighborIndex, Args...)
            end
        end
        NeighborCount = Int(NeighborCounts[CellId])
        for n in 1:NeighborCount
            NeighborCell = Int(NeighborCells[n, CellId])
            StartIndex = Int(ParticleRanges[NeighborCell])
            EndIndex = Int(ParticleRanges[NeighborCell + 1]) - 1
            for j in StartIndex:EndIndex
                NeighborIndex = Int(ParticleOrder[j])
                InteractionKernel(Index, NeighborIndex, Args...)
            end
        end
    end
    return nothing
end

function NeighborLoopPerParticleKernel!(InitKernel,
                                        InteractionKernel,
                                        FinalKernel,
                                        CellIds,
                                        ParticleOrder,
                                        ParticleRanges,
                                        NeighborCells,
                                        NeighborCounts,
                                        Args...)
    Index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if Index <= length(CellIds)
        CellId = Int(CellIds[Index])
        Accumulator = InitKernel(Index, Args...)
        StartIndex = Int(ParticleRanges[CellId])
        EndIndex = Int(ParticleRanges[CellId + 1]) - 1
        for j in StartIndex:EndIndex
            NeighborIndex = Int(ParticleOrder[j])
            if NeighborIndex != Index
                Accumulator = InteractionKernel(Index, NeighborIndex, Accumulator, Args...)
            end
        end
        NeighborCount = Int(NeighborCounts[CellId])
        for n in 1:NeighborCount
            NeighborCell = Int(NeighborCells[n, CellId])
            StartIndex = Int(ParticleRanges[NeighborCell])
            EndIndex = Int(ParticleRanges[NeighborCell + 1]) - 1
            for j in StartIndex:EndIndex
                NeighborIndex = Int(ParticleOrder[j])
                Accumulator = InteractionKernel(Index, NeighborIndex, Accumulator, Args...)
            end
        end
        FinalKernel(Index, Accumulator, Args...)
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

"""
    NeighborLoopPerParticleCUDA!(InitKernel, InteractionKernel, FinalKernel, NeighborList, Args...; Threads=256)

Launch a CUDA neighbor loop that accumulates per-particle state. The kernels
are called as:

- `InitKernel(i, Args...)` to initialize the per-particle accumulator.
- `InteractionKernel(i, j, Accumulator, Args...)` for each neighbor, returning an updated accumulator.
- `FinalKernel(i, Accumulator, Args...)` to store the final per-particle state.
"""
function NeighborLoopPerParticleCUDA!(InitKernel,
                                      InteractionKernel,
                                      FinalKernel,
                                      NeighborList::CUDANeighborList,
                                      Args...;
                                      Threads::Int = 256)
    Blocks = cld(length(NeighborList.CellIds), Threads)
    CUDA.@cuda threads=Threads blocks=Blocks NeighborLoopPerParticleKernel!(
        InitKernel,
        InteractionKernel,
        FinalKernel,
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
