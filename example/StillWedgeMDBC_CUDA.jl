using SPHExample
using CUDA
using StaticArrays

if !SPHExample.HasCUDA
    error("CUDA.jl is required to run this example.")
end

@inline function ComputeBounds(Position)
    MinCorner = reduce((A, B) -> min.(A, B), Position)
    MaxCorner = reduce((A, B) -> max.(A, B), Position)
    return MinCorner, MaxCorner
end

@inline function BuildGrid(MinCorner, MaxCorner, CellSize)
    Extents = MaxCorner - MinCorner
    Dims = SVector{length(MinCorner), Int}(ceil.(Int, Extents ./ CellSize) .+ 1)
    return CUDACellGrid(MinCorner, CellSize, Dims)
end

@inline function SquaredNorm(Vector)
    Accumulator = zero(eltype(Vector))
    @inbounds for Index in eachindex(Vector)
        Accumulator += Vector[Index] * Vector[Index]
    end
    return Accumulator
end

@inline function NeighborCountInteractionKernel(Index, NeighborIndex, Counts, Position, SupportRadiusSquared)
    Δx = Position[Index] - Position[NeighborIndex]
    if SquaredNorm(Δx) <= SupportRadiusSquared
        CUDA.atomic_add!(Counts, Index, 1)
    end
    return nothing
end

function CountNeighborsCUDA(Position, NeighborList, SupportRadiusSquared)
    Counts = CUDA.zeros(Int, length(Position))
    NeighborLoopCUDA!(NeighborCountInteractionKernel, NeighborList, Counts, Position, SupportRadiusSquared)
    return Counts
end

let
    Dimensions = 2
    FloatType  = Float32

    SimConstantsWedge = SimulationConstants{FloatType}(dx=0.02f0, c₀=42.485762f0, δᵩ=0.1f0, CFL=0.5f0)

    FixedBoundary = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Bound.csv",
        GroupMarker = 1,
        Type        = Fixed,
        Motion      = nothing,
    )

    Water = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Fluid.csv",
        GroupMarker = 2,
        Type        = Fluid,
        Motion      = nothing,
    )

    SimulationGeometry = [FixedBoundary; Water]
    SimParticles = AllocateDataStructures(SimulationGeometry)

    SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); dx = SimConstantsWedge.dx)

    PositionGPU = CuArray(SimParticles.Position)
    MinCorner, MaxCorner = ComputeBounds(SimParticles.Position)
    Grid = BuildGrid(MinCorner, MaxCorner, SimKernel.H)
    NeighborList = AllocateCUDANeighborList(Grid, length(PositionGPU))

    UpdateNeighborsCUDA!(NeighborList, PositionGPU)
    NeighborCounts = CountNeighborsCUDA(PositionGPU, NeighborList, SimKernel.H²)

    NeighborCountsHost = Array(NeighborCounts)
    MinimumNeighbors = minimum(NeighborCountsHost)
    MaximumNeighbors = maximum(NeighborCountsHost)
    AverageNeighbors = sum(NeighborCountsHost) / length(NeighborCountsHost)

    println("StillWedge CUDA neighbor stats (within H):")
    println("  Min neighbors: ", MinimumNeighbors)
    println("  Max neighbors: ", MaximumNeighbors)
    println("  Avg neighbors: ", AverageNeighbors)
end
