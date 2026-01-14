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

    SimMetaDataWedge  = SimulationMetaData{Dimensions, FloatType, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(
        SimulationName="StillWedge",
        SaveLocation="W:/Simulations/StillWedge2D_MDBC_CUDA",
        SimulationTime=4.0f0,
        OutputTimes=0.01f0,
        VisualizeInParaview=true,
        ExportSingleVTKHDF=true,
        ExportGridCells=true,
        OpenLogFile=true,
    )

    if !isdir(SimMetaDataWedge.SaveLocation)
        mkdir(SimMetaDataWedge.SaveLocation)
    end

    SimulationGeometry = [FixedBoundary; Water]
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaDataWedge)

    SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); dx = SimConstantsWedge.dx)
    SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation)

    CleanUpSimulationFolder(SimMetaDataWedge.SaveLocation)

    PositionGPU = CuArray(SimParticles.Position)

    ParticleRanges = zeros(Int, length(SimParticles) + 2)
    UniqueCells = zeros(CartesianIndex{Dimensions}, length(SimParticles))
    CellDict = Dict{CartesianIndex{Dimensions}, Int}()
    FullStencil = ConstructStencil(Val(Dimensions))
    NeighborCellLists = [Int[] for _ in 1:length(UniqueCells)]
    ParticleOrder = zeros(Int, length(SimParticles))
    CellOffsets = zeros(Int, length(ParticleRanges))
    CellIdsHost = zeros(Int, length(SimParticles))

    IndexCounter = UpdateNeighbors!(SimParticles, SimKernel.H⁻¹, ParticleRanges, UniqueCells, CellDict, ParticleOrder, CellOffsets)
    UniqueCellsView = view(UniqueCells, 1:IndexCounter)
    BuildNeighborCellLists!(NeighborCellLists, FullStencil, UniqueCellsView, ParticleRanges, CellDict)
    @inbounds for Index in eachindex(CellIdsHost, SimParticles.Cells)
        CellIdsHost[Index] = CellDict[SimParticles.Cells[Index]]
    end
    NeighborListPacked = UpdateNeighborsCPUToCUDA!(
        nothing,
        CellIdsHost,
        ParticleOrder,
        view(ParticleRanges, 1:(IndexCounter + 1)),
        view(NeighborCellLists, 1:IndexCounter),
    )

    NeighborCounts = CountNeighborsCUDA(PositionGPU, NeighborListPacked, SimKernel.H²)

    NeighborCountsHost = Array(NeighborCounts)
    MinimumNeighbors = minimum(NeighborCountsHost)
    MaximumNeighbors = maximum(NeighborCountsHost)
    AverageNeighbors = sum(NeighborCountsHost) / length(NeighborCountsHost)

    println("StillWedge CUDA neighbor stats (within H):")
    println("  Min neighbors: ", MinimumNeighbors)
    println("  Max neighbors: ", MaximumNeighbors)
    println("  Avg neighbors: ", AverageNeighbors)

    RunSimulationCUDA(
        SimGeometry         = SimulationGeometry,
        SimMetaData         = SimMetaDataWedge,
        SimConstants        = SimConstantsWedge,
        SimKernel           = SimKernel,
        SimLogger           = SimLogger,
        SimParticles        = SimParticles,
        SimViscosity        = ArtificialViscosity(),
        SimDensityDiffusion = LinearDensityDiffusion(),
    )
end
