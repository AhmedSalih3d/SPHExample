using Test
using HDF5
using SPHExample
using StaticArrays
using StructArrays
using TimerOutputs

@testset "time stepping" begin
    pos = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 0.0)]
    vel = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(0.0, 0.0)]
    acc = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(0.0, -9.81)]
    sc = SimulationConstants{Float64}()
    ker = SPHKernelInstance{2, Float64}(WendlandC2(); dx=sc.dx)
    max_acceleration = maximum(a -> sqrt(sum(abs2, a)), acc)
    dt  = Δt(max_acceleration, sc, ker)
    @test dt > 0
    alloc = @allocated Δt(max_acceleration, sc, ker)
    @test alloc == 0
end

@testset "gravity does not mutate carried acceleration" begin
    D = 2
    T = Float64
    MetaData = SimulationMetaData{D, T}(
        SimulationName="gravity_state",
        SaveLocation=".",
    )
    Constants = SimulationConstants{T}(g=9.81)
    Kernel = SPHKernelInstance{D, T}(WendlandC2(); dx=Constants.dx)
    HydrodynamicAcceleration = SVector{D, T}(0.25, -0.5)
    Particles = StructArray((
        Position=[zero(SVector{D, T})],
        Velocity=[zero(SVector{D, T})],
        Acceleration=[HydrodynamicAcceleration],
        Density=T[Constants.ρ₀],
        Type=ParticleType[Fluid],
    ))
    Positionₙ⁺ = similar(Particles.Position)
    Velocityₙ⁺ = similar(Particles.Velocity)
    ρₙ⁺ = similar(Particles.Density)
    dρdtI = zeros(T, 1)
    dt = T(0.01)

    SPHExample.TimeStepping.HalfTimeStep(
        MetaData, Constants, Particles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt / 2,
    )
    FirstPredictorVelocity = only(Velocityₙ⁺)
    @test only(Particles.Acceleration) == HydrodynamicAcceleration

    SPHExample.TimeStepping.HalfTimeStep(
        MetaData, Constants, Particles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt / 2,
    )
    @test only(Velocityₙ⁺) == FirstPredictorVelocity
    @test only(Particles.Acceleration) == HydrodynamicAcceleration

    SPHExample.TimeStepping.FullTimeStep(
        MetaData, Kernel, Constants, Particles, Velocityₙ⁺, SVector{D, T}[], T[], dt,
    )
    @test only(Particles.Acceleration) == HydrodynamicAcceleration
end

function RunOutputCadenceCase(OutputTimes; SimulationEnd=1 // 4,
                              Densities=nothing, Velocities=nothing,
                              Gravity=0, TimeStepping=SingleNeighborTimeStepping())
    D = 2
    T = Float64
    dx = T(1 // 32)
    SimulationEnd = T(SimulationEnd)
    Constants = SimulationConstants{T}(dx=dx, c₀=one(T), CFL=T(1 // 2), g=T(Gravity))
    Kernel = SPHKernelInstance{D, T}(WendlandC2(); dx=dx)
    MetaData = SimulationMetaData{D, T}(
        SimulationName="output_cadence",
        SaveLocation=".",
        SimulationTime=SimulationEnd,
        OutputTimes=OutputTimes isa AbstractVector ? T.(OutputTimes) : T(OutputTimes),
        VisualizeInParaview=false,
        OpenLogFile=false,
        TimeSteppingMode=TimeStepping,
    )

    Positions = SVector{D, T}[
        (0, 0),
        (dx, 0),
        (0, dx),
        (dx, dx),
    ]
    InitialVelocities = Velocities === nothing ? SVector{D, T}[
        (0.01, 0),
        (-0.005, 0.002),
        (0.001, -0.003),
        (-0.002, 0.004),
    ] : SVector{D, T}.(Velocities)
    InitialDensities = Densities === nothing ?
        T[1000, 1000.02, 999.98, 1000.01] : T.(Densities)
    Particles = StructArray((
        Cells=fill(CartesianIndex(0, 0), 4),
        Position=Positions,
        Acceleration=fill(zero(SVector{D, T}), 4),
        Velocity=InitialVelocities,
        Density=InitialDensities,
        Pressure=zeros(T, 4),
        ID=collect(1:4),
        Type=fill(Fluid, 4),
        GroupMarker=fill(UInt(1), 4),
    ))

    dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ =
        AllocateSupportDataStructures(MetaData, Particles.Position)
    ParticleRanges = zeros(Int, length(Particles) + 2)
    UniqueCells = zeros(CartesianIndex{D}, length(Particles) + 1)
    CellListIndices = zeros(Int, length(Particles))
    FullStencil = ConstructStencil(Val(D))
    NeighborCellLists = [Int[] for _ in eachindex(UniqueCells)]
    _, SortingScratchSpace = Base.Sort.make_scratch(
        nothing,
        eltype(Particles),
        length(Particles),
    )

    MetaData.OutputIterationCounter = 1
    MetaData.CurrentTimeStep = Constants.CFL * Kernel.h / Constants.c₀
    EmittedTimes = T[zero(T)]
    function RecordOutput!()
        push!(EmittedTimes, MetaData.TotalTime)
        return nothing
    end

    SPHExample.SPHCellList.SimulationLoop(
        ZeroDensityDiffusion(), ZeroViscosity(), Kernel, MetaData,
        Constants, Particles, FullStencil, ParticleRanges, UniqueCells,
        CellListIndices, SortingScratchSpace, NeighborCellLists, dρdtI,
        Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ, nothing, RecordOutput!,
    )

    Order = sortperm(Particles.ID)
    State = (
        Position=copy(Particles.Position[Order]),
        Velocity=copy(Particles.Velocity[Order]),
        Density=copy(Particles.Density[Order]),
        Pressure=copy(Particles.Pressure[Order]),
        Acceleration=copy(Particles.Acceleration[Order]),
    )
    return MetaData, State, EmittedTimes
end

@testset "output cadence does not restart integration" begin
    CoarseMetaData, CoarseState, CoarseOutputs = RunOutputCadenceCase(1 // 4)
    FineMetaData, FineState, FineOutputs = RunOutputCadenceCase(1 // 16)

    @test CoarseMetaData.Iteration == FineMetaData.Iteration == 8
    @test CoarseMetaData.TotalTime == FineMetaData.TotalTime == 1 / 4
    @test CoarseMetaData.TimeSteps == FineMetaData.TimeSteps == fill(1 / 32, 8)
    @test CoarseState == FineState
    @test CoarseOutputs == [0, 1 / 4]
    @test FineOutputs == [0, 1 / 16, 1 / 8, 3 / 16, 1 / 4]
    @test TimerOutputs.ncalls(CoarseMetaData.HourGlass["00 Initialize Neighbor Data"]) == 1
    @test TimerOutputs.ncalls(FineMetaData.HourGlass["00 Initialize Neighbor Data"]) == 1

    SymplecticCoarseMetaData, SymplecticCoarseState, _ = RunOutputCadenceCase(
        1 // 4;
        TimeStepping=SymplecticTimeStepping(),
    )
    SymplecticFineMetaData, SymplecticFineState, _ = RunOutputCadenceCase(
        1 // 16;
        TimeStepping=SymplecticTimeStepping(),
    )
    @test SymplecticCoarseMetaData.TimeSteps == SymplecticFineMetaData.TimeSteps
    @test SymplecticCoarseState == SymplecticFineState

    GravityCoarseMetaData, GravityCoarseState, _ = RunOutputCadenceCase(
        1 // 4;
        Gravity=9.81,
    )
    GravityFineMetaData, GravityFineState, _ = RunOutputCadenceCase(
        1 // 16;
        Gravity=9.81,
    )
    @test GravityCoarseMetaData.TimeSteps == GravityFineMetaData.TimeSteps
    @test GravityCoarseState == GravityFineState

    @test !SPHExample.SPHCellList.NeedsSingleNeighborCorrection(0, false)
    @test !SPHExample.SPHCellList.NeedsSingleNeighborCorrection(19, false)
    @test SPHExample.SPHCellList.NeedsSingleNeighborCorrection(20, false)
    @test !SPHExample.SPHCellList.NeedsSingleNeighborCorrection(20, true)
    @test !SPHExample.SPHCellList.NeedsSingleNeighborCorrection(21, false)
    @test SPHExample.SPHCellList.NeedsSingleNeighborCorrection(40, false)

    CorrectionCoarseMetaData, CorrectionCoarseState, _ = RunOutputCadenceCase(
        3 // 4;
        SimulationEnd=3 // 4,
    )
    CorrectionFineMetaData, CorrectionFineState, _ = RunOutputCadenceCase(
        1 // 20;
        SimulationEnd=3 // 4,
    )
    @test CorrectionCoarseMetaData.Iteration == CorrectionFineMetaData.Iteration
    @test CorrectionCoarseMetaData.Iteration >
          SPHExample.SPHCellList.SingleNeighborCorrectionInterval
    @test CorrectionCoarseMetaData.TimeSteps == CorrectionFineMetaData.TimeSteps
    @test CorrectionCoarseState == CorrectionFineState
    @test TimerOutputs.ncalls(
        CorrectionCoarseMetaData.HourGlass["00 Simulation Step"]["04 Periodic Single-Neighbor Correction"],
    ) == 1
    @test TimerOutputs.ncalls(
        CorrectionFineMetaData.HourGlass["00 Simulation Step"]["04 Periodic Single-Neighbor Correction"],
    ) == 1
    @test TimerOutputs.ncalls(
        CorrectionCoarseMetaData.HourGlass["00 Simulation Step"]["04 Periodic Single-Neighbor Correction"]["03 NeighborLoop"],
    ) == 1

    ScheduleMetaData = SimulationMetaData{2, Float64}(
        SimulationName="output_schedule",
        SaveLocation=".",
        SimulationTime=0.25,
        OutputTimes=[0.1, 0.2],
        OutputIterationCounter=1,
    )
    @test SPHExample.TimeStepping.next_output_time(ScheduleMetaData) == 0.1
    ScheduleMetaData.OutputIterationCounter = 2
    @test SPHExample.TimeStepping.next_output_time(ScheduleMetaData) == 0.2
    ScheduleMetaData.OutputIterationCounter = 3
    @test SPHExample.TimeStepping.next_output_time(ScheduleMetaData) == 0.25
    ScheduleMetaData.OutputTimes = [0.5]
    ScheduleMetaData.OutputIterationCounter = 1
    @test SPHExample.TimeStepping.next_output_time(ScheduleMetaData) == 0.25

    VectorMetaData, VectorState, VectorOutputs =
        RunOutputCadenceCase([0.07, 0.08])
    @test VectorMetaData.TotalTime == 1 / 4
    @test VectorMetaData.TimeSteps == CoarseMetaData.TimeSteps
    @test VectorState == CoarseState
    @test VectorOutputs == [0, 3 / 32, 3 / 32, 1 / 4]

    @test_throws ArgumentError SPHExample.SPHCellList.ValidateOutputSchedule(0.0)
    @test_throws ArgumentError SPHExample.SPHCellList.ValidateOutputSchedule(Inf)
    @test_throws ArgumentError SPHExample.SPHCellList.ValidateOutputSchedule([0.1, 0.1])
    @test_throws ArgumentError SPHExample.SPHCellList.ValidateOutputSchedule([0.1, NaN])

    AdaptiveDensities = [1000, 1700, 300, 1350]
    StationaryVelocities = fill(zero(SVector{2, Float64}), 4)
    AdaptiveCoarseMetaData, AdaptiveCoarseState, _ = RunOutputCadenceCase(
        0.1;
        SimulationEnd=0.1,
        Densities=AdaptiveDensities,
        Velocities=StationaryVelocities,
    )
    AdaptiveFineMetaData, AdaptiveFineState, _ = RunOutputCadenceCase(
        0.02;
        SimulationEnd=0.1,
        Densities=AdaptiveDensities,
        Velocities=StationaryVelocities,
    )
    @test AdaptiveCoarseMetaData.Iteration == AdaptiveFineMetaData.Iteration
    @test AdaptiveCoarseMetaData.TotalTime == AdaptiveFineMetaData.TotalTime == 0.1
    @test AdaptiveCoarseMetaData.TimeSteps == AdaptiveFineMetaData.TimeSteps
    @test AdaptiveCoarseState == AdaptiveFineState
end

function MakeVTKTestParticles(::Val{D}, ::Type{T}) where {D, T}
    Positions = [SVector{D, T}(ntuple(j -> T(10i + j), D)) for i in 1:3]
    Velocities = [SVector{D, T}(ntuple(j -> T(i + j / 10), D)) for i in 1:3]
    Cells = [CartesianIndex(ntuple(_ -> i, D)) for i in 1:3]
    Types = ParticleType[Fluid, Fixed, Moving]
    return StructArray((
        Cells = Cells,
        Position = Positions,
        Velocity = Velocities,
        Density = T[1001, 1002, 1003],
        BoundaryBool = UInt8[9, 9, 9],
        Type = Types,
    ))
end

function SetVTKTestFrame!(Particles, Frame, ::Val{D}, ::Type{T}) where {D, T}
    for i in eachindex(Particles.Position)
        Particles.Position[i] = SVector{D, T}(
            ntuple(j -> T(100Frame + 10i + j), D),
        )
        Particles.Velocity[i] = SVector{D, T}(
            ntuple(j -> T(10Frame + i + j / 10), D),
        )
        Particles.Density[i] = T(1000 + 10Frame + i)
    end
    return nothing
end

function PadTo3D(Data, ::Val{2})
    return reduce(hcat, (SVector(v[1], v[2], zero(eltype(v))) for v in Data))
end

PadTo3D(Data, ::Val{3}) = reduce(hcat, Data)

@testset "ParaView scalar bar formats" begin
    mktempdir() do Directory
        MetaData = SimulationMetaData{2, Float64}(
            SimulationName="scalar_bar_formats",
            SaveLocation=Directory,
            VisualizeInParaview=false,
            OpenLogFile=false,
        )
        AutoOpenParaview(MetaData, ["Density"]; paraview_cmd=nothing)

        StatePath = joinpath(
            Directory,
            "scalar_bar_formats_SingleVTKHDFStateFile.py",
        )
        State = read(StatePath, String)
        @test occursin("colorLegend.LabelFormat = '{:.0f}'", State)
        @test occursin("colorLegend.RangeLabelFormat = '{:.0f}'", State)
        @test occursin("colorLegend.DataRangeLabelFormat = '{:.0f}'", State)
        @test !occursin("%.0f", State)
    end
end

@testset "detailed performance timings" begin
    D = 2
    T = Float64
    MetaData = SimulationMetaData{
        D,
        T,
        NoShifting,
        NoKernelOutput,
        SimpleMDBC,
        NoLog,
    }(
        SimulationName="detailed_timings",
        SaveLocation=".",
        IndexCounter=2,
        VisualizeInParaview=false,
        OpenLogFile=false,
    )
    Constants = SimulationConstants{T}(dx=0.02)
    Kernel = SPHKernelInstance{D, T}(WendlandC2(); dx=Constants.dx)
    Particles = StructArray((
        Position = [SVector{D, T}(0, 0)],
        Density = T[1000],
        GhostPoints = [SVector{D, T}(0.01, 0)],
        GhostNormals = [SVector{D, T}(1, 0)],
        Type = ParticleType[Fixed],
    ))
    ParticleRanges = Int[1, 1, 2]
    UniqueCells = [
        CartesianIndex(typemin(Int), typemin(Int)),
        CartesianIndex(0, 0),
    ]

    @timeit MetaData.HourGlass "MDBC parent" begin
        SPHExample.SPHCellList.ApplyMDBCBeforeHalf!(
            MetaData,
            Kernel,
            Constants,
            Particles,
            ParticleRanges,
            UniqueCells,
        )
    end

    Parent = MetaData.HourGlass["MDBC parent"]
    @test TimerOutputs.ncalls(Parent["01 Acquire MDBC buffers"]) == 1
    @test TimerOutputs.ncalls(Parent["02 NeighborLoopMDBC!"]) == 1
    @test TimerOutputs.ncalls(Parent["03 ApplyMDBCCorrection"]) == 1
    @test all(isfinite, Particles.Density)

    VisibleLabels = ["visible timing $(lpad(Index, 2, '0'))" for Index in 1:30]
    for Label in VisibleLabels
        @timeit MetaData.HourGlass Label nothing
    end

    ReportBuffer = IOBuffer()
    LimitedIO = IOContext(
        ReportBuffer,
        :limit => true,
        :displaysize => (10, 80),
    )
    SPHExample.SPHCellList.ShowPerformanceReport(LimitedIO, MetaData.HourGlass)
    Report = String(take!(ReportBuffer))
    @test occursin("sorted by elapsed time", Report)
    @test occursin("globally sorted by allocations", Report)
    @test occursin("01 Acquire MDBC buffers", Report)
    @test all(Label -> occursin(Label, Report), VisibleLabels)
    @test !occursin("rows omitted", Report)
    @test !occursin("~Flattened~", Report)

    mktempdir() do Directory
        SimLogger = SimulationLogger(Directory; to_console=false)
        LogFinal(SimLogger, MetaData.HourGlass)
        close(SimLogger.LoggerIo)

        LogReport = read(joinpath(Directory, "SimulationOutput.log"), String)
        @test all(Label -> occursin(Label, LogReport), VisibleLabels)
        @test !occursin("rows omitted", LogReport)
    end
end

@testset "VTKHDF Bumper buffers" begin
    mktempdir() do Directory
        D = 2
        T = Float32
        Dimensions = Val(D)
        Particles = MakeVTKTestParticles(Dimensions, T)
        Constants = SimulationConstants{T}()
        Kernel = SPHKernelInstance{D, T}(WendlandC2(); dx=Constants.dx)
        MetaData = SimulationMetaData{D, T}(
            SimulationName="transient_buffers",
            SaveLocation=Directory,
            SimulationTime=zero(T),
            OutputTimes=T[],
            ExportSingleVTKHDF=true,
            VisualizeInParaview=false,
            OpenLogFile=false,
        )

        Output = SetupVTKOutput(MetaData, Particles, Kernel, D)
        ExpectedPositions = Vector{typeof(copy(Particles.Position))}()
        ExpectedVelocities = Vector{typeof(copy(Particles.Velocity))}()
        ExpectedDensities = Vector{typeof(copy(Particles.Density))}()
        ExpectedTimes = T[]

        # Four writes exceed the two-snapshot pool and reuse the same exact-size
        # Bumper arena for several 3D fields on every frame.
        for Frame in 1:4
            SetVTKTestFrame!(Particles, Frame, Dimensions, T)
            MetaData.TotalTime = T(Frame) / T(10)
            push!(ExpectedPositions, copy(Particles.Position))
            push!(ExpectedVelocities, copy(Particles.Velocity))
            push!(ExpectedDensities, copy(Particles.Density))
            push!(ExpectedTimes, MetaData.TotalTime)
            Output.enqueue_particles(Frame)

            # Mutating immediately verifies that queued jobs own a stable snapshot.
            SetVTKTestFrame!(Particles, -Frame, Dimensions, T)
            GC.gc(false)
        end
        Output.close_files()

        h5open(joinpath(Directory, "transient_buffers.vtkhdf"), "r") do File
            Root = File["VTKHDF"]
            Points = read(Root["Points"])
            Velocities = read(Root["PointData"]["Velocity"])
            Densities = read(Root["PointData"]["Density"])
            TypeData = read(Root["PointData"]["Type"])
            BoundaryData = read(Root["PointData"]["BoundaryBool"])
            NumberOfParticles = length(Particles)

            for Frame in 1:4
                Indices = ((Frame - 1) * NumberOfParticles + 1):(Frame * NumberOfParticles)
                @test Points[:, Indices] == PadTo3D(ExpectedPositions[Frame], Dimensions)
                @test Velocities[:, Indices] == PadTo3D(ExpectedVelocities[Frame], Dimensions)
                @test Densities[Indices] == ExpectedDensities[Frame]
            end

            @test eltype(TypeData) == Int8
            @test eltype(BoundaryData) == UInt8
            @test TypeData[1:NumberOfParticles] == Int8[1, 2, 3]
            @test BoundaryData[1:NumberOfParticles] == UInt8[0, 1, 1]
            @test read(Root["Steps"]["Values"]) == ExpectedTimes
            @test read(Root["Steps"]["NumberOfParts"]) == ones(Int64, 4)
            @test HDF5.read_attribute(Root["Steps"], "NSteps") == 4
            @test eltype(Points) == T
            @test eltype(Velocities) == T
            @test eltype(Densities) == T
        end
    end

    mktempdir() do Directory
        D = 2
        T = Float64
        Dimensions = Val(D)
        Particles = MakeVTKTestParticles(Dimensions, T)
        Constants = SimulationConstants{T}()
        Kernel = SPHKernelInstance{D, T}(WendlandC2(); dx=Constants.dx)
        MetaData = SimulationMetaData{D, T}(
            SimulationName="static_buffers",
            SaveLocation=Directory,
            SimulationTime=zero(T),
            OutputTimes=T[0.1, 0.2],
            ExportSingleVTKHDF=false,
            VisualizeInParaview=false,
            OpenLogFile=false,
        )

        SetVTKTestFrame!(Particles, 1, Dimensions, T)
        Output = SetupVTKOutput(MetaData, Particles, Kernel, D)
        ExpectedPositions = Vector{typeof(copy(Particles.Position))}()
        ExpectedVelocities = Vector{typeof(copy(Particles.Velocity))}()
        for Frame in 1:3
            SetVTKTestFrame!(Particles, Frame, Dimensions, T)
            push!(ExpectedPositions, copy(Particles.Position))
            push!(ExpectedVelocities, copy(Particles.Velocity))
            Output.enqueue_particles(Frame)
            SetVTKTestFrame!(Particles, -Frame, Dimensions, T)
        end
        Output.close_files()

        for Frame in 1:3
            FileName = "static_buffers_$(lpad(Frame, 6, '0')).vtkhdf"
            h5open(joinpath(Directory, FileName), "r") do File
                Root = File["VTKHDF"]
                Points = read(Root["Points"])
                Velocities = read(Root["PointData"]["Velocity"])
                @test eltype(Points) == T
                @test eltype(Velocities) == T
                @test Points == PadTo3D(ExpectedPositions[Frame], Dimensions)
                @test Velocities == PadTo3D(ExpectedVelocities[Frame], Dimensions)
            end
        end
    end


    mktempdir() do Directory
        D = 3
        T = Float32
        Dimensions = Val(D)
        Particles = MakeVTKTestParticles(Dimensions, T)
        Constants = SimulationConstants{T}()
        Kernel = SPHKernelInstance{D, T}(WendlandC2(); dx=Constants.dx)
        MetaData = SimulationMetaData{D, T}(
            SimulationName="grid_buffers",
            SaveLocation=Directory,
            SimulationTime=zero(T),
            OutputTimes=T[],
            ExportSingleVTKHDF=true,
            ExportGridCells=true,
            VisualizeInParaview=false,
            OpenLogFile=false,
        )

        Cells = [CartesianIndex(0, 0, 0), CartesianIndex(1, 0, 0)]
        Output = SetupVTKOutput(MetaData, Particles, Kernel, D)
        Output.enqueue_grid(1, Cells)
        Output.close_files()

        GridPath = joinpath(Directory, "grid_buffers_GridCells.vtkhdf")
        h5open(GridPath, "r") do File
            Root = File["VTKHDF"]
            Points = read(Root["Points"])
            @test eltype(Points) == T
            @test size(Points) == (3, 16)
            @test read(Root["NumberOfPoints"]) == Int64[16]
            @test read(Root["NumberOfCells"]) == Int64[2]
            @test read(Root["NumberOfConnectivityIds"]) == Int64[16]
            @test read(Root["Offsets"]) == Int64[0, 8, 16]
            @test read(Root["Types"]) == UInt8[12, 12]
            @test read(Root["CellData"]["CellData"]) == Int64[1, 2]
        end
    end

    mktempdir() do Directory
        D = 2
        T = Float64
        Dimensions = Val(D)
        Particles = MakeVTKTestParticles(Dimensions, T)
        Constants = SimulationConstants{T}()
        Kernel = SPHKernelInstance{D, T}(WendlandC2(); dx=Constants.dx)
        MetaData = SimulationMetaData{D, T}(
            SimulationName="sentinel_free_grid",
            SaveLocation=Directory,
            SimulationTime=zero(T),
            OutputTimes=T[],
            ExportSingleVTKHDF=true,
            ExportGridCells=true,
            ExportGridCellParticleCounts=true,
            VisualizeInParaview=false,
            OpenLogFile=false,
        )

        Sentinel = CartesianIndex(typemin(Int), typemin(Int))
        UniqueCells = [Sentinel, CartesianIndex(0, 0), CartesianIndex(1, 0)]
        IndexCounter = length(UniqueCells)
        ParticleRanges = Int[1, 1, 3, 4]
        NeighborCellLists = [Int[], Int[3], Int[2]]
        PhysicalCells = SPHExample.SPHCellList.PhysicalCellView(
            UniqueCells,
            IndexCounter,
        )
        ParticleCounts, NeighborCounts =
            SPHExample.SPHCellList.PrepareGridExportData(
                Val(true),
                ParticleRanges,
                IndexCounter,
                NeighborCellLists,
            )

        @test collect(PhysicalCells) == UniqueCells[2:IndexCounter]
        @test collect(ParticleCounts) == [2, 1]
        @test collect(NeighborCounts) == [2, 2]

        Output = SetupVTKOutput(MetaData, Particles, Kernel, D)
        Output.enqueue_grid(
            1,
            PhysicalCells,
            cell_particle_counts=ParticleCounts,
            cell_neighbor_counts=NeighborCounts,
        )
        Output.close_files()

        GridPath = joinpath(Directory, "sentinel_free_grid_GridCells.vtkhdf")
        h5open(GridPath, "r") do File
            Root = File["VTKHDF"]
            Points = read(Root["Points"])
            @test all(isfinite, Points)
            @test maximum(abs, Points) < one(T)
            @test size(Points) == (3, 8)
            @test read(Root["NumberOfCells"]) == Int64[2]
            @test read(Root["CellData"]["CellData"]) == Int64[1, 2]
            @test read(Root["CellData"]["ParticleCount"]) == Int64[2, 1]
            @test read(Root["CellData"]["ParticleNeighborsPerCell"]) == Int64[2, 2]
        end
    end

    mktempdir() do Directory
        D = 2
        T = Float32
        Constants = SimulationConstants{T}()
        Kernel = SPHKernelInstance{D, T}(WendlandC2(); dx=Constants.dx)
        Cells = [CartesianIndex(0, 0), CartesianIndex(1, 0)]
        GridPath = joinpath(Directory, "direct_grid.vtkhdf")
        SaveCellGridVTKHDF(GridPath, Kernel, Cells)

        h5open(GridPath, "r") do File
            Root = File["VTKHDF"]
            @test eltype(read(Root["Points"])) == T
            @test size(Root["Points"]) == (3, 8)
            @test read(Root["Connectivity"]) == Int64.(0:7)
            @test read(Root["Offsets"]) == Int64[0, 4, 8]
            @test read(Root["Types"]) == UInt8[9, 9]
        end
    end
end

@testset "isolated particle" begin
    D = 2
    T = Float64
    sc = SimulationConstants{T}()
    ker = SPHKernelInstance{D,T}(WendlandC2(); dx=sc.dx)
    meta = SimulationMetaData{D,T}(SimulationName="iso", SaveLocation=".")

    pos = [SVector{D,T}(0, 0)]
    vel = [SVector{D,T}(0, 0)]
    acc = [SVector{D,T}(0, 0)]
    dens = [sc.ρ₀]
    press = [0.0]
    gf = [-1.0]
    limiter = [1.0]
    bound = UInt8[0]
    id = [1]
    typ = [Fluid]
    group = UInt[1]
    kernel_val = [0.0]
    kernel_grad = [SVector{D,T}(0, 0)]
    cell = [CartesianIndex(0, 0)]
    gpoint = [SVector{D,T}(0, 0)]
    gnorm = [SVector{D,T}(0, 0)]

    particles = StructArray((
        Cells=cell, Kernel=kernel_val, KernelGradient=kernel_grad,
        Position=pos, Acceleration=acc, Velocity=vel, Density=dens, Pressure=press,
        GravityFactor=gf, MotionLimiter=limiter, BoundaryBool=bound, ID=id,
        Type=typ, GroupMarker=group, GhostPoints=gpoint, GhostNormals=gnorm,
    ))

    dρdtI, vel_n, pos_n, ρ_n, ∇C, ∇r =
        AllocateSupportDataStructures(meta, particles.Position)

    for _ in 1:1000
        ResetArrays!(dρdtI, particles.Acceleration)
        dt = Δt(zero(T), sc, ker)
        dt2 = dt / 2

        SPHExample.SPHCellList.HalfTimeStep(meta, sc, particles, pos_n, vel_n,
                                           ρ_n, dρdtI, dt2)
        LimitDensityAtBoundary!(ρ_n, sc.ρ₀, particles.MotionLimiter)
        Pressure!(press, ρ_n, sc)
        SPHExample.SPHCellList.FullTimeStep(meta, ker, sc, particles, vel_n, ∇C, ∇r, dt)
        DensityEpsi!(dens, dρdtI, ρ_n, dt)
        LimitDensityAtBoundary!(dens, sc.ρ₀, particles.MotionLimiter)
        SPHExample.SPHCellList.UpdateMetaData!(meta, dt)

        @test isapprox(dens[1], sc.ρ₀; atol=1e-10)
        @test isapprox(press[1], 0; atol=1e-10)
    end

    @test particles.Position[1][1] == 0
    @test particles.Velocity[1][1] == 0
    @test particles.Velocity[1][2] < 0
end
