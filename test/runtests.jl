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
    dt  = Δt(pos, vel, acc, sc, ker)
    @test dt > 0
    alloc = @allocated Δt(pos, vel, acc, sc, ker)
    @test alloc == 0
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

    Report = sprint(SPHExample.SPHCellList.ShowPerformanceReport, MetaData.HourGlass)
    @test occursin("sorted by elapsed time", Report)
    @test occursin("globally sorted by allocations", Report)
    @test occursin("01 Acquire MDBC buffers", Report)
    @test !occursin("rows omitted", Report)
    @test !occursin("~Flattened~", Report)
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
        dt = Δt(particles.Position, particles.Velocity, particles.Acceleration,
                 sc, ker)
        dt2 = dt / 2

        SPHExample.SPHCellList.HalfTimeStep(meta, sc, particles, pos_n, vel_n,
                                           ρ_n, dρdtI, dt2)
        LimitDensityAtBoundary!(ρ_n, sc.ρ₀, particles.MotionLimiter)
        Pressure!(press, ρ_n, sc)
        SPHExample.SPHCellList.FullTimeStep(meta, ker, sc, particles, ∇C, ∇r, dt)
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

@testset "simulation failure finalizes output" begin
    mktempdir() do dir
        meta = SimulationMetaData{2,Float64}(
            SimulationName="failfinalizer",
            SaveLocation=dir,
            VisualizeInParaview=false,
            OpenLogFile=false,
        )
        logger = SimulationLogger(dir; to_console=false)
        closed = Ref(false)
        output = (
            close_files = () -> (closed[] = true; nothing),
            variable_names = String[],
        )

        @test_throws ErrorException SPHExample.SPHCellList.RunWithSimulationFinalizer!(meta, logger, output) do _
            error("boom")
        end
        @test closed[]
        close(logger.LoggerIo)
    end
end
