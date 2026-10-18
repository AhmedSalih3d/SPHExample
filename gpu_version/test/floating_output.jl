module FloatingOutputTests

using Test, SPHExampleGPU, CUDA, StaticArrays, HDF5, CSV

const DRIVER = SPHExampleGPU.SPHCellList

function floating_fixture(D, T, save; double_position = false)
    corners = [SVector{D, T}(ntuple(d -> T(0.04) * T((i >> (d - 1)) & 1), D))
               for i in 0:(2^D - 1)]
    geometry = [SPHGeometry{D, T}(
        corners .+ Ref(SVector{D, T}(ntuple(d -> d == D ? one(T) : T(2b), D)));
        Density = 1000, GroupMarker = marker, Type = Floating,
        Floating = FloatingDetails{T}(RelativeWeight = 1.2, PauseTime = b == 2 ? 0.003 : 0))
        for (b, marker) in enumerate((7, 3))]
    metadata = SimulationMetaData{D, T}(SimulationName = "FloatingOutput", SaveLocation = save,
        SimulationTime = T(0.012), OutputTimes = T(0.002), GPUDoublePosition = double_position,
        VisualizeInParaview = false, OpenLogFile = false,
        OutputVariables = ["ID", "GroupMarker", "Velocity", "Density"])
    constants = SimulationConstants{T}(dx = 0.04, c₀ = 20)
    return (; geometry, metadata, constants)
end

function seed_body_state!(floating, D, phase)
    fields = D == 2 ? (:center, :velocity, :angle, :omega) : (:center, :velocity, :orientation, :omega)
    for (k, name) in enumerate(fields)
        array = getproperty(floating, name)
        E = eltype(array)
        values = if E <: StaticVector
            [E(ntuple(d -> phase + 10k + b + d / 100, length(E))) for b in eachindex(array)]
        else
            [E(phase + 10k + b) for b in eachindex(array)]
        end
        copyto!(array, values)
    end
    return floating_state(floating)
end

@testset "floating snapshots survive staging reuse" begin
    for D in (2, 3), T in (Float32, Float64), double_position in (false, true)
        fixture = floating_fixture(D, T, tempdir(); double_position)
        particles = AllocateDataStructures(fixture.geometry, fixture.metadata)
        gpu = upload_particles(particles, fixture.metadata)
        floating = FloatingArrays(fixture.geometry, particles, fixture.constants;
            position_type = position_float_type(fixture.metadata))
        download = DRIVER.OutputDownload(gpu, (:ID,); cells = false, floating)
        try
            first_frame = DRIVER.host_output_frame(download)
            second_frame = DRIVER.host_output_frame(download)
            @test_throws ArgumentError DRIVER.enqueue_download!(download, gpu)
            first_state = seed_body_state!(floating, D, 1)
            DRIVER.enqueue_download!(download, gpu; floating)
            DRIVER.collect_download!(first_frame, download)
            second_state = seed_body_state!(floating, D, 2)
            DRIVER.enqueue_download!(download, gpu; floating)
            DRIVER.collect_download!(second_frame, download)
            @test first_frame.floating == first_state
            @test second_frame.floating == second_state
            @test typeof(first_frame.floating) === typeof(first_state)
            @test isempty(first_frame.memory)
            for name in keys(first_state)
                @test getproperty(first_frame.floating, name) !== getproperty(download.floating, name)
                @test getproperty(first_frame.floating, name) !== getproperty(second_frame.floating, name)
            end
        finally
            CUDA.synchronize()
            DRIVER.free!(download)
        end
        inactive = DRIVER.OutputDownload(gpu, (:ID,); cells = false,
            floating = merge(floating, (; active = false)))
        try
            @test inactive.floating === nothing
            DRIVER.enqueue_download!(inactive, gpu)
        finally
            CUDA.synchronize()
            DRIVER.free!(inactive)
        end
    end
end

function simulate_output(D, T; double_position, scheme, async, queue_bytes)
    return mktempdir() do save
        (; geometry, metadata, constants) = floating_fixture(D, T, save; double_position)
        metadata.GPUAsyncOutput = async
        metadata.GPUOutputQueueBytes = queue_bytes
        metadata.GPUUseGraph = async
        metadata.GPUMaxStepsPerSync = async ? 32 : 1
        particles = AllocateDataStructures(geometry, metadata)
        logger = SimulationLogger(save)
        try
            redirect_stdout(devnull) do
                RunSimulation(SimGeometry = geometry, SimMetaData = metadata, SimConstants = constants,
                    SimKernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx),
                    SimLogger = logger, SimParticles = particles, SimViscosity = ZeroViscosity(),
                    SimDensityDiffusion = ZeroDensityDiffusion(), SimTimeStepping = scheme)
            end
        finally
            isopen(logger.LoggerIo) && close(logger.LoggerIo)
        end
        csv = read(joinpath(save, "FloatingOutput_Floating.csv"), String)
        output = h5open(joinpath(save, "FloatingOutput.vtkhdf"), "r") do file
            root = file["VTKHDF"]
            (; times = read(root["Steps/Values"]), positions = read(root["Points"]),
               groups = read(root["PointData/GroupMarker"]), velocity = read(root["PointData/Velocity"]))
        end
        return (; csv, output, iteration = metadata.Iteration, position = particles.Position,
            velocity = particles.Velocity, density = particles.Density)
    end
end

@testset "queued floating CSV matches synchronous state and particle frames" begin
    for D in (2, 3), T in (Float32, Float64), double_position in (false, true),
        scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
        reference = simulate_output(D, T; double_position, scheme, async = false, queue_bytes = 0)
        for queue_bytes in (0, 1024^2)
            result = simulate_output(D, T; double_position, scheme, async = true, queue_bytes)
            @test result == reference
        end
        rows = collect(CSV.File(IOBuffer(reference.csv)))
        frames = length(reference.output.times)
        @test frames > 2
        @test length(rows) == 2frames
        @test Tuple(propertynames(first(rows))) == (D == 2 ?
            (:Time, :Body, :GroupMarker, Symbol("Center:0"), Symbol("Center:1"),
             Symbol("Velocity:0"), Symbol("Velocity:1"), :Angle, :Omega) :
            (:Time, :Body, :GroupMarker, Symbol("Center:0"), Symbol("Center:1"), Symbol("Center:2"),
             Symbol("Velocity:0"), Symbol("Velocity:1"), Symbol("Velocity:2"),
             Symbol("Orientation:0"), Symbol("Orientation:1"), Symbol("Orientation:2"), Symbol("Orientation:3"),
             Symbol("Omega:0"), Symbol("Omega:1"), Symbol("Omega:2")))
        n = length(reference.position)
        for frame in 1:frames, (body, marker) in enumerate((7, 3))
            row = rows[2(frame - 1) + body]
            @test row.Body == body
            @test row.GroupMarker == marker
            @test row.Time ≈ reference.output.times[frame] rtol = 2eps(T)
            columns = ((frame - 1) * n + 1):(frame * n)
            members = columns[reference.output.groups[columns] .== marker]
            for axis in 1:D
                center = sum(reference.output.positions[axis, members]) / length(members)
                velocity = sum(reference.output.velocity[axis, members]) / length(members)
                @test getproperty(row, Symbol("Center:$(axis - 1)")) ≈ center atol = 8eps(T) rtol = 8eps(T)
                @test getproperty(row, Symbol("Velocity:$(axis - 1)")) ≈ velocity atol = 8eps(T) rtol = 8eps(T)
            end
        end
    end
end

end
