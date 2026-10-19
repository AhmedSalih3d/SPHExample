module GPULoggingTests

using Test, SPHExampleGPU, CUDA, StaticArrays, StructArrays

const LOGGER = SPHExampleGPU.SimulationLoggerConfiguration

function fixture(D, T, save; double_position = false, compact = true,
                 boundary = NoMDBC, log_mode = StoreLog, async = false)
    normal = SVector{D, T}(ntuple(d -> d == D ? T(0.02) : zero(T), D))
    fluid = [SVector{D, T}(ntuple(d -> T(4) + (d == 1 ? T(i) * T(0.02) : zero(T)), D))
             for i in 0:1]
    geometry = [
        SPHGeometry{D, T}(Particles = particle_struct_array([zero(normal)], T(1000);
            GhostPoints = [normal], GhostNormals = [normal]), GroupMarker = 1, Type = Fixed),
        SPHGeometry{D, T}(fluid; Density = 1000, GroupMarker = 2, Type = Fluid),
    ]
    metadata = SimulationMetaData{D, T, NoShifting, NoKernelOutput, boundary, log_mode}(
        SimulationName = "Logging", SaveLocation = save,
        SimulationTime = T(0.001), OutputTimes = T(0.001),
        GPUDoublePosition = double_position, GPUCompactBuffers = compact,
        GPUAsyncOutput = async, GPUOutputQueueBytes = 1024,
        VisualizeInParaview = false, OpenLogFile = false)
    constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(20), g = zero(T))
    kernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx)
    host = AllocateDataStructures(geometry, metadata)
    host.Velocity[end] = SVector{D, T}(ntuple(d -> d == 1 ? T(0.1) : zero(T), D))
    return (; geometry, metadata, constants, kernel, host)
end

@testset "host particle memory in GPU logs" begin
    for D in (2, 3), T in (Float32, Float64), double_position in (false, true)
        mktempdir() do dir
            f = fixture(D, T, dir; double_position)
            expected = sum(sizeof, components(f.host))
            aliased = StructArray(merge(components(f.host), (; DensityAlias = f.host.Density)))
            @test LOGGER.host_particle_bytes(aliased) == expected
            logger = SimulationLogger(dir)
            try
                redirect_stdout(devnull) do
                    InitializeLogger(logger, f.constants, f.metadata, f.kernel, ZeroViscosity(),
                        ZeroDensityDiffusion(), f.geometry, f.host)
                end
            finally
                close(logger.LoggerIo)
            end
            output = read(joinpath(dir, "SimulationOutput.log"), String)
            @test occursin("CPU particle arrays: $expected bytes", output)
            @test occursin("Simulation dimensions: $D", output)
            @test occursin("CPU positions $(double_position ? Float64 : T)", output)
            @test occursin("GPU buffer memory: not reported (device buffers not supplied)", output)
            @test occursin("CUDA information:", output)
            if CUDA.functional()
                @test occursin("runtime $(CUDA.runtime_version())", output)
                @test occursin("driver ", output)
                @test occursin("GPU: $(CUDA.name(CUDA.device())), compute capability", output)
                @test occursin("CUDA.jl $(pkgversion(CUDA))", output)
            else
                @test occursin("CUDA information: unavailable (CUDA is not functional)", output)
            end
        end
    end
    @test LOGGER.host_particle_bytes(StructArray((Density = Float32[],))) == 0
    @test LOGGER.memory_usage_line("GPU accounted buffers total", 16, 0) ==
        "GPU accounted buffers total: 16 bytes (0.0000 MiB; n/a bytes/particle)"
    @test LOGGER.output_pipeline_memory_line((; pinned_bytes = 2^20, queue_bytes = 2 * 2^20,
        queue_frames = 4, writer_bytes = 8 * 2^20)) ==
        "Output staging & buffer capacity: 1.0000 MiB pinned + 2.0000 MiB queue (up to 4 frames) + 8.0000 MiB HDF5 writer buffer"
end

if CUDA.functional()
    @testset "alias-aware live GPU buffer accounting" begin
        for D in (2, 3), T in (Float32, Float64)
            vector = CUDA.zeros(SVector{D, T}, 5)
            ids = CuArray(Int32[1, 2, 3])
            nested = (; fields = (vector, ids), repeated = vector,
                host = zeros(Float64, 100), absent = nothing, flags = (true, 1))
            expected = 5 * D * sizeof(T) + 3 * sizeof(Int32)
            @test LOGGER.device_buffer_bytes(nested) == expected
            seen = Base.IdSet()
            @test LOGGER.device_buffer_bytes(vector, seen) == 5 * D * sizeof(T)
            @test LOGGER.device_buffer_bytes(nested, seen) == 3 * sizeof(Int32)
            @test LOGGER.device_buffer_bytes(CuArray(T[])) == 0
        end
        for compact in (false, true), double_position in (false, true)
            f = fixture(2, Float32, tempdir(); compact, double_position, boundary = UpdatedMDBC)
            gpu = upload_particles(f.host, f.metadata)
            support = GPUSupportArrays{2, Float32}(gpu, f.metadata)
            cells = CellListWorkspace{2, position_float_type(f.metadata)}(length(f.host))
            auxiliary = CUDA.zeros(Float64, 2)
            buffers = (; particles = gpu, support, cells,
                auxiliary = (gpu.BoundarySlot, auxiliary))
            bytes = LOGGER.gpu_memory_bytes(buffers)
            @test bytes.particles == LOGGER.device_buffer_bytes(gpu)
            @test sum(bytes) == LOGGER.device_buffer_bytes(buffers)
            @test bytes.auxiliary == 2 * sizeof(Float64)
            if compact
                @test gpu.BoundarySlot === support.boundary_data.slots
                @test bytes.support == LOGGER.device_buffer_bytes(support) - sizeof(gpu.BoundarySlot)
            else
                @test bytes.support == LOGGER.device_buffer_bytes(support)
            end
        end
    end

    function simulate(log_mode, async, scheme)
        return mktempdir() do dir
            f = fixture(2, Float32, dir; double_position = true, log_mode, async)
            logger = SimulationLogger(dir)
            logger.clock.start_ns -= UInt64(60_000_000_000)
            initial_time = f.metadata.TotalTime
            before_run = time_ns()
            console_path = joinpath(dir, "console.log")
            try
                open(console_path, "w") do console
                    redirect_stdout(console) do
                        RunSimulation(; SimGeometry = f.geometry, SimMetaData = f.metadata,
                            SimConstants = f.constants, SimKernel = f.kernel, SimLogger = logger,
                            SimParticles = f.host, SimViscosity = ZeroViscosity(),
                            SimDensityDiffusion = ZeroDensityDiffusion(), SimTimeStepping = scheme)
                    end
                end
            finally
                close(logger.LoggerIo)
            end
            order = sortperm(f.host.ID)
            state = (; iteration = f.metadata.Iteration, time = f.metadata.TotalTime,
                position = f.host.Position[order], density = f.host.Density[order],
                velocity = f.host.Velocity[order], pressure = f.host.Pressure[order],
                acceleration = f.host.Acceleration[order])
            return (; state, log = read(joinpath(dir, "SimulationOutput.log"), String),
                console = read(console_path, String),
                clock_restarted = logger.clock.start_ns >= before_run && logger.clock.initial_time == initial_time)
        end
    end

    @testset "CUDA and memory logging preserves simulation results" begin
        for async in (false, true), scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            reference = simulate(NoLog, async, scheme)
            logged = simulate(StoreLog, async, scheme)
            @test reference.clock_restarted
            @test logged.clock_restarted
            @test logged.state == reference.state
            @test logged.state.iteration > 0
            @test logged.state.position[3][1] > 4.02
            @test isempty(reference.log)
            @test occursin("CUDA information:", logged.log)
            @test occursin("GPU float type: Float32", logged.log)
            @test occursin("GPU position type: Float64", logged.log)
            @test occursin("Total number of particles: 3", logged.log)
            @test occursin("CPU particle arrays:", logged.log)
            @test occursin("GPU live-buffer memory (startup, before first cell-list rebuild):", logged.log)
            @test occursin("GPU live-buffer memory (final, current cell-list capacity):", logged.log)
            @test occursin("GPU particle arrays (state + reorder):", logged.log)
            @test occursin("GPU auxiliary arrays (motion, reductions, step state, floating):", logged.log)
            @test occursin("allocator reservations", logged.log)
            @test occursin("not process or peak VRAM", logged.log)
            cells = [parse(Int, m.captures[1]) for m in
                eachmatch(r"^GPU cell-list workspace: (\d+) bytes"m, logged.log)]
            @test length(cells) == 2
            @test last(cells) > first(cells)
            totals = collect(eachmatch(r"^GPU accounted buffers total: \d+ bytes \(\d+\.\d{4} MiB; \d+\.\d{2} bytes/particle\)"m, logged.log))
            @test length(totals) == 2
            pools = collect(eachmatch(r"^CUDA pool: \d+\.\d{4} MiB used / \d+\.\d{4} MiB reserved \(\d+\.\d{4} MiB free device VRAM\)"m, logged.log))
            @test length(pools) == 2
            pipeline_re = async ?
                r"^Output staging & buffer capacity: \d+\.\d{4} MiB pinned \+ \d+\.\d{4} MiB queue \(up to [1-9]\d* frames\) \+ \d+\.\d{4} MiB HDF5 writer buffer"m :
                r"^Output staging & buffer capacity: \d+\.\d{4} MiB pinned \+ 0\.0000 MiB queue \(up to 0 frames\) \+ \d+\.\d{4} MiB HDF5 writer buffer"m
            pipelines = collect(eachmatch(pipeline_re, logged.log))
            @test length(pipelines) == 2
            @test occursin("Simulation finished at:", logged.log)
            for output in (reference.console, logged.console, logged.log)
                headers = collect(eachmatch(r"(?m)^\s*Section\s+ncalls\b", output))
                @test length(headers) == 1
                @test occursin(r"─{10,}\s*\z", output)
                if occursin("Simulation finished at:", output)
                    @test first(findfirst("Simulation took", output)) < only(headers).offset
                    @test first(findfirst("GPU live-buffer memory (final, current cell-list capacity):", output)) < only(headers).offset
                end
            end
        end
    end
end

end
