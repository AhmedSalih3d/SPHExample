module TimestepDiagnosticTests

using Test, SPHExampleGPU, CUDA, StaticArrays
using SPHExampleGPU.GPUReductions: ReductionWorkspace
using SPHExampleGPU.GPUStepState

const K = SPHExampleGPU.GPUKernels

@testset "timestep diagnostic configuration" begin
    metadata = SimulationMetaData{2, Float32}(SimulationName = "dt", SaveLocation = tempdir())
    @test metadata.GPUTimestepDiagnosticsEvery == 0
    @test_throws ArgumentError SimulationMetaData{2, Float32}(
        SimulationName = "dt", SaveLocation = tempdir(), GPUTimestepDiagnosticsEvery = -1)
end

if CUDA.functional()
    @testset "timestep criterion and stable integer IDs" begin
        for T in (Float32, Float64), I in (Int32, Int64), force in (false, true), graph in (false, true)
            n = 513
            V = SVector{2,T}
            constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(20), g = zero(T))
            kernel = SPHKernelInstance{2,T}(WendlandC2(); dx = constants.dx)
            position = CuArray(fill(V(1, 0), n))
            velocities = fill(zero(V), n)
            velocities[3] = velocities[259] = V(20, 0)
            accelerations = fill(zero(V), n)
            if force
                accelerations[7] = accelerations[300] = V(1e8, 0)
            end
            offset = I === Int64 ? Int64(2)^54 : Int32(2)^25
            ids = I[offset + n - i for i in 1:n]
            red = ReductionWorkspace{SVector{3,T}}(n)
            diagnostics = K.TimestepDiagnostics{T,I}(n)
            time = 0.250000000123
            state = StepState{T}(; time, iteration = 17)
            set_output_time!(state, one(T))
            K.launch_step_reduction!(red, position, CuArray(velocities), CuArray(accelerations), position, kernel;
                diagnostics, particle_ids = CuArray(ids))
            finish!() = K.launch_finish!(state, red, kernel, constants; diagnostics)
            if graph
                executable = CUDA.instantiate(CUDA.capture(finish!))
                CUDA.launch(executable)
            else
                finish!()
            end
            readback!(state)
            record = only(Array(diagnostics.result))
            visc = abs(kernel.h * T(20) / (one(T) + kernel.η²))
            expected_force = force ? sqrt(kernel.h / T(1e8)) : T(Inf)
            expected_acoustic = kernel.h / (constants.c₀ + visc)
            @test record.iteration == 18
            @test record.time === time
            @test record.dt isa T
            @test record.dt == state.fh[F_DT] == constants.CFL * min(expected_force, expected_acoustic)
            @test record.force_dt == constants.CFL * expected_force
            @test record.acoustic_viscous_dt == constants.CFL * expected_acoustic
            @test record.criterion == (force ? 1 : 2)
            @test record.particle_id == ids[force ? 300 : 259]
            @test record.has_particle
            # A rebuild keeps the already selected timestep and its provenance.
            before = record
            resume_after_rebuild!(state)
            finish!()
            @test only(Array(diagnostics.result)) === before
            io = IOBuffer()
            K.write_timestep_diagnostic!(io, diagnostics, T(0.5))
            row = split(chomp(String(take!(io))), ',')
            @test row[2] == "18"
            @test parse(Float64, row[3]) === time
            @test row[5] == (force ? "force" : "acoustic_viscous")
            @test parse(I, row[6]) == record.particle_id
            K.launch_commit!(state)
            set_output_time!(state, zero(T))
            finish!()
            readback!(state)
            @test state.ih[I_STOP] == STOP_OUTPUT
            @test only(Array(diagnostics.result)) === before
        end
    end

    @testset "final integration provenance excludes frozen particles" begin
        for T in (Float32, Float64), n in (7, 513), frozen in (false, true)
            V = SVector{2,T}
            constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(20), g = zero(T))
            kernel = SPHKernelInstance{2,T}(WendlandC2(); dx = constants.dx)
            position = CuArray(fill(V(1, 0), n))
            velocity = CUDA.zeros(V, n)
            acceleration = CuArray([i == 1 ? V(1e20, 0) : V(i, 0) for i in 1:n])
            types = CuArray([frozen || i == 1 ? Frozen : Fixed for i in 1:n])
            density = CUDA.fill(T(1000), n)
            rate = CUDA.zeros(T, n)
            ids = CuArray(Int64.(n:-1:1) .+ Int64(2)^54)
            red = ReductionWorkspace{SVector{3,T}}(n)
            diagnostics = K.TimestepDiagnostics{T,Int64}(n)
            K.launch_final_step!(position, velocity, acceleration, density, rate, density, position, velocity,
                types, nothing, nothing, HostStep(T(1e-5), zero(T)), kernel, constants, red, Val(false);
                diagnostics, particle_ids = ids)
            state = StepState{T}()
            set_output_time!(state, one(T))
            K.launch_finish!(state, red, kernel, constants; diagnostics)
            record = only(Array(diagnostics.result))
            @test record.has_particle == !frozen
            @test record.force_dt == (frozen ? T(Inf) : constants.CFL * sqrt(kernel.h / T(n)))
            if !frozen
                @test record.criterion == 2
                # Every eligible particle has zero velocity: lowest stable ID wins.
                @test record.particle_id == Int64(2)^54 + 1
            else
                io = IOBuffer()
                K.write_timestep_diagnostic!(io, diagnostics, record.dt)
                @test split(chomp(String(take!(io))), ',')[6] == ""
            end
        end
    end

    function simulate(T; every = 0, graph = false, async = false, single = false, compact = true)
        mktempdir() do dir
            points = [SVector{2,T}(x, 0) for x in (4.05, 0, 4)]
            geometry = [SPHGeometry{2,T}(points; Density = T(1000), Type = Fluid, GroupMarker = 1)]
            metadata = SimulationMetaData{2,T}(
                SimulationName = "Diagnostics", SaveLocation = dir,
                SimulationTime = T(0.004), OutputTimes = T[0.0008, 0.0015, 0.003, 0.004],
                GPUTimestepDiagnosticsEvery = every, GPUUseGraph = graph, GPUAsyncOutput = async,
                GPUCompactBuffers = compact, GPUDoublePosition = T === Float32,
                GPUOutputBufferBytes = 0, VisualizeInParaview = false, OpenLogFile = false)
            particles = AllocateDataStructures(geometry)
            particles.ID .= Int64(2)^54 .+ [3, 1, 2]
            particles.Velocity .= Ref(SVector{2,T}(1, 0))
            constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(20), g = zero(T))
            kernel = SPHKernelInstance{2,T}(WendlandC2(); dx = constants.dx)
            logger = SimulationLogger(dir)
            try
                redirect_stdout(devnull) do
                    RunSimulation(; SimGeometry = geometry, SimMetaData = metadata,
                        SimConstants = constants, SimKernel = kernel, SimLogger = logger,
                        SimParticles = particles, SimViscosity = ZeroViscosity(),
                        SimDensityDiffusion = ZeroDensityDiffusion(),
                        SimTimeStepping = single ? SingleNeighborTimeStepping() : SymplecticTimeStepping())
                end
            finally
                close(logger.LoggerIo)
            end
            order = sortperm(particles.ID)
            path = joinpath(dir, "Diagnostics_Timestep.csv")
            rows = isfile(path) ? readlines(path) : String[]
            state = (; iteration = metadata.Iteration, time = metadata.TotalTime, dt = metadata.CurrentTimeStep,
                positions = particles.Position[order], velocities = particles.Velocity[order],
                density = particles.Density[order], acceleration = particles.Acceleration[order])
            return (; rows, state)
        end
    end

    @testset "selected output intervals preserve trajectories" begin
        for T in (Float32, Float64), graph in (false, true), single in (false, true)
            baseline = simulate(T; graph, single)
            all_intervals = simulate(T; every = 1, graph, single)
            selected = simulate(T; every = 2, graph, single, async = true, compact = false)
            @test isempty(baseline.rows)
            @test all_intervals.state == baseline.state == selected.state
            @test length(all_intervals.rows) == 5
            @test selected.rows == all_intervals.rows[[1, 3, 5]]
            @test first(all_intervals.rows) == "OutputTime,Iteration,Time,TimeStep,Criterion,ParticleID,ForceTimeStep,AcousticViscousTimeStep"
            for line in all_intervals.rows[2:end]
                row = split(line, ',')
                @test row[5] == "acoustic_viscous"
                @test parse(Int64, row[6]) == Int64(2)^54 + 1
                @test parse(T, row[4]) == parse(T, row[8])
                @test parse(T, row[1]) ≈ parse(T, row[3]) + parse(T, row[4])
            end
            last_row = split(last(all_intervals.rows), ',')
            @test parse(Int, last_row[2]) == baseline.state.iteration
            @test parse(T, last_row[4]) == baseline.state.dt
        end
    end
else
    @testset "timestep diagnostics require CUDA" begin
        @test_skip CUDA.functional()
    end
end

end
