module SimulationResultTests

using Test, SPHExampleGPU, CUDA, StaticArrays, HDF5, TimerOutputs

const DRIVER = SPHExampleGPU.SPHCellList

function fixture(save, D, T; async = false, scheme = SymplecticTimeStepping(),
                 body = false, resumed = false)
    points = [SVector{D,T}(ntuple(d -> T(d == 1 ? 0 : 1) + T((i >> (d - 1)) & 1) * T(0.02), D))
        for i in reverse(0:(2^D - 1))]
    geometry = [SPHGeometry{D,T}(points; Density = T[1001 + i for i in eachindex(points)],
        Type = Fluid, GroupMarker = 1)]
    if body
        body_points = [p + SVector{D,T}(ntuple(d -> d == 1 ? T(2) : zero(T), D)) for p in points]
        push!(geometry, SPHGeometry{D,T}(body_points; Density = 1000, Type = Floating,
            GroupMarker = 2, Floating = FloatingDetails{T}(RelativeWeight = 1)))
    end
    start_time = resumed ? T(0.01) : zero(T)
    start_iteration = resumed ? 7 : 0
    meta = SimulationMetaData{D,T}(
        SimulationName = "SimulationResult", SaveLocation = save,
        SimulationTime = start_time + T(0.001),
        OutputTimes = [start_time + T(0.0005), start_time + T(0.001)],
        TotalTime = start_time, Iteration = start_iteration,
        GPUAsyncOutput = async, GPUUseGraph = async, GPUSyncTimers = !async,
        GPUCompactBuffers = async, GPUDoublePosition = T === Float32,
        GPUOutputQueueBytes = 1024, OutputVariables = ["Density"],
        VisualizeInParaview = false, OpenLogFile = false)
    constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(20), g = zero(T), m₀ = T(1000) * T(0.02)^D)
    kernel = SPHKernelInstance{D,T}(WendlandC2(); dx = constants.dx)
    particles = AllocateDataStructures(geometry, meta)
    return (; SimGeometry = geometry, SimMetaData = meta, SimConstants = constants,
        SimKernel = kernel, SimParticles = particles, SimViscosity = ZeroViscosity(),
        SimDensityDiffusion = ZeroDensityDiffusion(), SimTimeStepping = scheme)
end

function run_fixture(f; viscosity = f.SimViscosity)
    logger = SimulationLogger(f.SimMetaData.SaveLocation)
    try
        return redirect_stdout(devnull) do
            RunSimulation(; merge(f, (; SimViscosity = viscosity))..., SimLogger = logger)
        end
    finally
        close(logger.LoggerIo)
    end
end

# Inject exits at deterministic driver boundaries without depending on OS signals.
struct StopRun{E} <: SPHViscosity
    exception::E
    advance::Bool
end

function DRIVER.SimulationLoop(diffusion::SPHDensityDiffusion, stop::StopRun, kernel,
        meta::SimulationMetaData{D,T}, constants, gpu::GPUParticles{D,T},
        cells::CellListWorkspace, support::GPUSupportArrays, reduction::ReductionWorkspace,
        motion, state::StepState{T}; kwargs...) where {D,T}
    if stop.advance
        DRIVER.SimulationLoop(diffusion, ZeroViscosity(), kernel, meta, constants,
            gpu, cells, support, reduction, motion, state; kwargs...)
    end
    throw(stop.exception)
end

function output_times(f)
    # Opening in r+ also checks that RunSimulation closed its output handles.
    return h5open(joinpath(f.SimMetaData.SaveLocation, "SimulationResult.vtkhdf"), "r+") do file
        read(file["VTKHDF/Steps/Values"])
    end
end

if CUDA.functional()
    @testset "structured simulation results" begin
        for (D,T) in ((2, Float32), (3, Float64)), async in (false, true),
            scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            mktempdir() do save
                f = fixture(save, D, T; async, scheme, body = true, resumed = true)
                meta = f.SimMetaData
                initial_time, initial_iteration = meta.TotalTime, meta.Iteration
                initial_positions = copy(f.SimParticles.Position)
                before = time_ns()
                result = run_fixture(f)
                elapsed = (time_ns() - before) / 1e9
                @test result isa SimulationResult
                @test result.termination_reason === :completed
                state, perf, diag = result.final_state, result.performance, result.diagnostics
                @test state.complete
                @test state.particles === f.SimParticles
                @test state.time == meta.TotalTime > meta.SimulationTime
                @test state.iteration == meta.Iteration > initial_iteration
                @test state.timestep == meta.CurrentTimeStep > 0
                @test perf.steps == meta.Iteration - initial_iteration
                @test perf.simulated_time == Float64(state.time) - Float64(initial_time)
                @test 0 < perf.wall_time_seconds <= elapsed
                @test perf.mean_timestep == perf.simulated_time / perf.steps
                @test perf.milliseconds_per_step == 1e3 * perf.wall_time_seconds / perf.steps
                @test perf.mparticle_steps_per_second == perf.steps * length(state.particles) / 1e6 / perf.wall_time_seconds
                @test 0 <= perf.output_write_seconds <= perf.wall_time_seconds
                @test perf.phase_timings_synchronized == !async
                @test perf.phase_timings !== meta.HourGlass
                @test diag.particle_count == length(state.particles)
                @test diag.output_frames == length(output_times(f)) == meta.OutputIterationCounter
                @test diag.cell_list_rebuilds >= 1
                @test diag.step_state_readbacks >= 1
                @test all(>(0), diag.grid_dims)
                @test diag.frozen_particles == 0
                @test diag.grid_status_readbacks == 0
                @test diag.capacity_growths >= 0
                @test diag.bbox_host_readbacks == 0
                @test (diag.captured_graphs > 0) == async

                # Velocity was omitted from output; the final download must still refresh it.
                order = sortperm(state.particles.ID)
                @test all(v -> all(isfinite, v), state.particles.Velocity)
                @test any(!iszero, state.particles.Velocity)
                @test state.particles.Position[order] != initial_positions
                @test length(state.floating.center) == length(state.floating.velocity) == 1
                members = findall(==(Floating), state.particles.Type)
                @test state.floating.center[1] ≈ sum(state.particles.Position[members]) / length(members)
                @test state.floating.velocity[1] ≈ sum(state.particles.Velocity[members]) / length(members) atol = 1e-6
                @test hasproperty(state.floating, D == 2 ? :angle : :orientation)

                # Scalar diagnostics and copied timers survive later metadata mutations.
                calls = TimerOutputs.ncalls(perf.phase_timings["00 SimulationLoop"])
                reset_timer!(meta.HourGlass)
                meta.TotalTime = zero(T)
                meta.Iteration = 0
                @test result.final_state.time == state.time > 0
                @test result.final_state.iteration == state.iteration > 0
                @test TimerOutputs.ncalls(perf.phase_timings["00 SimulationLoop"]) == calls > 0
            end
        end
    end

    @testset "interrupt results and failure propagation" begin
        for async in (false, true), advance in (false, true)
            mktempdir() do save
                f = fixture(save, 2, Float32; async)
                result = run_fixture(f; viscosity = StopRun(InterruptException(), advance))
                @test result.termination_reason === :interrupted
                @test !result.final_state.complete
                @test result.final_state.particles === f.SimParticles
                @test result.final_state.floating === nothing
                @test result.final_state.time == f.SimMetaData.TotalTime
                @test result.diagnostics.output_frames == length(output_times(f)) == 1
                @test (result.performance.steps > 0) == advance
                if advance
                    @test result.final_state.time > 0
                    @test any(!iszero, result.final_state.particles.Velocity)
                else
                    @test result.performance.simulated_time == 0
                    @test isnan(result.performance.mean_timestep)
                    @test isnan(result.performance.milliseconds_per_step)
                    @test isnan(result.performance.mparticle_steps_per_second)
                end
            end
            mktempdir() do save
                f = fixture(save, 2, Float32; async)
                failure = ErrorException("injected solver failure")
                caught = try
                    run_fixture(f; viscosity = StopRun(failure, advance))
                catch error
                    error
                end
                @test caught === failure
                @test length(output_times(f)) == 1
            end
        end
    end
else
    @testset "simulation results require CUDA" begin
        @test_skip CUDA.functional()
    end
end

end # module
