module EffectiveConfigurationTests

using Test, SPHExampleGPU, CUDA, StaticArrays

const CONFIG = SPHExampleGPU.GPUConfiguration
const DRIVER = SPHExampleGPU.SPHCellList

function metadata(D, T; kwargs...)
    SimulationMetaData{D,T}(SimulationName = "EffectiveConfiguration", SaveLocation = tempdir(),
        SimulationTime = T(0.001), OutputTimes = T(0.001), GPULanesPerParticle = 1,
        VisualizeInParaview = false, OpenLogFile = false; kwargs...)
end

@testset "effective numerical configuration and capabilities" begin
    for D in (2, 3), T in (Float32, Float64)
        meta = metadata(D, T; GPURadialInteractions = true, GPUScalarSPSViscosity = true,
            GPUDoublePosition = true, GPUCacheInteractionFields = true, GPUCellSubdivision = 2)
        kernel = SPHKernelInstance{D,T}(WendlandC2(); dx = T(0.02))
        config = effective_configuration(meta, kernel, LinearDensityDiffusion(), LaminarSPS(); particle_count = 8)
        @test config.capabilities == (; radial_interactions = true, scalar_sps_viscosity = true)
        @test config.effective.sph_kernel === WendlandC2
        @test config.effective.interaction_kernel === :interaction_kernel!
        @test config.effective.interaction_path === :radial
        @test config.effective.density_diffusion_operator === :symmetric_reciprocal_density
        @test config.effective.viscosity_path === :scalar_sps
        @test config.effective.position_type === Float64
        @test config.effective.pair_coordinates === (T === Float32 ? :cell_relative : :absolute)
        @test config.effective.cache_interaction_fields
        @test config.effective.cell_subdivision == 2
        @test isempty(config.fallbacks)

        meta.GPUScalarSPSViscosity = false
        ordered_sps = effective_configuration(meta, kernel, LinearDensityDiffusion(), LaminarSPS(); particle_count = 8)
        @test ordered_sps.effective.viscosity_path === :reference_ordered
        meta.GPURadialInteractions = false
        reference = effective_configuration(meta, kernel, LinearDensityDiffusion(), LaminarSPS(); particle_count = 8)
        @test reference.capabilities.radial_interactions
        @test !reference.effective.radial_interactions
        @test reference.effective.density_diffusion_operator === :reference
        @test config.requested.radial_interactions # snapshots survive metadata edits

        meta.GPURadialInteractions = true
        meta.GPUScalarSPSViscosity = true
        meta.GPUSyncTimers = true
        meta.GPUUseGraph = true
        meta.GPUMaxStepsPerSync = 32
        cubic = SPHKernelInstance{D,T}(CubicSpline{T}(); dx = T(0.02))
        fallback = effective_configuration(meta, cubic, LinearDensityDiffusion(), ArtificialViscosity(); particle_count = 8)
        @test !fallback.capabilities.radial_interactions
        @test !fallback.capabilities.scalar_sps_viscosity
        @test fallback.effective.interaction_path === :reference
        @test fallback.effective.density_diffusion_operator === :reference
        @test fallback.effective.viscosity_path === :reference_ordered
        @test !fallback.effective.use_graph
        @test fallback.effective.max_steps_per_sync == 1
        @test map(f -> f.reason, fallback.fallbacks) ==
            (:unsupported_kernel_or_models, :unsupported_kernel_or_viscosity, :synchronized_timers)
        # Cubic kernels support scalar SPS independently of radial interactions.
        independent = effective_configuration(meta, cubic, ComplexDensityDiffusion(), LaminarSPS(); particle_count = 8)
        @test !independent.effective.radial_interactions
        @test independent.effective.scalar_sps_viscosity
        @test independent.effective.viscosity_path === :scalar_sps
        @test independent.effective.density_diffusion_operator === :reference

        disabled = effective_configuration(meta, kernel, ZeroDensityDiffusion(), ZeroViscosity(); particle_count = 0)
        @test disabled.effective.density_diffusion_operator === :disabled
        @test disabled.effective.viscosity_path === :disabled
        @test_throws ArgumentError effective_configuration(meta, kernel, ZeroDensityDiffusion(), ZeroViscosity(); particle_count = -1)
    end
end

# Exercise the real capture/fallback accounting with one deterministic device update.
struct CaptureContext{S, G, C}
    state::S
    gpu::G
    cl::C
    fail_in_capture::Base.RefValue{Bool}
end

CaptureContext(state, gpu, cl) = CaptureContext(state, gpu, cl, Ref(false))

function increment_kernel!(values)
    values[1] += 1
    return nothing
end

function DRIVER.enqueue_step!(ctx::CaptureContext, timed::Bool)
    ctx.fail_in_capture[] && error("synthetic kernel linkage failure")
    @cuda threads=1 increment_kernel!(ctx.gpu.Position)
    return nothing
end

function DRIVER.prepare_step_graph!(ctx::CaptureContext)
    @cuda launch=false increment_kernel!(ctx.gpu.Position)
    ctx.state.graph_kernels_prepared = true
    return nothing
end

if CUDA.functional()
    function run_case(save; kernel_model = WendlandC2(), diffusion = LinearDensityDiffusion(),
                      viscosity = ArtificialViscosity(), requested = true, log_mode = StoreLog,
                      timed = false)
        T = Float32
        points = [SVector{2,T}(x, y) for x in T[0, 0.02], y in T[0, 0.02]][:]
        geometry = [SPHGeometry{2,T}(points; Density = T[999, 1001, 1003, 998], Type = Fluid, GroupMarker = 1)]
        meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,log_mode}(
            SimulationName = "EffectiveConfiguration", SaveLocation = save,
            SimulationTime = T(0.001), OutputTimes = T(0.001),
            GPURadialInteractions = requested, GPUScalarSPSViscosity = requested,
            GPULanesPerParticle = 0, GPUSyncTimers = timed, GPUUseGraph = timed,
            VisualizeInParaview = false, OpenLogFile = false)
        constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(20), g = zero(T))
        kernel = SPHKernelInstance{2,T}(kernel_model; dx = constants.dx)
        particles = AllocateDataStructures(geometry, meta)
        logger = SimulationLogger(save)
        result = try
            redirect_stdout(devnull) do
                RunSimulation(; SimGeometry = geometry, SimMetaData = meta, SimConstants = constants,
                    SimKernel = kernel, SimParticles = particles, SimLogger = logger,
                    SimViscosity = viscosity, SimDensityDiffusion = diffusion,
                    SimTimeStepping = SymplecticTimeStepping())
            end
        finally
            close(logger.LoggerIo)
        end
        return (; result, log = read(joinpath(save, "SimulationOutput.log"), String))
    end

    @testset "simulation reports resolved fallbacks with and without logging" begin
        for log_mode in (NoLog, StoreLog)
            mktempdir() do save
                fallback = run_case(save; kernel_model = CubicSpline{Float32}(), log_mode, timed = true)
                config = fallback.result.configuration
                @test config.effective.sph_kernel === CubicSpline{Float32}
                @test config.effective.interaction_path === :reference
                @test config.effective.density_diffusion_operator === :reference
                @test !config.effective.scalar_sps_viscosity
                @test config.effective.lanes_per_particle == choose_lanes(4, Val(2))
                @test config.execution.path === :direct
                @test config.execution.graph_capture_failures == 0
                @test length(config.fallbacks) == 3
                if log_mode === StoreLog
                    @test occursin("Effective GPU configuration (final):", fallback.log)
                    @test occursin("operator=reference", fallback.log)
                    @test occursin("reason=unsupported_kernel_or_models", fallback.log)
                    @test occursin("reason=unsupported_kernel_or_viscosity", fallback.log)
                    @test occursin("reason=synchronized_timers", fallback.log)
                else
                    @test isempty(fallback.log)
                end
                mktempdir() do reference_save
                    reference = run_case(reference_save; kernel_model = CubicSpline{Float32}(),
                        requested = false, log_mode, timed = true)
                    @test reference.result.final_state.particles == fallback.result.final_state.particles
                    @test reference.result.final_state.iteration == fallback.result.final_state.iteration
                end
            end
        end
        mktempdir() do save
            independent = run_case(save; diffusion = ComplexDensityDiffusion(), viscosity = LaminarSPS())
            config = independent.result.configuration
            @test !config.effective.radial_interactions
            @test config.effective.scalar_sps_viscosity
            @test config.effective.viscosity_path === :scalar_sps
            @test config.effective.density_diffusion_operator === :reference
            @test only(config.fallbacks).option === :GPURadialInteractions
        end
    end

    @testset "graph capture failure, retry and persistent execution snapshot" begin
        state = StepState{Float32}()
        meta = metadata(2, Float32)
        kernel = SPHKernelInstance{2,Float32}(WendlandC2(); dx = 0.02f0)
        state.configuration = effective_configuration(meta, kernel, LinearDensityDiffusion(), LaminarSPS(); particle_count = 1)
        ctx = CaptureContext(state, (; Position = CUDA.zeros(Int32, 1)), (; generation = 0))
        @test CONFIG.configuration_snapshot(state).execution.path === :not_run
        fail_capture(callback; kwargs...) = nothing
        @test_logs (:warn, r"CUDA graph capture failed for a 2-step batch.*Graphs remain enabled.*Cause:") DRIVER.launch_step_graph!(ctx, 2; capture = fail_capture)
        first_snapshot = CONFIG.configuration_snapshot(state)
        @test first_snapshot.execution == (; path = :direct, graph_batches = 0, direct_batches = 1, graph_capture_failures = 1)
        @test last(first_snapshot.fallbacks).reason === :graph_capture_failed
        @test occursin("without exposing an underlying exception", last(first_snapshot.fallbacks).detail)
        @test state.graph_kernels_prepared
        @test only(Array(ctx.gpu.Position)) == 2
        @test isempty(state.graphs)
        # Repeated failures count every batch and emit only the first warning.
        @test_logs DRIVER.launch_step_graph!(ctx, 2; capture = fail_capture)
        @test only(Array(ctx.gpu.Position)) == 4
        DRIVER.launch_step_graph!(ctx, 2)
        DRIVER.launch_step_graph!(ctx, 2)
        @test only(Array(ctx.gpu.Position)) == 8
        @test length(state.graphs) == 1
        SPHExampleGPU.GPUStepState.invalidate_graphs!(state)
        @test !state.graph_kernels_prepared
        snapshot = CONFIG.configuration_snapshot(state)
        @test snapshot.execution == (; path = :mixed, graph_batches = 2, direct_batches = 2, graph_capture_failures = 2)
        @test first_snapshot.execution.graph_capture_failures == 1
        @test last(snapshot.fallbacks).reason === :graph_capture_failed
    end

    @testset "capture invalidation preserves the original exception" begin
        state = StepState{Float32}()
        meta = metadata(2, Float32)
        kernel = SPHKernelInstance{2,Float32}(WendlandC2(); dx = 0.02f0)
        state.configuration = effective_configuration(meta, kernel, LinearDensityDiffusion(), LaminarSPS(); particle_count = 1)
        ctx = CaptureContext(state, (; Position = CUDA.zeros(Int32, 1)), (; generation = 0))
        # Model CUDA.capture's finally block suppressing the callback exception.
        function suppress_error(callback; kwargs...)
            ctx.fail_in_capture[] = true
            try
                callback()
            catch
                return nothing
            finally
                ctx.fail_in_capture[] = false
            end
        end
        @test_logs (:warn, r"Cause: synthetic kernel linkage failure") DRIVER.launch_step_graph!(ctx, 2; capture = suppress_error)
        @test only(Array(ctx.gpu.Position)) == 2
        first_snapshot = CONFIG.configuration_snapshot(state)
        @test last(first_snapshot.fallbacks).detail == "synthetic kernel linkage failure"
        DRIVER.launch_step_graph!(ctx, 2)
        @test only(Array(ctx.gpu.Position)) == 4
        SPHExampleGPU.GPUStepState.invalidate_graphs!(state)
        @test CONFIG.configuration_snapshot(state).execution.path === :mixed
        @test last(CONFIG.configuration_snapshot(state).fallbacks).detail == "synthetic kernel linkage failure"
    end
end

end # module
