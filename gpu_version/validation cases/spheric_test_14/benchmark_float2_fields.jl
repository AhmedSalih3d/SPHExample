# Paired whole-driver comparison on the full-width Float 2 geometry.
include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "full_case.jl"))
using CUDA

function run_float2_variant(candidate, enabled, duration)
    mktempdir() do path
        release = merge(last(RELEASES), (; draft = 0.21988712735907776))
        case = build_full_case(release; half_width = 2.4, duration, output_interval = 0.1,
            save_location = path)
        case.metadata.GPUCacheInteractionFields = candidate == :cache && enabled
        case.metadata.GPUInteractionThreads = candidate == :threads && enabled ? 256 : 128
        sampling = (; times = collect(0.005:0.005:duration), floating_history = true)
        case.metadata.OutputTimes = sampling.times[20:20:end]
        logger = SimulationLogger(path)
        host = AllocateDataStructures(case.geometry, case.metadata)
        result = try
            redirect_stdout(devnull) do
                RunSimulation(SimGeometry = case.geometry, SimMetaData = case.metadata,
                    SimConstants = case.constants, SimKernel = case.kernel, SimLogger = logger,
                    SimParticles = host, SimViscosity = case.viscosity, SimDensityDiffusion = case.diffusion,
                    SimTimeStepping = case.stepping, SimDamping = case.damping, SimStateSampling = sampling)
            end
        finally
            isopen(logger.LoggerIo) && close(logger.LoggerIo)
        end
        d = result.diagnostics
        result.final_state.complete && d.frozen_particles == 0 || error("Incomplete benchmark")
        body = CSV.File(joinpath(path, "FreeHeave_Floating.csv"))
        order = sortperm(host.ID)
        return (; stats = (wall_s = result.performance.wall_time_seconds, steps = result.performance.steps,
            rebuilds = d.cell_list_rebuilds, readbacks = d.step_state_readbacks,
            grid_readbacks = d.grid_status_readbacks, fluid_frames = d.output_frames, body_samples = length(body)),
            center = collect(getproperty(body, Symbol("Center:2"))),
            force = collect(getproperty(body, Symbol("Force:2"))),
            id = host.ID[order], position = host.Position[order], density = host.Density[order])
    end
end

function benchmark_float2_gpu(candidate, duration)
    println("GPU: ", CUDA.name(CUDA.device()))
    for enabled in (false, true)
        run_float2_variant(candidate, enabled, 0.1)
    end
    rows = NamedTuple[]
    for (pair, order) in enumerate(((false, true), (true, false)))
        runs = map(enabled -> run_float2_variant(candidate, enabled, duration), order)
        baseline, variant = order[1] ? reverse(runs) : runs
        baseline.id == variant.id || error("Particle IDs differ")
        push!(rows, (; pair, duration_s = duration, first_variant = order[1] ? string(candidate) : "baseline",
            baseline_s = baseline.stats.wall_s, variant_s = variant.stats.wall_s,
            variant_speedup = baseline.stats.wall_s / variant.stats.wall_s,
            baseline_steps = baseline.stats.steps, variant_steps = variant.stats.steps,
            baseline_rebuilds = baseline.stats.rebuilds, variant_rebuilds = variant.stats.rebuilds,
            baseline_readbacks = baseline.stats.readbacks, variant_readbacks = variant.stats.readbacks,
            baseline_grid_readbacks = baseline.stats.grid_readbacks, variant_grid_readbacks = variant.stats.grid_readbacks,
            baseline_fluid_frames = baseline.stats.fluid_frames, variant_fluid_frames = variant.stats.fluid_frames,
            baseline_body_samples = baseline.stats.body_samples, variant_body_samples = variant.stats.body_samples,
            max_heave_delta_mm = 1000maximum(abs, baseline.center - variant.center),
            max_vertical_force_delta_n = maximum(abs, baseline.force - variant.force),
            max_position_delta_m = maximum(norm, baseline.position - variant.position),
            max_density_delta_kg_m3 = maximum(abs, baseline.density - variant.density)))
        println(last(rows))
    end
    folder = joinpath(@__DIR__, "results", "benchmark_3d_48m")
    suffix = duration == 0.1 ? "" : "_$(round(Int, 1000duration))ms"
    CSV.write(joinpath(folder, "float2_$(candidate)_benchmark$suffix.csv"), rows)
end

durations = filter(!=("--threads"), ARGS)
length(durations) <= 1 || error("Usage: benchmark_float2_fields.jl [--threads] [duration]")
duration = isempty(durations) ? 0.1 : parse(Float64, only(durations))
isfinite(duration) && duration >= 0.1 && isapprox(duration, 0.1round(duration / 0.1); atol = 1e-12) ||
    error("Duration must be a positive multiple of 0.1 s")
benchmark_float2_gpu("--threads" in ARGS ? :threads : :cache, duration)
