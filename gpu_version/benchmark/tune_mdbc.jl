# Run from gpu_version: julia --project=. benchmark/tune_mdbc.jl [--quick] [--float64] [case substrings...]
using SPHExampleGPU, CUDA, LinearAlgebra, Printf, Statistics

include(joinpath(@__DIR__, "cases.jl"))

function run_mdbc_case(case, ::Type{T}, shape, quick) where {T}
    return mktempdir() do save
        kw = case.build(T, save)
        meta = kw.SimMetaData
        meta.GPUMDBCThreads, meta.GPUMDBCLanesPerGhost = shape
        meta.GPUSyncTimers = false
        meta.GPUUseGraph = true
        meta.GPUAsyncOutput = false
        if quick
            meta.SimulationTime = min(meta.SimulationTime, T(0.02))
            meta.OutputTimes = meta.SimulationTime
        end
        particles = AllocateDataStructures(kw.SimGeometry, meta)
        logger = SimulationLogger(save)
        result = try
            redirect_stdout(devnull) do
                RunSimulation(; kw..., SimLogger = logger, SimParticles = particles)
            end
        finally
            isopen(logger.LoggerIo) && close(logger.LoggerIo)
        end
        result.termination_reason === :completed || error("$(case.name) did not finish")
        e, d = result.configuration.effective, result.diagnostics
        order = sortperm(particles.ID)
        state = (; ids = particles.ID[order], position = particles.Position[order],
            velocity = particles.Velocity[order], density = particles.Density[order])
        return (; time = result.performance.wall_time_seconds, simulated_time = result.performance.simulated_time, state,
            work = (; steps = result.performance.steps, rebuilds = d.cell_list_rebuilds,
                readbacks = d.step_state_readbacks, frames = d.output_frames,
                execution = result.configuration.execution),
            particles = d.particle_count, ghosts = count(!=(Fluid), particles.Type),
            force = (e.interaction_threads, e.lanes_per_particle),
            mdbc = (e.mdbc_threads, e.mdbc_lanes_per_ghost))
    end
end

function state_error(a, b)
    a.ids == b.ids || return Inf
    return maximum(maximum(norm(x - y) / max(1, norm(x), norm(y)) for (x, y) in zip(field_a, field_b))
        for (field_a, field_b) in ((a.position, b.position), (a.velocity, b.velocity), (a.density, b.density)))
end

function main(args)
    CUDA.functional() || error("The mDBC sweep requires a functional CUDA device")
    T = "--float64" in args ? Float64 : Float32
    quick = "--quick" in args
    names = filter(a -> !startswith(a, "--"), args)
    cases = filter(c -> occursin("MDBC", c.name),
        select_cases(isempty(names) ? ["StillWedge2D_MDBC", "Duckling3D_MDBC_dp0.01"] : names))
    isempty(cases) && error("Select at least one mDBC benchmark case")
    shapes = quick ? [(64, 4), (128, 4), (128, 8)] :
        [(threads, lanes) for threads in (64, 128, 256) for lanes in (1, 2, 4, 8, 16, 32)]
    baseline_shape = (0, 0)
    repeats = 3
    tolerance = T === Float32 ? 5e-5 : 1e-10
    println(CUDA.name(CUDA.device()), ", ", T, ", ", repeats, " paired runs per shape")
    for case in cases
        baseline = run_mdbc_case(case, T, baseline_shape, quick)
        @printf("%s | %d particles, %d wall particles | force %d threads × %d lanes | baseline mDBC %d × %d\n",
            case.name, baseline.particles, baseline.ghosts, baseline.force..., baseline.mdbc...)
        println("mDBC threads/lanes | baseline/variant median wall time (s) | paired speed ratio [range] | steps/rebuilds/readbacks | state error | comparable")
        for shape in shapes
            shape == baseline.mdbc && continue
            run_mdbc_case(case, T, shape, quick)
            baseline_times, variant_times = Float64[], Float64[]
            error_bound, comparable = 0.0, true
            work = baseline.work
            for repeat in 1:repeats
                if isodd(repeat)
                    ref = run_mdbc_case(case, T, baseline_shape, quick)
                    variant = run_mdbc_case(case, T, shape, quick)
                else
                    variant = run_mdbc_case(case, T, shape, quick)
                    ref = run_mdbc_case(case, T, baseline_shape, quick)
                end
                ref.force == variant.force == baseline.force || error("Force gather shape changed")
                variant.mdbc == shape || error("mDBC launch shape did not take effect")
                work = variant.work
                error_bound = max(error_bound, state_error(ref.state, variant.state))
                same_work = ref.work == variant.work && work.execution.graph_capture_failures == 0
                if !same_work
                    println("  Work differs or graph capture failed for ", shape, ", pair ", repeat,
                        "\n    baseline ", ref.work, "\n    variant  ", variant.work)
                end
                comparable &= same_work && error_bound <= tolerance &&
                    isapprox(ref.simulated_time, variant.simulated_time; rtol = 8eps(T))
                push!(baseline_times, ref.time)
                push!(variant_times, variant.time)
            end
            ratios = baseline_times ./ variant_times
            @printf("%4d / %2d | %9.4f / %9.4f | %7.3f [%.3f, %.3f] | %d/%d/%d | %.3g | %s\n",
                shape..., median(baseline_times), median(variant_times), median(ratios), extrema(ratios)...,
                work.steps, work.rebuilds, work.readbacks, error_bound, comparable ? "yes" : "no")
            flush(stdout)
        end
    end
end

main(ARGS)
