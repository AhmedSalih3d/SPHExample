using SPHExampleGPU, CUDA, Statistics, Printf, TimerOutputs, LinearAlgebra
include("still_wedge_instructions.jl")
include("cases.jl")

function measure_case(variant, case, duration)
    mktempdir() do save
        kw = case.build(Float32, save)
        meta = kw.SimMetaData
        meta.SimulationTime = Float32(duration)
        meta.OutputTimes = Float32(duration)
        meta.GPURadialInteractions = true
        meta.GPUCacheInteractionFields = true
        meta.GPUDoublePosition = true
        meta.GPUXCellSubdivision = 4
        model = kw.SimDensityDiffusion
        replacement = Base.invokelatest(getproperty(variant.diffusion, nameof(typeof(model))))
        kw = merge(kw, (; SimDensityDiffusion = replacement))
        host = AllocateDataStructures(kw.SimGeometry, meta)
        logger = SimulationLogger(save)
        GC.gc(); CUDA.synchronize()
        seconds = try
            redirect_stdout(devnull) do
                @elapsed begin
                    Base.invokelatest(variant.driver.RunSimulation; kw..., SimParticles = host, SimLogger = logger)
                    CUDA.synchronize()
                end
            end
        finally
            close(logger.LoggerIo)
        end
        order = sortperm(host.ID)
        state = map(name -> getproperty(host, name)[order],
            (:Position, :Velocity, :Density, :Acceleration, :ID, :Type))
        loop = meta.HourGlass["00 SimulationLoop"]
        work = (; steps = meta.Iteration,
            rebuilds = TimerOutputs.ncalls(loop["02a Actual Calculate IndexCounter"]),
            readbacks = TimerOutputs.ncalls(loop["13 Read Back State"]))
        return (; seconds, loop_seconds = TimerOutputs.time(loop) / 1e9,
            time = meta.TotalTime, work, state, particles = length(host))
    end
end

function main_cases(args)
    option(key, default) = begin
        values = filter(a -> startswith(a, key * "="), args)
        isempty(values) ? default : split(only(values), '='; limit = 2)[2]
    end
    rounds = parse(Int, option("--rounds", "5"))
    duration = parse(Float64, option("--duration", "0.01"))
    names = filter(a -> !startswith(a, "--"), args)
    cases = select_cases(isempty(names) ?
        ["StillWedge2D_DBC_dp0.01", "DamBreak3D_dp0.02", "DamBreak3D_dp0.0085"] : names)
    checked = checked_driver()
    optimized = (; driver = SPHExampleGPU.SPHCellList,
        diffusion = SPHExampleGPU.SPHDensityDiffusionModels)
    for case in cases
        variants = (checked, optimized)
        for variant in variants
            measure_case(variant, case, duration)
        end
        times = (Float64[], Float64[])
        loops = (Float64[], Float64[])
        for round in 1:rounds
            results = Vector{NamedTuple}(undef, 2)
            for i in (isodd(round) ? (1, 2) : (2, 1))
                results[i] = measure_case(variants[i], case, duration)
                push!(times[i], results[i].seconds)
                push!(loops[i], results[i].loop_seconds)
            end
            a, b = results
            a.work == b.work || error("work counts differ for $(case.name) round $round: $(a.work), $(b.work)")
            errors = map((x, y) -> maximum(norm.(x .- y)), a.state[1:4], b.state[1:4])
            @printf("case=%s round=%d particles=%d checked=%.6fs optimized=%.6fs ratio=%.4f loop_ratio=%.4f time_diff=%.9g max_errors=%s exact=%s work=%s\n",
                case.name, round, a.particles, a.seconds, b.seconds, b.seconds/a.seconds,
                b.loop_seconds/a.loop_seconds, b.time-a.time, string(errors),
                string(a.time == b.time && isequal(a.state, b.state)), string(a.work))
            flush(stdout)
        end
        ratios = times[2] ./ times[1]
        loop_ratios = loops[2] ./ loops[1]
        @printf("summary case=%s median_ratio=%.4f range=[%.4f,%.4f] median_loop_ratio=%.4f faster=%d/%d\n",
            case.name, median(ratios), minimum(ratios), maximum(ratios), median(loop_ratios), count(<(1), ratios), rounds)
    end
end

main_cases(ARGS)
