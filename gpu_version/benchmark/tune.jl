# Tuning sweep for the GPU kernels: lanes per particle and threads per block.
#
#     julia --project=gpu_version gpu_version/benchmark/tune.jl [--float64] [--repeat N] [case substrings...]
#
# All settings of a case are compiled first, then measured `N` times in an
# interleaved fashion (setting 1, 2, ..., setting 1, 2, ...) and the best time
# per setting is reported. Interleaving and taking the minimum makes the
# comparison robust against clock/thermal throttling of laptop GPUs.
using SPHExampleGPU
using CUDA
using TimerOutputs
using Printf

include(joinpath(@__DIR__, "cases.jl"))

function run_case(case::BenchCase, ::Type{T}; lanes, threads, bforces = true, warmup = false) where {T}
    save = mktempdir()
    kw   = case.build(T, save)
    if warmup
        kw.SimMetaData.SimulationTime = T(1e-7)
        kw.SimMetaData.OutputTimes    = T(1e-7)
    end
    kw.SimMetaData.GPULanesPerParticle   = lanes
    kw.SimMetaData.GPUInteractionThreads = threads
    kw.SimMetaData.GPUBoundaryForces     = bforces
    particles = AllocateDataStructures(kw.SimGeometry)
    logger    = SimulationLogger(save; to_console = false)
    RunSimulation(; kw..., SimLogger = logger, SimParticles = particles)
    hg    = kw.SimMetaData.HourGlass
    loop  = TimerOutputs.time(hg["00 SimulationLoop"]) / 1e9
    iters = kw.SimMetaData.Iteration
    return (n = length(particles), iters = iters, loop = loop)
end

function getopt(args, name, default)
    i = findfirst(==(name), args)
    i === nothing && return default, args
    val = parse(Int, args[i + 1])
    return val, [args[1:i-1]; args[i+2:end]]
end

function main(args)
    T = "--float64" in args ? Float64 : Float32
    repeat, args = getopt(args, "--repeat", 3)
    names = filter(a -> !startswith(a, "--"), args)
    cases = select_cases(names)
    settings = if "--quick" in args
        [(lanes = 0, threads = 128, bforces = true), (lanes = 0, threads = 128, bforces = false),
         (lanes = 2, threads = 128, bforces = true), (lanes = 2, threads = 128, bforces = false)]
    else
        [(lanes = 1, threads = 128, bforces = true), (lanes = 0, threads = 128, bforces = true),
         (lanes = 2, threads = 128, bforces = true), (lanes = 1, threads = 256, bforces = true),
         (lanes = 0, threads = 256, bforces = true), (lanes = 2, threads = 256, bforces = true),
         (lanes = 1, threads = 64, bforces = true), (lanes = 0, threads = 128, bforces = false)]
    end
    @printf("%-32s %8s %8s %6s %8s %8s %12s %12s\n", "case ($T, best of $repeat)", "N", "lanes", "thr",
            "bforces", "steps", "ms/step", "worst")
    for c in cases
        for s in settings
            run_case(c, T; s..., warmup = true)
        end
        GC.gc(); CUDA.reclaim()
        best  = fill(Inf, length(settings))
        worst = fill(0.0, length(settings))
        n = iters = 0
        for _ in 1:repeat, (k, s) in enumerate(settings)
            r = run_case(c, T; s...)
            t = 1e3 * r.loop / r.iters
            best[k]  = min(best[k], t)
            worst[k] = max(worst[k], t)
            n, iters = r.n, r.iters
            GC.gc(); CUDA.reclaim()
        end
        for (k, s) in enumerate(settings)
            auto = s.lanes == 0 ? choose_lanes(n) : s.lanes
            @printf("%-32s %8d %8s %6d %8s %8d %12.3f %12.3f\n", c.name, n,
                    s.lanes == 0 ? "auto=$auto" : string(s.lanes), s.threads, s.bforces, iters, best[k], worst[k])
        end
        flush(stdout)
    end
end

main(ARGS)
