# GPU benchmark. Run from the repository root with, for example
#
#     julia --project=gpu_version gpu_version/benchmark/benchmark_gpu.jl [--float32] [--repeat N] [--timers] [case substrings...]
#
# Every case is first run for a single output interval of negligible length to
# compile all kernels, then timed `N` times (default 3); the best time is
# reported (laptop GPUs throttle, the minimum is the least noisy statistic).
# Prints the wall time spent in the time stepping loop and the time per step.
# Case definitions are shared with the CPU benchmark, see `cases.jl`.

using SPHExampleGPU
using CUDA
using TimerOutputs
using Printf

include(joinpath(@__DIR__, "cases.jl"))

function run_case(case::BenchCase, ::Type{T}; sync = false, warmup = false) where {T}
    save = mktempdir()
    kw   = case.build(T, save)
    if warmup
        kw.SimMetaData.SimulationTime = T(1e-7)
        kw.SimMetaData.OutputTimes    = T(1e-7)
    end
    kw.SimMetaData.GPUSyncTimers = sync
    particles = AllocateDataStructures(kw.SimGeometry)
    logger    = SimulationLogger(save; to_console = false)
    RunSimulation(; kw..., SimLogger = logger, SimParticles = particles)
    hg    = kw.SimMetaData.HourGlass
    loop  = TimerOutputs.time(hg["00 SimulationLoop"]) / 1e9
    total = TimerOutputs.tottime(hg) / 1e9
    iters = kw.SimMetaData.Iteration
    return (n = length(particles), iters = iters, loop = loop, total = total, hg = hg)
end

function getopt(args, name, default)
    i = findfirst(==(name), args)
    i === nothing && return default, args
    val = parse(Int, args[i + 1])
    return val, [args[1:i-1]; args[i+2:end]]
end

function main(args)
    T     = "--float32" in args ? Float32 : Float64
    sync  = "--sync" in args
    show_timers = "--timers" in args
    repeat, args = getopt(args, "--repeat", 3)
    names = filter(a -> !startswith(a, "--"), args)
    cases = select_cases(names)

    dev = CUDA.device()
    header = "case (GPU $(CUDA.name(dev)), $T, best of $repeat)"
    @printf("%-60s %8s %6s %10s %10s %10s\n", header, "N", "steps", "loop [s]", "ms/step", "total [s]")
    for c in cases
        run_case(c, T; sync = sync, warmup = true)   # compile
        GC.gc(); CUDA.reclaim()
        best = nothing
        for _ in 1:repeat
            r = run_case(c, T; sync = sync)
            if best === nothing || r.loop / r.iters < best.loop / best.iters
                best = r
            end
            GC.gc(); CUDA.reclaim()
        end
        r = best
        @printf("%-60s %8d %6d %10.3f %10.3f %10.3f\n", c.name, r.n, r.iters, r.loop,
                1e3 * r.loop / r.iters, r.total)
        show_timers && (show(r.hg; sortby = :name); println())
        flush(stdout)
    end
end

main(ARGS)
