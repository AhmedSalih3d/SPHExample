# CPU baseline benchmark. Run from the repository root with, for example
#
#     julia -t 24,0 --project=. gpu_version/benchmark/benchmark_cpu.jl [--repeat N] [case substrings...]
#
# Every case is first run for a single output interval of negligible length to
# compile everything, then timed `N` times (default 2) and the best time is
# reported. Prints the wall time spent in the time stepping loop and the time
# per step. Case definitions are shared with the GPU benchmark, see `cases.jl`.
#
# Note: on Julia 1.12 use `-t N,0` rather than `-t auto`; `-t auto` adds an
# interactive thread and the CPU code indexes its per thread arrays with
# `threadid()`, which then exceeds `nthreads()`.

using SPHExample
using TimerOutputs
using Printf

include(joinpath(@__DIR__, "cases.jl"))

const FloatType = Float64

function run_case(case::BenchCase, ::Type{T}; warmup = false) where {T}
    save = mktempdir()
    kw   = case.build(T, save)
    if warmup
        kw.SimMetaData.SimulationTime = T(1e-7)
        kw.SimMetaData.OutputTimes    = T(1e-7)
    end
    particles = AllocateDataStructures(kw.SimGeometry)
    logger    = SimulationLogger(save; to_console=false)
    RunSimulation(; kw..., SimLogger=logger, SimParticles=particles)
    hg    = kw.SimMetaData.HourGlass
    loop  = TimerOutputs.time(hg["00 SimulationLoop"]) / 1e9
    total = TimerOutputs.tottime(hg) / 1e9
    iters = kw.SimMetaData.Iteration
    return (n=length(particles), iters=iters, loop=loop, total=total)
end

function getopt(args, name, default)
    i = findfirst(==(name), args)
    i === nothing && return default, args
    val = parse(Int, args[i + 1])
    return val, [args[1:i-1]; args[i+2:end]]
end

function main(args)
    repeat, args = getopt(args, "--repeat", 2)
    names = filter(a -> !startswith(a, "--"), args)
    cases = select_cases(names)
    header = "case (CPU, $(Threads.nthreads()) threads, $FloatType, best of $repeat)"
    @printf("%-60s %8s %6s %10s %10s %10s\n", header, "N", "steps", "loop [s]", "ms/step",
            "total [s]")
    for c in cases
        run_case(c, FloatType; warmup = true)   # compile
        GC.gc()
        best = nothing
        for _ in 1:repeat
            r = run_case(c, FloatType)
            if best === nothing || r.loop / r.iters < best.loop / best.iters
                best = r
            end
            GC.gc()
        end
        r = best
        @printf("%-60s %8d %6d %10.3f %10.3f %10.3f\n", c.name, r.n, r.iters, r.loop,
                1e3 * r.loop / r.iters, r.total)
        flush(stdout)
    end
end

main(ARGS)
