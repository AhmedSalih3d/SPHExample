# Sensitivity of the GPU run time to the output interval. Run from the
# repository root with, for example
#
#     julia --project=gpu_version gpu_version/benchmark/bench_output_interval.jl --float32 --time 0.5 --intervals 0.0025,0.01,0.05,0.5 DamBreak2D
#
# Every selected case (see `cases.jl`) is run once per interval for `--time`
# seconds of simulated time (after a compile run) and the wall time, the time
# stepping loop time and the number of frames written are reported. Pass
# `--grid` to also export the cell grid, `--sync-output` to write the frames on
# the simulation thread (`GPUAsyncOutput = false`), `--buffer-mib M` to set the
# host memory for buffered output frames (`GPUOutputBufferBytes`; `0` writes
# every frame at once) and `--timers` to print the timer table and the frame
# write time of every run.

using SPHExampleGPU
using CUDA
using TimerOutputs
using Printf

include(joinpath(@__DIR__, "cases.jl"))

function run_case(case::BenchCase, ::Type{T}, sim_time, interval; grid = false, warmup = false, timers = false, async = true,
                  buffer_bytes = nothing) where {T}
    save = mktempdir()
    kw   = case.build(T, save)
    kw.SimMetaData.SimulationTime  = T(warmup ? 1e-7 : sim_time)
    kw.SimMetaData.OutputTimes     = T(warmup ? 1e-7 : interval)
    kw.SimMetaData.ExportGridCells = grid
    kw.SimMetaData.GPUAsyncOutput  = async
    buffer_bytes === nothing || (kw.SimMetaData.GPUOutputBufferBytes = buffer_bytes)
    particles = AllocateDataStructures(kw.SimGeometry, kw.SimMetaData)
    logger    = SimulationLogger(save; to_console = false)
    gc0 = Base.gc_num()
    t0  = time_ns()
    redirect_stdout(devnull) do
        RunSimulation(; kw..., SimLogger = logger, SimParticles = particles)
    end
    wall = (time_ns() - t0) / 1e9
    gcd  = Base.GC_Diff(Base.gc_num(), gc0)
    hg   = kw.SimMetaData.HourGlass
    if timers && !warmup
        show(hg; sortby = :name); println()
        for line in eachline(joinpath(save, "SimulationOutput.log"))
            occursin("write time", line) && println(line)
        end
    end
    return (n = length(particles), iters = kw.SimMetaData.Iteration, frames = kw.SimMetaData.OutputIterationCounter,
            wall = wall, loop = loop_time(hg), total = TimerOutputs.tottime(hg) / 1e9,
            gc = gcd.total_time / 1e9, alloc = gcd.allocd / 2^20)
end

function getopt(args, name, default, parser)
    i = findfirst(==(name), args)
    i === nothing && return default, args
    return parser(args[i + 1]), [args[1:i-1]; args[i+2:end]]
end

function main(args)
    T    = "--float32" in args ? Float32 : Float64
    grid = "--grid" in args
    timers = "--timers" in args
    async  = !("--sync-output" in args)
    sim_time,  args = getopt(args, "--time", 0.5, s -> parse(Float64, s))
    intervals, args = getopt(args, "--intervals", [0.0025, 0.01, 0.05, 0.5], s -> parse.(Float64, split(s, ',')))
    buffer_mib, args = getopt(args, "--buffer-mib", nothing, s -> parse(Float64, s))
    buffer_bytes = buffer_mib === nothing ? nothing : round(Int, buffer_mib * 2^20)
    names = filter(a -> !startswith(a, "--"), args)
    cases = select_cases(names)

    println("GPU $(CUDA.name(CUDA.device())), $T, $(Threads.nthreads()) threads, simulated time $sim_time s, grid export $grid, async output $async, " *
            "output buffer $(buffer_mib === nothing ? "default" : "$(buffer_mib) MiB")")
    @printf("%-32s %10s %8s %6s %7s %10s %10s %10s %8s %11s\n", "case", "interval", "N", "steps", "frames", "wall [s]", "loop [s]", "ms/step", "GC [s]", "alloc [MiB]")
    for c in cases
        run_case(c, T, sim_time, first(intervals); grid = grid, warmup = true)
        GC.gc(); CUDA.reclaim()
        for dt_out in intervals
            r = run_case(c, T, sim_time, dt_out; grid = grid, timers = timers, async = async, buffer_bytes = buffer_bytes)
            @printf("%-32s %10.4f %8d %6d %7d %10.3f %10.3f %10.3f %8.3f %11.1f\n", c.name, dt_out, r.n, r.iters, r.frames,
                    r.wall, r.loop, 1e3 * r.loop / r.iters, r.gc, r.alloc)
            flush(stdout)
            GC.gc(); CUDA.reclaim()
        end
    end
end

main(ARGS)
