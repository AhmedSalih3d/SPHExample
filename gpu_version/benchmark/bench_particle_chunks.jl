# Compare frame/batch-sized HDF5 chunks with bounded particle chunks in a
# fresh Julia process. Run from the repository root:
#
# julia --project=gpu_version -t 2,1 gpu_version/benchmark/bench_particle_chunks.jl
#
# Defaults match MovingSquare2D's output interval. Use --interval 0.001 --time 0.1
# for an output-heavy run. Both variants use the same buffer accounting. This
# benchmark overrides only the chunk selector in its own process, warms both
# variants, alternates their order and checks all output data and solver counters.

using SPHExampleGPU, CUDA, HDF5, Printf
include(joinpath(@__DIR__, "cases.jl"))

const LEGACY_PARTICLE_CHUNKS = Ref(false)
const OUTPUT = SPHExampleGPU.ProduceHDFVTK
@eval OUTPUT particle_chunk_size(points_per_flush::Integer, bytes_per_point::Integer) =
    $(LEGACY_PARTICLE_CHUNKS)[] ? points_per_flush :
    min(points_per_flush, max(1, MAX_PARTICLE_CHUNK_BYTES ÷ bytes_per_point))

function chunk_benchmark_run(legacy, dx, duration, interval; warmup = false)
    LEGACY_PARTICLE_CHUNKS[] = legacy
    return mktempdir() do dir
        kw = moving_square_2d(Float32, dir; dx)
        meta = kw.SimMetaData
        meta.SimulationTime = warmup ? 2interval : duration
        meta.OutputTimes = interval
        particles = AllocateDataStructures(kw.SimGeometry, meta)
        logger = SimulationLogger(dir)
        GC.gc()
        t0 = time_ns()
        result = redirect_stdout(devnull) do
            RunSimulation(; kw..., SimParticles = particles, SimLogger = logger)
        end
        wall = (time_ns() - t0) / 1e9
        order = sortperm(particles.ID)
        data = h5open(joinpath(dir, "MovingSquare2D.vtkhdf"), "r") do file
            root = file["VTKHDF"]
            (; points = read(root["Points"]), times = read(root["Steps/Values"]),
                fields = Dict(name => read(root["PointData/$name"]) for name in keys(root["PointData"])))
        end
        @printf("dx=%.3f layout=%-7s n=%d steps=%d frames=%d wall=%.4f s writer=%.4f s\n",
            dx, legacy ? "legacy" : "bounded", length(particles), result.performance.steps,
            result.diagnostics.output_frames, wall, result.performance.output_write_seconds)
        flush(stdout)
        return (; wall, diagnostics = result.diagnostics, final_time = meta.TotalTime,
            position = copy(particles.Position[order]), velocity = copy(particles.Velocity[order]),
            density = copy(particles.Density[order]), data)
    end
end

function chunk_benchmark_main(args)
    option(name, default) = begin
        index = findfirst(==(name), args)
        index === nothing ? default : parse(Float32, args[index + 1])
    end
    duration, interval = option("--time", 0.32f0), option("--interval", 0.01f0)
    duration > 0 && interval > 0 || throw(ArgumentError("time and interval must be positive"))
    println("GPU $(CUDA.name(CUDA.device())); Float32; threads $(Threads.nthreads(:default)),$(Threads.nthreads(:interactive)); interval $interval s")
    for dx in (0.04, 0.02)
        chunk_benchmark_run(false, dx, duration, interval; warmup = true)
        chunk_benchmark_run(true, dx, duration, interval; warmup = true)
        for legacy_first in (true, false, true)
            a = chunk_benchmark_run(legacy_first, dx, duration, interval)
            b = chunk_benchmark_run(!legacy_first, dx, duration, interval)
            @assert a.diagnostics == b.diagnostics && a.final_time == b.final_time
            @assert a.position == b.position && a.velocity == b.velocity && a.density == b.density
            @assert a.data == b.data
            @printf("paired whole-run speedup %.3fx; counters, states and saved fields identical\n",
                legacy_first ? a.wall / b.wall : b.wall / a.wall)
            flush(stdout)
            GC.gc()
            CUDA.reclaim()
        end
    end
end

chunk_benchmark_main(ARGS)
