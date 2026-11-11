# julia --project=. -t 4,0 benchmark/predictor_fusion.jl --rounds=9 --duration=4
module PredictorFusionBenchmark

using SPHExampleGPU, CUDA, StaticArrays, HDF5, Statistics, Printf, TimerOutputs

include(joinpath(@__DIR__, "memory_buffers.jl"))

function frames(save)
    h5open(joinpath(save, "StillWedge.vtkhdf"), "r") do file
        root = file["VTKHDF"]
        times = read(root["Steps/Values"])
        offsets = read(root["Steps/PointOffsets"])
        counts = read(root["NumberOfPoints"])
        n = first(counts)
        all(==(n), counts) || error("Particle counts changed across frames")
        ids, types, groups = (read(root["PointData/" * name]) for name in ("ID", "Type", "GroupMarker"))
        raw_position, raw_velocity = read(root["Points"]), read(root["PointData/Velocity"])
        raw_density, raw_pressure = read(root["PointData/Density"]), read(root["PointData/Pressure"])
        position = Array{eltype(raw_position)}(undef, size(raw_position, 1), n, length(times))
        velocity = Array{eltype(raw_velocity)}(undef, size(raw_velocity, 1), n, length(times))
        density, pressure = (Matrix{eltype(raw_density)}(undef, n, length(times)) for _ in 1:2)
        identity = nothing
        for frame in eachindex(times)
            columns = offsets[frame] .+ (1:n)
            columns = columns[sortperm(ids[columns])]
            current = (; ids = ids[columns], types = types[columns], groups = groups[columns])
            identity === nothing ? (identity = current) : current == identity || error("Particle identities changed")
            position[:, :, frame] = raw_position[:, columns]
            velocity[:, :, frame] = raw_velocity[:, columns]
            density[:, frame] = raw_density[columns]
            pressure[:, frame] = raw_pressure[columns]
        end
        all(isfinite, position) && all(isfinite, velocity) && all(isfinite, pressure) &&
            all(x -> isfinite(x) && x > 0, density) || error("Invalid saved particle state")
        fixed = findall(==(Int(Fixed)), identity.types)
        for frame in eachindex(times)
            position[:, fixed, frame] == position[:, fixed, 1] || error("Fixed wall moved")
            all(iszero, velocity[:, fixed, frame]) || error("Fixed wall has nonzero velocity")
        end
        return (; times, position, velocity, density, pressure, identity)
    end
end

function differences(a, b)
    a.identity == b.identity || error("Particle identities differ between variants")
    size(a.position) == size(b.position) || error("Frame shapes differ")
    position_squared = sum(abs2, a.position .- b.position; dims = 1)
    velocity_squared = sum(abs2, Float64.(a.velocity) .- b.velocity; dims = 1)
    dp = Float64.(a.pressure) .- b.pressure
    return (; max_position_mm = 1000sqrt(maximum(position_squared)), rms_position_mm = 1000sqrt(mean(position_squared)),
        max_velocity_m_s = sqrt(maximum(velocity_squared)), max_density_kg_m3 = maximum(abs, Float64.(a.density) .- b.density),
        max_pressure_pa = maximum(abs, dp), rms_pressure_pa = sqrt(mean(abs2, dp)),
        max_time_difference_s = maximum(abs, a.times .- b.times))
end

function measure(fused; duration, dx, cache)
    mktempdir() do save
        result, meta, particles = nothing, nothing, nothing
        GC.gc()
        CUDA.synchronize()
        seconds = redirect_stdout(devnull) do
            @elapsed begin
                kw = GPUMemoryBenchmark.generated_wedge_case(Float32, save; dx)
                meta = kw.SimMetaData
                meta.SimulationTime = duration
                meta.OutputTimes = 0.01
                meta.ExportGridCells = true
                meta.ExportSingleVTKHDF = true
                meta.GPUDoublePosition = true
                meta.GPUCacheInteractionFields = cache
                meta.GPURadialInteractions = true
                meta.GPUFusePredictor = fused
                meta.GPUOutputQueueBytes = 8 * 1024^2
                particles = AllocateDataStructures(kw.SimGeometry, meta)
                logger = SimulationLogger(save)
                try
                    result = RunSimulation(; kw..., SimParticles = particles, SimLogger = logger,
                        SimTimeStepping = SymplecticTimeStepping())
                    CUDA.synchronize()
                finally
                    close(logger.LoggerIo)
                end
            end
        end
        result.configuration.effective.fuse_predictor == fused || error("Unexpected effective fusion setting")
        all(isfinite, particles.Density) && all(v -> all(isfinite, v), particles.Acceleration) || error("Invalid final state")
        loop = meta.HourGlass["00 SimulationLoop"]
        work = (; steps = meta.Iteration, rebuilds = TimerOutputs.ncalls(loop["02a Actual Calculate IndexCounter"]),
            readbacks = TimerOutputs.ncalls(loop["13 Read Back State"]))
        return (; seconds, loop_seconds = TimerOutputs.time(loop) / 1e9, work, time = meta.TotalTime,
            particles = length(particles), frames = frames(save))
    end
end

function main(args)
    value(key, default) = begin
        matches = filter(a -> startswith(a, key * "="), args)
        isempty(matches) ? default : split(only(matches), '='; limit = 2)[2]
    end
    rounds = parse(Int, value("--rounds", "9"))
    duration = parse(Float64, value("--duration", "4"))
    dx = parse(Float64, value("--dx", "0.02"))
    cache = parse(Bool, value("--cache", "true"))
    tolerance = parse(Float64, value("--position-mm", "5"))
    rounds >= 3 || error("Use at least three alternating rounds")
    isfinite(duration) && duration > 0 && isfinite(dx) && dx > 0 || error("Duration and dx must be positive and finite")
    isfinite(tolerance) && tolerance >= 0 || error("Tolerance must be finite and nonnegative")
    println("GPU=", CUDA.name(CUDA.device()), "; Julia=", VERSION, "; dx=", dx,
        "; duration=", duration, "; cached=", cache, "; radial=true; symplectic; all particle/grid output included")
    options = (; duration, dx, cache)
    warmups = [measure(fused; options...) for fused in (false, true)]
    println("Particles: ", warmups[1].particles, "; warmup work: ", map(r -> r.work, warmups))
    warmup_delta = differences(warmups[2].frames, warmups[1].frames)
    println("Warmup differences: ", warmup_delta)
    warmup_delta.max_position_mm <= tolerance || error("Warmup position difference exceeds $tolerance mm")
    "--validate-only" in args && return nothing
    warmups[1].work == warmups[2].work || error("Warmup work counts differ")
    times = (Float64[], Float64[])
    output = value("--output", joinpath(@__DIR__, "predictor_fusion_results.csv"))
    open(output, "w") do out
        println(out, "round,fused,particles,total_s,loop_s,steps,rebuilds,readbacks,time,max_position_mm,rms_position_mm,max_velocity_m_s,max_density_kg_m3,max_pressure_pa,rms_pressure_pa,max_time_difference_s")
        for round in 1:rounds
            results = Dict(fused => measure(fused; options...) for fused in (isodd(round) ? (false, true) : (true, false)))
            baseline = results[false]
            for fused in (false, true)
                r = results[fused]
                r.work == baseline.work || error("Work counts differ")
                delta = differences(r.frames, baseline.frames)
                delta.max_position_mm <= tolerance || error("Position difference exceeds $tolerance mm: $delta")
                println(out, join((round, fused, r.particles, r.seconds, r.loop_seconds, Tuple(r.work)..., r.time, Tuple(delta)...), ','))
                push!(times[Int(fused) + 1], r.seconds)
            end
            flush(out)
            @printf("round=%d baseline=%.6f fused=%.6f ratio=%.4f max_position_mm=%.6f\n",
                round, baseline.seconds, results[true].seconds, results[true].seconds / baseline.seconds,
                differences(results[true].frames, baseline.frames).max_position_mm)
            flush(stdout)
        end
    end
    @printf("median baseline=%.6f fused=%.6f paired_ratio=%.4f\n", median(times[1]), median(times[2]), median(times[2] ./ times[1]))
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    PredictorFusionBenchmark.main(ARGS)
end
