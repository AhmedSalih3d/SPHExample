# julia --project=gpu_version -t 2,0 gpu_version/benchmark/step_state.jl
module StepStateBenchmark

using SPHExampleGPU, CUDA, HDF5, SHA, Statistics, Printf, TimerOutputs

include(joinpath(@__DIR__, "memory_buffers.jl"))

const SOURCE = joinpath(@__DIR__, "..", "src", "SPHCellList.jl")

function baseline_driver()
    source = read(SOURCE, String)
    occursin("readback!(state)", source) || error("Cannot locate the solver readback call")
    source = replace(source,
        "module SPHCellList" => """module SynchronousReadbackDriver
        function synchronous_readback!(state)
            copyto!(state.fh, state.f)
            copyto!(state.ih, state.i)
            state.readbacks += 1
            return state
        end
        """,
        "readback!(state)" => "synchronous_readback!(state)")
    Base.include_string(SPHExampleGPU, source, SOURCE)
    return Base.invokelatest(getproperty, SPHExampleGPU, :SynchronousReadbackDriver)
end

const BASELINE_DRIVER = baseline_driver()

function output_fingerprint(save)
    datasets = Dict{String,Tuple{DataType,Tuple,String}}()
    function visit(group, prefix)
        for name in keys(group)
            item = group[name]
            key = prefix * "/" * name
            if item isa HDF5.Group
                visit(item, key)
            else
                data = read(item)
                datasets[key] = (eltype(data), size(data), bytes2hex(sha256(reinterpret(UInt8, vec(data)))))
            end
        end
    end
    for name in ("StillWedge.vtkhdf", "StillWedge_GridCells.vtkhdf")
        h5open(joinpath(save, name), "r") do file
            visit(file, name)
        end
    end
    return datasets
end

function measure(driver, ::Type{T}; duration, dx, single, graph) where {T}
    return mktempdir() do save
        kw = GPUMemoryBenchmark.generated_wedge_case(T, save; dx)
        meta = kw.SimMetaData
        meta.SimulationTime = T(duration)
        meta.OutputTimes = T(0.01)
        meta.ExportGridCells = true
        meta.GPUDoublePosition = true
        meta.GPUCellSubdivision = 1
        meta.GPURadialInteractions = false
        meta.GPUUseGraph = graph
        meta.GPUMaxStepsPerSync = 32
        meta.GPUSyncTimers = false
        particles = AllocateDataStructures(kw.SimGeometry, meta)
        logger = SimulationLogger(save)
        GC.gc()
        CUDA.synchronize()
        seconds = try
            redirect_stdout(devnull) do
                @elapsed begin
                    driver.RunSimulation(; kw..., SimParticles = particles, SimLogger = logger,
                        SimTimeStepping = single ? SingleNeighborTimeStepping() : SymplecticTimeStepping())
                    CUDA.synchronize()
                end
            end
        finally
            close(logger.LoggerIo)
        end
        order = sortperm(particles.ID)
        fields = map(name -> getproperty(particles, name)[order],
            (:Position, :Velocity, :Density, :Acceleration, :ID, :Type, :GroupMarker))
        all(isfinite, particles.Density) && all(>(0), particles.Density) ||
            error("Invalid accepted density")
        loop = meta.HourGlass["00 SimulationLoop"]
        work = (; steps = meta.Iteration, time = meta.TotalTime,
            rebuilds = TimerOutputs.ncalls(loop["02a Actual Calculate IndexCounter"]),
            readbacks = TimerOutputs.ncalls(loop["13 Read Back State"]))
        return (; seconds, loop_seconds = TimerOutputs.time(loop) / 1e9,
            readback_seconds = TimerOutputs.time(loop["13 Read Back State"]) / 1e9,
            work, fields, output = output_fingerprint(save), particles = length(particles))
    end
end

function equivalent(a, b)
    a.work == b.work || error("Step/rebuild/readback counts or accepted time changed")
    isequal(a.fields, b.fields) || error("Accepted particle state changed")
    isequal(a.output, b.output) || error("Exported particle/grid datasets changed")
    return nothing
end

function readback_seconds(readback, state; repeats = 405)
    CUDA.synchronize()
    seconds = @elapsed for _ in 1:repeats
        readback(state)
    end
    state.fh == Array(state.f) && state.ih == Array(state.i) ||
        error("Incomplete control-state readback")
    return seconds / repeats
end

function main(args)
    value(key, default) = begin
        matches = filter(a -> startswith(a, key * "="), args)
        isempty(matches) ? default : split(only(matches), '='; limit = 2)[2]
    end
    rounds = parse(Int, value("--rounds", "61"))
    duration = parse(Float64, value("--duration", "4"))
    dx = parse(Float64, value("--dx", "0.02"))
    rounds >= 3 || throw(ArgumentError("At least three alternating pairs are required"))
    duration > 0 && dx > 0 || throw(ArgumentError("Duration and spacing must be positive"))
    T = "--float64" in args ? Float64 : Float32
    single, graph = "--single-neighbor" in args, !("--no-graph" in args)
    output = value("--output", joinpath(@__DIR__, "step_state_results.csv"))
    drivers = (BASELINE_DRIVER, SPHExampleGPU.SPHCellList)
    readbacks = (BASELINE_DRIVER.synchronous_readback!, SPHExampleGPU.readback!)
    state = StepState{T}(dt = T(0.125), time = T(0.5), iteration = 73)
    foreach(readback -> readback_seconds(readback, state), readbacks)
    println("GPU=", CUDA.name(CUDA.device()), "; Julia=", VERSION,
        "; only readback differs; physics=", T, "; Float64 positions; R=1; single=", single,
        "; graph=", graph, "; duration=", duration, "; dx=", dx,
        "; payload=", sizeof(state.fh), "+", sizeof(state.ih), " bytes")
    for _ in 1:2
        warm = map(driver -> measure(driver, T; duration, dx, single, graph), drivers)
        equivalent(warm...)
        println("warm exact=true particles=", warm[1].particles, " work=", warm[1].work)
        flush(stdout)
    end
    total_times, copy_times, solver_readback_times = [Float64[] for _ in drivers],
        [Float64[] for _ in drivers], [Float64[] for _ in drivers]
    open(output, "w") do out
        println(out, "round,variant,total_s,loop_s,solver_readback_s,idle_readback_us,steps,rebuilds,readbacks,time,exact_state,exact_output")
        for round in 1:rounds
            indices = isodd(round) ? (1, 2) : (2, 1)
            copies = Dict(i => readback_seconds(readbacks[i], state) for i in indices)
            results = Dict(i => measure(drivers[i], T; duration, dx, single, graph) for i in indices)
            equivalent(results[1], results[2])
            for i in 1:2
                r = results[i]
                push!(total_times[i], r.seconds)
                push!(copy_times[i], copies[i])
                push!(solver_readback_times[i], r.readback_seconds)
                println(out, join((round, i == 1 ? "synchronous" : "queued", r.seconds,
                    r.loop_seconds, r.readback_seconds, 1e6 * copies[i], r.work.steps,
                    r.work.rebuilds, r.work.readbacks, r.work.time, true, true), ','))
            end
            flush(out)
            @printf("pair=%d synchronous=%.6fs queued=%.6fs ratio=%.4f exact=true\n",
                round, results[1].seconds, results[2].seconds, results[2].seconds / results[1].seconds)
            flush(stdout)
        end
    end
    ratios = total_times[2] ./ total_times[1]
    @printf("complete-run medians synchronous=%.6fs queued=%.6fs\n",
        median(total_times[1]), median(total_times[2]))
    @printf("idle readback medians synchronous=%.3fus queued=%.3fus; paired reduction=%.2f%%\n",
        1e6 * median(copy_times[1]), 1e6 * median(copy_times[2]),
        100 * (1 - median(copy_times[2] ./ copy_times[1])))
    @printf("complete-run paired ratio median=%.4f IQR=[%.4f, %.4f] range=[%.4f, %.4f]; reduction=%.2f%%; faster=%d/%d\n",
        median(ratios), quantile(ratios, 0.25), quantile(ratios, 0.75),
        minimum(ratios), maximum(ratios), 100 * (1 - median(ratios)), count(<(1), ratios), rounds)
    @printf("solver readback timer paired reduction=%.2f%% (includes pending device work)\n",
        100 * (1 - median(solver_readback_times[2] ./ solver_readback_times[1])))
    println("Results: ", output)
    return nothing
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    StepStateBenchmark.main(ARGS)
end
