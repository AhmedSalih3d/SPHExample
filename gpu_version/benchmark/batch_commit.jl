# Paired complete solver runs with exact state and particle/grid output checks.
# julia --project=. -t 4,0 benchmark/batch_commit.jl --rounds=15 --duration=20
module BatchCommitBenchmark

using SPHExampleGPU, CUDA, HDF5, SHA, Statistics, Printf, TimerOutputs

include(joinpath(@__DIR__, "memory_buffers.jl"))

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const SOURCE = joinpath(@__DIR__, "..", "src")

function baseline_driver(revision)
    names = ("GPUKernels", "SPHCellList")
    function source(file)
        code = read(`git -C $REPO show $(revision * ":gpu_version/src/" * file)`, String)
        return replace(code, r"include\(\"([^\"]+)\"\)" =>
            matchtext -> source(match(r"\"([^\"]+)\"", matchtext)[1]))
    end
    for name in names
        code = replace(source(name * ".jl"), "module $name" => "module BatchBaseline$name")
        for original in names
            code = replace(code, "using ..$original" => "using ..BatchBaseline$original",
                "$original." => "BatchBaseline$original.")
        end
        Base.include_string(SPHExampleGPU, code, joinpath(SOURCE, name * ".jl"))
    end
    return Base.invokelatest(getproperty, SPHExampleGPU, :BatchBaselineSPHCellList)
end

function output_digests(save)
    datasets = Dict{String,Any}()
    function visit(group, prefix)
        for name in keys(group)
            item = group[name]
            path = prefix * "/" * name
            if item isa HDF5.Group
                visit(item, path)
            else
                data = read(item)
                datasets[path] = (eltype(data), size(data), bytes2hex(sha256(reinterpret(UInt8, vec(data)))))
            end
        end
    end
    for name in sort(filter(name -> endswith(name, ".vtkhdf"), readdir(save)))
        h5open(file -> visit(file, name), joinpath(save, name), "r")
    end
    isempty(datasets) && error("No VTKHDF output was written")
    return datasets
end

function measure(driver, ::Type{T}; duration, dx, single, graph, diagnostics, clip, radial) where {T}
    return mktempdir() do save
        kw = GPUMemoryBenchmark.generated_wedge_case(T, save; dx)
        meta = kw.SimMetaData
        meta.SimulationTime = duration
        meta.OutputTimes = 0.01
        meta.ExportGridCells = true
        meta.ExportSingleVTKHDF = true
        meta.GPUDoublePosition = true
        meta.GPUCellSubdivision = 1
        meta.GPURadialInteractions = radial
        meta.GPUUseGraph = graph
        meta.GPUMaxStepsPerSync = 32
        meta.GPUTimestepDiagnosticsEvery = diagnostics ? 1 : 0
        meta.GPUClipTimeStepToOutput = clip
        meta.GPUOutputQueueBytes = 8 * 1024^2
        particles = AllocateDataStructures(kw.SimGeometry, meta)
        logger = SimulationLogger(save)
        GC.gc()
        CUDA.synchronize()
        result = nothing
        seconds = try
            redirect_stdout(devnull) do
                @elapsed begin
                    result = driver.RunSimulation(; kw..., SimParticles = particles, SimLogger = logger,
                        SimTimeStepping = single ? SingleNeighborTimeStepping() : SymplecticTimeStepping())
                    CUDA.synchronize()
                end
            end
        finally
            close(logger.LoggerIo)
        end
        execution = result.configuration.execution
        execution.graph_capture_failures == 0 || error("CUDA graph capture failed")
        graph && execution.graph_batches == 0 && error("No CUDA graph batches executed")
        order = sortperm(particles.ID)
        fields = map(name -> getproperty(particles, name)[order],
            (:Position, :Velocity, :Density, :Acceleration, :ID, :Type, :GroupMarker))
        all(field -> all(value -> value isa Number ? isfinite(value) : all(isfinite, value), field), fields[1:4]) ||
            error("Nonfinite accepted state")
        all(>(0), particles.Density) || error("Nonpositive accepted density")
        loop = meta.HourGlass["00 SimulationLoop"]
        work = (; steps = meta.Iteration, time = meta.TotalTime,
            rebuilds = TimerOutputs.ncalls(loop["02a Actual Calculate IndexCounter"]),
            readbacks = TimerOutputs.ncalls(loop["13 Read Back State"]))
        diagnostic_path = joinpath(save, meta.SimulationName * "_Timestep.csv")
        diagnostic_output = diagnostics ? read(diagnostic_path, String) : nothing
        return (; seconds, loop_seconds = TimerOutputs.time(loop) / 1e9, work, execution,
            fields, output = output_digests(save), diagnostic_output, particles = length(particles))
    end
end

function equivalent(a, b)
    a.work == b.work || error("Step/rebuild/readback counts or accepted time changed")
    isequal(a.fields, b.fields) || error("Accepted particle state changed")
    isequal(a.output, b.output) || error("Exported particle/grid datasets changed")
    isequal(a.diagnostic_output, b.diagnostic_output) || error("Timestep diagnostic output changed")
    return nothing
end

function run_comparison(drivers, options; rounds, output)
    # Warm complete runs, including all output and graph specializations.
    for _ in 1:2
        warm = map(driver -> measure(driver, options.T; options.settings...), drivers)
        equivalent(warm...)
        println("warm exact=true particles=", warm[1].particles, " work=", warm[1].work)
        flush(stdout)
    end
    ratios = Float64[]
    open(output, "w") do out
        println(out, "round,variant,particles,total_s,loop_s,steps,rebuilds,readbacks,time,graph_batches,direct_batches,capture_failures,exact_state,exact_output")
        for round in 1:rounds
            results = Dict(i => measure(drivers[i], options.T; options.settings...)
                for i in (isodd(round) ? (1, 2) : (2, 1)))
            equivalent(results[1], results[2])
            ratio = results[2].seconds / results[1].seconds
            push!(ratios, ratio)
            for i in 1:2
                r = results[i]
                println(out, join((round, i == 1 ? "baseline" : "batch_commit", r.particles,
                    r.seconds, r.loop_seconds, r.work.steps, r.work.rebuilds, r.work.readbacks,
                    r.work.time, r.execution.graph_batches, r.execution.direct_batches,
                    r.execution.graph_capture_failures, true, true), ','))
            end
            flush(out)
            @printf("pair=%d baseline=%.6fs batch_commit=%.6fs ratio=%.4f exact=true\n",
                round, results[1].seconds, results[2].seconds, ratio)
            flush(stdout)
        end
    end
    @printf("paired ratio median=%.5f range=[%.5f, %.5f]; elapsed reduction=%.2f%%; faster=%d/%d\n",
        median(ratios), minimum(ratios), maximum(ratios), 100 * (1 - median(ratios)), count(<(1), ratios), rounds)
    return nothing
end

function main(args)
    value(key, default) = begin
        matches = filter(a -> startswith(a, key * "="), args)
        isempty(matches) ? default : split(only(matches), '='; limit = 2)[2]
    end
    rounds = parse(Int, value("--rounds", "15"))
    duration = parse(Float64, value("--duration", "20"))
    dx = parse(Float64, value("--dx", "0.02"))
    rounds >= 3 || throw(ArgumentError("At least three alternating pairs are required"))
    all(x -> isfinite(x) && x > 0, (duration, dx)) || throw(ArgumentError("Duration and spacing must be finite and positive"))
    T = "--float64" in args ? Float64 : Float32
    settings = (; duration, dx, single = "--single-neighbor" in args, graph = !("--no-graph" in args),
        diagnostics = "--diagnostics" in args, clip = "--clip-output" in args,
        radial = !("--reference-interactions" in args))
    revision = value("--baseline", "c90337d")
    output = value("--output", joinpath(@__DIR__, "batch_commit_results.csv"))
    println("GPU=", CUDA.name(CUDA.device()), "; Julia=", VERSION, "; CUDA.jl=", pkgversion(CUDA),
        "; baseline=", revision, "; physics=", T, "; Float64 positions; R=1; ", settings)
    drivers = (baseline_driver(revision), SPHExampleGPU.SPHCellList)
    Base.invokelatest(run_comparison, drivers, (; T, settings); rounds, output)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    BatchCommitBenchmark.main(ARGS)
end
