module ReductionBenchmark

using SPHExampleGPU, CUDA, HDF5, Statistics, Printf, TimerOutputs

include(joinpath(@__DIR__, "memory_buffers.jl"))

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const BASELINE = "629e97255448a5d067673a61e2cf6c6681f196d6"

function baseline_driver()
    source(file) = read(joinpath(REPO, "gpu_version", "src", file), String)
    reductions = read(`git -C $REPO show $(BASELINE * ":gpu_version/src/GPUReductions.jl")`, String)
    reductions = replace(reductions, "module GPUReductions" => "module TreeReductions")
    kernels = replace(source("GPUKernels.jl"),
        "module GPUKernels" => "module TreeReductionKernels",
        "using ..GPUReductions" => "using ..TreeReductions",
        "include(\"UpdatedMDBC.jl\")" => source("UpdatedMDBC.jl"))
    driver = replace(source("SPHCellList.jl"),
        "module SPHCellList" => "module TreeReductionDriver",
        "using ..GPUReductions" => "using ..TreeReductions",
        "using ..GPUKernels" => "using ..TreeReductionKernels",
        "GPUKernels." => "TreeReductionKernels.")
    for code in (reductions, kernels, driver)
        Base.include_string(SPHExampleGPU, code, joinpath(REPO, "gpu_version", "src", "reduction_baseline.jl"))
    end
    return Base.invokelatest(getproperty, SPHExampleGPU, :TreeReductionDriver)
end

const BASELINE_DRIVER = baseline_driver()

function output_data(save)
    datasets = Dict{String,Any}()
    function visit(group, prefix)
        for name in keys(group)
            item = group[name]
            key = prefix * "/" * name
            if item isa HDF5.Group
                visit(item, key)
            else
                datasets[key] = read(item)
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
        meta.OutputTimes = T(0.1)
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
        return (; seconds, loop_seconds = TimerOutputs.time(loop) / 1e9, work,
            fields, output = output_data(save), particles = length(particles))
    end
end

function equivalent(a, b)
    a.work == b.work || error("Step/rebuild/readback counts or accepted time changed")
    isequal(a.fields, b.fields) || error("Accepted particle state changed")
    isequal(a.output, b.output) || error("Exported particle/grid datasets changed")
    return nothing
end

function main(args)
    value(key, default) = begin
        matches = filter(a -> startswith(a, key * "="), args)
        isempty(matches) ? default : split(only(matches), '='; limit = 2)[2]
    end
    rounds = parse(Int, value("--rounds", "9"))
    duration = parse(Float64, value("--duration", "4"))
    dx = parse(Float64, value("--dx", "0.02"))
    rounds >= 3 || throw(ArgumentError("At least three alternating pairs are required"))
    duration > 0 && dx > 0 || throw(ArgumentError("Duration and spacing must be positive"))
    T = "--float64" in args ? Float64 : Float32
    single, graph = "--single-neighbor" in args, !("--no-graph" in args)
    output = value("--output", joinpath(@__DIR__, "reductions_results.csv"))
    drivers = (BASELINE_DRIVER, SPHExampleGPU.SPHCellList)
    println("GPU=", CUDA.name(CUDA.device()), "; Julia=", VERSION, "; tree revision=", BASELINE,
        "; only GPUReductions differs; physics=", T, "; Float64 positions; R=1; single=", single,
        "; graph=", graph, "; duration=", duration, "; dx=", dx)
    # Two complete runs per variant warm both compiled kernels and graph paths.
    for _ in 1:2
        warm = map(driver -> measure(driver, T; duration, dx, single, graph), drivers)
        equivalent(warm...)
        println("warm exact=true particles=", warm[1].particles, " work=", warm[1].work)
        flush(stdout)
    end
    ratios = Float64[]
    open(output, "w") do out
        println(out, "round,variant,total_s,loop_s,steps,rebuilds,readbacks,time,exact_state,exact_output")
        for round in 1:rounds
            results = Dict(i => measure(drivers[i], T; duration, dx, single, graph)
                for i in (isodd(round) ? (1, 2) : (2, 1)))
            equivalent(results[1], results[2])
            ratio = results[2].seconds / results[1].seconds
            push!(ratios, ratio)
            for i in 1:2
                r = results[i]
                println(out, join((round, i == 1 ? "tree" : "warp", r.seconds,
                    r.loop_seconds, r.work.steps, r.work.rebuilds, r.work.readbacks,
                    r.work.time, true, true), ','))
            end
            flush(out)
            @printf("pair=%d tree=%.6fs warp=%.6fs ratio=%.4f exact=true\n",
                round, results[1].seconds, results[2].seconds, ratio)
            flush(stdout)
        end
    end
    @printf("paired complete-run ratio median=%.4f range=[%.4f, %.4f]; improvement=%.2f%%; faster=%d/%d\n",
        median(ratios), minimum(ratios), maximum(ratios), 100 * (1 - median(ratios)), count(<(1), ratios), rounds)
    println("Results: ", output)
    return nothing
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    ReductionBenchmark.main(ARGS)
end
