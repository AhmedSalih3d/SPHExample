# Warmed, alternating complete runs against the pre-fusion cached driver.
# julia --project=gpu_version -t 2,0 gpu_version/benchmark/final_state_caches.jl
module FinalStateCacheBenchmark

using SPHExampleGPU, CUDA, HDF5, SHA, Statistics, Printf, TimerOutputs

include(joinpath(@__DIR__, "memory_buffers.jl"))

const REPO = normpath(joinpath(@__DIR__, "..", ".."))
const BASELINE = "a2f57c4ea518711e9a34c07f5967bb317b56ee8e"

function baseline_driver()
    source(file) = replace(read(`git -C $REPO show $(BASELINE * ":gpu_version/src/" * file)`, String), "\r\n" => "\n")
    kernels = replace(source("GPUKernels.jl"),
        "module GPUKernels" => "module FinalCacheBaselineKernels",
        "include(\"UpdatedMDBC.jl\")" => source("UpdatedMDBC.jl"))
    driver = replace(source("SPHCellList.jl"),
        "module SPHCellList" => "module FinalCacheBaselineDriver",
        "using ..GPUKernels" => "using ..FinalCacheBaselineKernels",
        "GPUKernels." => "FinalCacheBaselineKernels.")
    Base.include_string(SPHExampleGPU, kernels, joinpath(REPO, "gpu_version", "src", "GPUKernels.jl"))
    Base.include_string(SPHExampleGPU, driver, joinpath(REPO, "gpu_version", "src", "SPHCellList.jl"))
    return Base.invokelatest(getproperty, SPHExampleGPU, :FinalCacheBaselineDriver)
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

function measure(driver; duration, dx, collect_garbage = true)
    return mktempdir() do save
        kw = GPUMemoryBenchmark.generated_wedge_case(Float32, save; dx)
        meta = kw.SimMetaData
        meta.SimulationTime = Float32(duration)
        meta.OutputTimes = 0.01f0
        meta.ExportGridCells = true
        meta.GPUDoublePosition = true
        meta.GPUCacheInteractionFields = true
        meta.GPUCellSubdivision = 1
        meta.GPURadialInteractions = false
        meta.GPUUseGraph = true
        meta.GPUMaxStepsPerSync = 32
        meta.GPUSyncTimers = false
        particles = AllocateDataStructures(kw.SimGeometry, meta)
        logger = SimulationLogger(save)
        collect_garbage && GC.gc()
        CUDA.synchronize()
        seconds = try
            redirect_stdout(devnull) do
                @elapsed begin
                    driver.RunSimulation(; kw..., SimParticles = particles, SimLogger = logger,
                        SimTimeStepping = SymplecticTimeStepping())
                    CUDA.synchronize()
                end
            end
        finally
            close(logger.LoggerIo)
        end
        order = sortperm(particles.ID)
        fields = map(name -> getproperty(particles, name)[order],
            (:Position, :Velocity, :Density, :Acceleration, :ID, :Type, :GroupMarker))
        all(isfinite, particles.Density) || error("Nonfinite accepted density")
        all(>(0), particles.Density) || error("Nonpositive accepted density")
        loop = meta.HourGlass["00 SimulationLoop"]
        work = (; steps = meta.Iteration, time = meta.TotalTime,
            rebuilds = TimerOutputs.ncalls(loop["02a Actual Calculate IndexCounter"]),
            readbacks = TimerOutputs.ncalls(loop["13 Read Back State"]))
        return (; seconds, loop_seconds = TimerOutputs.time(loop) / 1e9,
            work, fields, output = output_fingerprint(save), particles = length(particles))
    end
end

function equivalent(a, b)
    a.work == b.work || error("Step/rebuild/readback counts or accepted time changed")
    a.fields == b.fields || error("Accepted particle state changed")
    a.output == b.output || error("Exported particle/grid datasets changed")
    return nothing
end

function profile_preparation(driver; duration, dx)
    result = nothing
    GC.gc()
    profile = CUDA.@profile trace=true begin
        result = measure(driver; duration, dx, collect_garbage = false)
    end
    names = String.(profile.device.name)
    preparation = count(name -> occursin("prepare_interactions_kernel", name), names)
    finals = count(name -> occursin("final_step_kernel", name), names)
    return (; preparation, finals, work = result.work)
end

function main(args)
    value(key, default) = begin
        matches = filter(a -> startswith(a, key * "="), args)
        isempty(matches) ? default : split(only(matches), '='; limit = 2)[2]
    end
    rounds = parse(Int, value("--rounds", "21"))
    duration = parse(Float64, value("--duration", "4"))
    dx = parse(Float64, value("--dx", "0.02"))
    rounds >= 3 || throw(ArgumentError("At least three alternating rounds are required"))
    duration > 0 && dx > 0 || throw(ArgumentError("Duration and spacing must be positive"))
    output_path = value("--output", joinpath(@__DIR__, "final_state_caches_results.csv"))
    drivers = (BASELINE_DRIVER, SPHExampleGPU.SPHCellList)
    println("GPU=", CUDA.name(CUDA.device()), "; Julia=", VERSION, "; baseline=", BASELINE,
        "; Float32 physics / Float64 positions; R=1; cached; symplectic; duration=", duration, "; dx=", dx)
    warmups = map(driver -> measure(driver; duration, dx), drivers)
    equivalent(warmups...)
    println("particles=", warmups[1].particles, "; work=", warmups[1].work,
        "; exact accepted state and all exported datasets=true")
    flush(stdout)
    times = [Float64[] for _ in drivers]
    open(output_path, "w") do out
        println(out, "round,variant,total_s,loop_s,steps,rebuilds,readbacks,time,exact_state,exact_output")
        for round in 1:rounds
            results = Vector{typeof(first(warmups))}(undef, length(drivers))
            for i in (isodd(round) ? eachindex(drivers) : reverse(eachindex(drivers)))
                results[i] = measure(drivers[i]; duration, dx)
            end
            equivalent(results...)
            for i in eachindex(drivers)
                r = results[i]
                push!(times[i], r.seconds)
                println(out, join((round, i == 1 ? "baseline" : "fused", r.seconds,
                    r.loop_seconds, r.work.steps, r.work.rebuilds, r.work.readbacks,
                    r.work.time, true, true), ','))
            end
            flush(out)
            @printf("round=%d baseline=%.6fs fused=%.6fs ratio=%.4f exact=true\n",
                round, results[1].seconds, results[2].seconds, results[2].seconds / results[1].seconds)
            flush(stdout)
        end
    end
    ratios = times[2] ./ times[1]
    @printf("complete-run paired ratio median=%.4f range=[%.4f, %.4f]; median improvement=%.2f%%\n",
        median(ratios), minimum(ratios), maximum(ratios), 100 * (1 - median(ratios)))
    println("fused faster in ", count(<(1), ratios), "/", rounds, " pairs")
    if "--profile" in args
        profiles = map(driver -> profile_preparation(driver; duration, dx), drivers)
        println("separate profiled runs: baseline=", profiles[1], "; fused=", profiles[2])
        profiles[1].work == profiles[2].work || error("Profile work counts changed")
        profiles[2].finals == profiles[1].finals > 0 || error("Profiled final-integration entries changed")
        profiles[2].preparation == profiles[2].work.rebuilds ||
            error("Stationary cache preparation should run only after each reorder")
        profiles[2].preparation < profiles[1].preparation || error("Preparation entries did not decrease")
    end
    println("Results: ", output_path)
    return nothing
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    FinalStateCacheBenchmark.main(ARGS)
end
