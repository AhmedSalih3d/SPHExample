using SPHExampleGPU
using CUDA
using HDF5
using SHA
using Statistics
using Printf
using TimerOutputs

include(joinpath(@__DIR__, "..", "example", "StillWedgeMDBC.jl"))

const ORIGINAL_WEDGE_REVISION = "2167212"
const WEDGE_SOURCE_DIR = normpath(joinpath(@__DIR__, "..", "src"))

function original_source(file)
    root = normpath(joinpath(@__DIR__, "..", ".."))
    return replace(read(`git -C $root show $(ORIGINAL_WEDGE_REVISION * ":gpu_version/" * file)`, String),
        "\r\n" => "\n")
end

function load_original_wedge()
    auxiliary_name = Symbol(replace(string(gensym(:OriginalWedgeAuxiliary)), "#" => "_"))
    kernels_name = Symbol(replace(string(gensym(:OriginalWedgeKernels)), "#" => "_"))
    driver_name = Symbol(replace(string(gensym(:OriginalWedgeDriver)), "#" => "_"))
    auxiliary = replace(original_source("src/AuxiliaryFunctions.jl"),
        "module AuxiliaryFunctions" => "module $auxiliary_name")
    kernels = replace(original_source("src/GPUKernels.jl"),
        "module GPUKernels" => "module $kernels_name",
        "include(\"UpdatedMDBC.jl\")" => original_source("src/UpdatedMDBC.jl"))
    driver = replace(original_source("src/SPHCellList.jl"),
        "module SPHCellList" => "module $driver_name",
        "using ..GPUKernels" => "using ..$kernels_name",
        "GPUKernels." => "$kernels_name.",
        "using ..AuxiliaryFunctions" => "using ..$auxiliary_name")
    Base.include_string(SPHExampleGPU, auxiliary, joinpath(WEDGE_SOURCE_DIR, "AuxiliaryFunctions.jl"))
    Base.include_string(SPHExampleGPU, kernels, joinpath(WEDGE_SOURCE_DIR, "GPUKernels.jl"))
    Base.include_string(SPHExampleGPU, driver, joinpath(WEDGE_SOURCE_DIR, "SPHCellList.jl"))
    original_driver = Base.invokelatest(getproperty, SPHExampleGPU, driver_name)
    original_auxiliary = Base.invokelatest(getproperty, SPHExampleGPU, auxiliary_name)

    scope = Module(gensym(:OriginalStillWedge))
    Core.eval(scope, :(include(path) = Base.include($scope, path)))
    Core.eval(scope, quote
        RunSimulation(; kwargs...) = $original_driver.RunSimulation(; kwargs...)
        CleanUpSimulationFolder(path) = $original_auxiliary.CleanUpSimulationFolder(path)
    end)
    source = original_source("example/StillWedgeMDBC.jl")
    occursin("\nlet\n", source) || error("original example no longer matches the recorded benchmark baseline")
    source = replace(source, "\nlet\n" => "\nfunction original_wedge(save_location)\n"; count = 1)
    source = replace(source,
        r"SaveLocation = \x22[^\x22]*\x22" => "SaveLocation = save_location",
        "VisualizeInParaview = true" => "VisualizeInParaview = false",
        "OpenLogFile = true" => "OpenLogFile = false")
    ending = findlast("\nend", source)
    ending === nothing && error("original example has no closing scope")
    source = source[1:prevind(source, first(ending))] *
        "\n    return (; particles = SimParticles, metadata = SimMetaDataWedge)" *
        source[first(ending):end]
    Base.include_string(scope, source, joinpath(@__DIR__, "..", "example", "StillWedgeMDBC.jl"))
    return Base.invokelatest(getproperty, scope, :original_wedge)
end

function wedge_output_fingerprint(save)
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

function measure_same_work(run)
    return mktempdir() do save
        result = nothing
        seconds = redirect_stdout(devnull) do
            @elapsed begin
                result = Base.invokelatest(run, save)
                CUDA.synchronize()
            end
        end
        particles, meta = result.particles, result.metadata
        meta.TimeSteppingMode isa SymplecticTimeStepping || error("benchmark must use symplectic integration")
        length(particles) == 3027 || error("benchmark particle count changed")
        meta.Iteration == 8500 || error("benchmark step count changed")
        meta.OutputTimes == 0.01f0 || error("benchmark output schedule changed")
        eltype(eltype(particles.Position)) === Float64 || error("position precision changed")
        eltype(particles.Density) === Float32 || error("physics precision changed")
        order = sortperm(particles.ID)
        state = map(name -> getproperty(particles, name)[order],
            (:Position, :Velocity, :Density, :Acceleration, :ID, :Type, :GroupMarker))
        output = wedge_output_fingerprint(save)
        output["StillWedge.vtkhdf/VTKHDF/Steps/Values"][2] == (401,) || error("particle frame count changed")
        output["StillWedge_GridCells.vtkhdf/VTKHDF/Steps/Values"][2] == (400,) || error("grid frame count changed")
        return (; seconds, loop = TimerOutputs.time(meta.HourGlass["00 SimulationLoop"]) / 1e9,
            time = meta.TotalTime, steps = meta.Iteration, state, output)
    end
end

"""
Compare the original and current four-second symplectic example, including input
preparation, cleanup, upload, stepping, logging, particle/grid output and closing.
Package loading, one warmup per variant, output verification and viewers are excluded.
An identical-original control exposes timing noise. All numerical state and exported
datasets must match exactly. `require_half_runtime` enforces candidate/original <= 0.5.
"""
function compare_same_work(; rounds = 5, require_half_runtime = false,
                           output_path = joinpath(@__DIR__, "still_wedge_same_work_results.csv"))
    rounds >= 3 || throw(ArgumentError("at least three alternating rounds are required"))
    original = load_original_wedge()
    current(save) = run_still_wedge(; time_stepping = SymplecticTimeStepping(),
        save_location = save, visualize = false, open_log_file = false)
    names = ("original", "original_control", "current")
    runners = (original, original, current)
    println("Baseline ", ORIGINAL_WEDGE_REVISION, "; GPU ", CUDA.name(CUDA.device()),
        "; Julia ", VERSION, "; symplectic; 3027 particles; four seconds")
    warmups = map(measure_same_work, runners)
    all(result -> result.state == warmups[1].state && result.output == warmups[1].output &&
        result.time == warmups[1].time, warmups) || error("warmup states or outputs differ")
    times = [Float64[] for _ in runners]
    open(output_path, "w") do out
        println(out, "revision,round,variant,total_s,loop_s,steps,final_time,particle_frames,grid_frames,exact_state,exact_output")
        for round in 1:rounds
            results = Vector{typeof(warmups[1])}(undef, length(runners))
            for index in (isodd(round) ? eachindex(runners) : reverse(eachindex(runners)))
                results[index] = measure_same_work(runners[index])
            end
            reference = results[1]
            for index in eachindex(runners)
                result = results[index]
                exact_state = result.state == reference.state && result.time == reference.time
                exact_output = result.output == reference.output
                println(out, join((ORIGINAL_WEDGE_REVISION, round, names[index], result.seconds,
                    result.loop, result.steps, result.time, 401, 400, exact_state, exact_output), ','))
                flush(out)
                exact_state && exact_output || error("state or exported datasets changed: $(names[index])")
                push!(times[index], result.seconds)
            end
            @printf("round=%d original=%.6f control=%.6f current=%.6f ratio=%.4f exact=true\n",
                round, results[1].seconds, results[2].seconds, results[3].seconds,
                results[3].seconds / results[1].seconds)
            flush(stdout)
        end
    end
    ratio = median(times[3] ./ times[1])
    @printf("Median total %.6f -> %.6f s; paired ratio %.4f (%.2f%% less time); control ratio %.4f\n",
        median(times[1]), median(times[3]), ratio, 100 * (1 - ratio), median(times[2] ./ times[1]))
    println("50% less runtime target: ", ratio <= 0.5 ? "PASS" : "NOT MET")
    require_half_runtime && ratio > 0.5 &&
        error("required current/original <= 0.5, measured $ratio; results saved to $output_path")
    return (; ratio, original = median(times[1]), current = median(times[3]))
end

if abspath(PROGRAM_FILE) == @__FILE__
    all(==("--assert-half-runtime"), ARGS) || throw(ArgumentError("expected [--assert-half-runtime]"))
    compare_same_work(; require_half_runtime = "--assert-half-runtime" in ARGS)
end
