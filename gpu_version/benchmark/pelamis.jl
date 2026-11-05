# julia --project=. -t 2,0 benchmark/pelamis.jl [output]
# Optional reference: DUALSPHYSICS_ROOT and DUALSPHYSICS_PELAMIS_CASE specify
# the installation and a GenCase-generated CasePelamis prefix (no extension).
include("pelamis_case.jl")
include("pelamis_validation.jl")
using Serialization, TOML

function measure_pelamis(directory; reference_directory = nothing, settings...)
    mkpath(directory)
    result = open(joinpath(directory,"console.log"),"w") do io
        redirect_stdout(io) do
            run_pelamis_2d(; save_location = directory, visualize = false, open_log_file = false, settings...)
        end
    end
    case = prepare_pelamis_2d(; settings...)
    validation = pelamis_validation(case,directory;reference_directory)
    metrics = Dict{String,Any}(string(k) => v for (k,v) in pairs(validation))
    for key in (:steps,:simulated_time,:wall_time_seconds,:milliseconds_per_step,:output_write_seconds)
        metrics[string(key)] = getproperty(result.performance,key)
    end
    for key in (:particle_count,:output_frames,:cell_list_rebuilds,:step_state_readbacks,:captured_graphs,:frozen_particles)
        metrics[string(key)] = getproperty(result.diagnostics,key)
    end
    metrics["physics_precision"] = string(typeof(case.constants.g))
    metrics["scheme"] = string(nameof(typeof(case.scheme)))
    metrics["iterations"] = case.dynamics.iterations
    metrics["time_control"] = "SPH acoustic/force controller"
    serialize(joinpath(directory,"result.jls"),result)
    open(joinpath(directory,"metrics.toml"),"w") do io
        TOML.print(io,metrics)
    end
    println(basename(directory)," ",metrics)
    flush(stdout)
    return result,case
end

function measure_dual_pelamis(directory, installation, prefix)
    mkpath(directory)
    executable = joinpath(installation,"bin","windows","DualSPHysics5.4_win64.exe")
    command = Cmd(`$executable -gpu $prefix $directory -svres`; dir = dirname(prefix))
    open(joinpath(directory,"console.log"),"w") do io
        run(pipeline(command; stdout = io, stderr = io))
    end
    row = only(CSV.File(joinpath(directory,"Run.csv"); delim = ';'))
    metrics = Dict("steps" => row.Steps, "simulated_time" => row.PhysicalTime,
        "solver_seconds" => row.TSimul, "total_seconds" => row.TTotal, "frames" => row.PartFiles)
    extractor = joinpath(installation,"bin","windows","FloatingInfo_win64.exe")
    run(pipeline(`$extractor -dirdata $(joinpath(directory,"data")) -savedata $(joinpath(directory,"floating")) -csvsep:1`;
        stdout = devnull, stderr = devnull))
    open(joinpath(directory,"metrics.toml"),"w") do io
        TOML.print(io,metrics)
    end
    println(basename(directory)," ",metrics)
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    directory = abspath(isempty(ARGS) ? joinpath(@__DIR__,"..","output","pelamis_benchmark") : first(ARGS))
    installation = get(ENV,"DUALSPHYSICS_ROOT","")
    prefix = get(ENV,"DUALSPHYSICS_PELAMIS_CASE","")
    # Warm the whole ten-second path, including moving-particle rebuilds and
    # graph variants. These are unprofiled, whole-solver timings with output.
    measure_pelamis(joinpath(directory,"warm"))
    measure_pelamis(joinpath(directory,"native_1"))
    if !isempty(prefix)
        measure_dual_pelamis(joinpath(directory,"dual_1"),installation,prefix)
        measure_dual_pelamis(joinpath(directory,"dual_2"),installation,prefix)
    end
    _,case = measure_pelamis(joinpath(directory,"native_2"))
    Base.invokelatest(case.save_surfaces,joinpath(directory,"native_2"),case.geometry;piston_state = case.piston_state)
end
