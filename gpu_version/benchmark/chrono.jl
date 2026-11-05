# Warmed whole-solver measurements for the direct reference examples.
# julia --project=. -t 2,0 benchmark/chrono.jl FlexibleGate2d.jl [output]
include("chrono_case.jl")
include("chrono_validation.jl")
using Serialization,TOML,CSV,LinearAlgebra

function measure_chrono_case(filename,directory;options...)
    mkpath(directory)
    result,case = open(joinpath(directory,"console.log"),"w") do io
        redirect_stdout(io) do
            run_chrono_case(filename;save_location = directory,visualize = false,open_log_file = false,options...)
        end
    end
    metrics = Dict{String,Any}(string(k) => getproperty(result.performance,k) for k in
        (:steps,:simulated_time,:wall_time_seconds,:milliseconds_per_step,:output_write_seconds))
    for key in (:particle_count,:output_frames,:cell_list_rebuilds,:step_state_readbacks,:captured_graphs,:frozen_particles)
        metrics[string(key)] = getproperty(result.diagnostics,key)
    end
    merge!(metrics,Dict(string(k) => v for (k,v) in pairs(chrono_joint_validation(case,directory))))
    metrics["physics_precision"] = string(typeof(case.constants.g))
    metrics["position_precision"] = string(eltype(eltype(result.final_state.particles.Position)))
    if hasproperty(result.final_state,:open_boundary)
        merge!(metrics,Dict("open_"*string(k) => v for (k,v) in pairs(result.final_state.open_boundary)))
    end
    open(joinpath(directory,"metrics.toml"),"w") do io
        TOML.print(io,metrics)
    end
    serialize(joinpath(directory,"result.jls"),result)
    println(basename(directory)," ",metrics);flush(stdout)
    return result,case
end

if abspath(PROGRAM_FILE) == @__FILE__
    filename = isempty(ARGS) ? "FlexibleGate2d.jl" : first(ARGS)
    directory = abspath(length(ARGS) > 1 ? ARGS[2] : joinpath(@__DIR__,"..","output",splitext(filename)[1]*"_benchmark"))
    measure_chrono_case(filename,joinpath(directory,"warm"))
    result,case = measure_chrono_case(filename,joinpath(directory,"native"))
    Base.invokelatest(case.save_surfaces,case.meta.SaveLocation,case.geometry)
end
