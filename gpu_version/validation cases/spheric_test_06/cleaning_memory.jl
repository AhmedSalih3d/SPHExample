using SPHExampleGPU, CUDA, TOML
const DC = SPHExampleGPU.GPUDivergenceCleaning
const LOGGER = SPHExampleGPU.SimulationLoggerConfiguration

function cleaning_memory()
    n = 84864
    result = Dict{String,Any}("particles"=>n,"disabled_device_bytes"=>LOGGER.device_buffer_bytes(nothing),
        "accounting"=>"live unique CuArray storage; excludes allocator reservations and program state")
    for T in (Float32,Float64)
        workspace = DC.CleaningWorkspace(DivergenceCleaning(),n,T)
        result[string(T)] = Dict("live_device_buffer_bytes"=>LOGGER.device_buffer_bytes(workspace),
            "psi_device_bytes"=>sizeof(workspace.psi),"predictor_device_bytes"=>sizeof(workspace.half),
            "derivative_device_bytes"=>sizeof(workspace.derivative),"signal_device_bytes"=>sizeof(workspace.signal.partial),
            "signal_host_bytes"=>sizeof(workspace.signal.host),"final_psi_host_bytes"=>n*sizeof(T))
    end
    directory = mkpath(joinpath(@__DIR__,"results","divergence_cleaning"))
    open(joinpath(directory,"memory.toml"),"w") do io
        TOML.print(io,result; sorted=true)
    end
    println(result)
end

cleaning_memory()
