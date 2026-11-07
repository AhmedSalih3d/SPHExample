using SPHExampleGPU
include("cases.jl")
using .IISPHCases

save = isempty(ARGS) ? joinpath(pwd(),"IISPHTank") : ARGS[1]
case = tank_case(Float32,save;duration=1.0,output_interval=0.05,
    solver=IISPH(max_timestep=0.002,max_iterations=500))
try
    result = RunSimulation(;case...)
    println(result.diagnostics.iisph)
finally
    isopen(case.SimLogger.LoggerIo) && close(case.SimLogger.LoggerIo)
end
