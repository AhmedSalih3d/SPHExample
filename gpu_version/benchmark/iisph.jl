# Numerical/performance screen for the initial IISPH implementation. Whole-run
# times include startup and output; this script makes no WCSPH speedup claim.
using SPHExampleGPU, Statistics, LinearAlgebra
include(joinpath(@__DIR__,"..","example","IISPH","cases.jl"))
using .IISPHCases

save = isempty(ARGS) ? mktempdir() : ARGS[1]
duration = length(ARGS) >= 2 ? parse(Float64,ARGS[2]) : 0.2
for (name,nx,ny,width,height) in (("Tank",24,16,24,16),("DamBreak",48,24,16,20))
    case = tank_case(Float32,joinpath(save,name);name,dx=0.025,nx,ny,
        fluid_width=width,fluid_height=height,duration,
        solver=IISPH(max_timestep=0.002,max_iterations=1000))
    try
        result = RunSimulation(;case...)
        p = result.final_state.particles
        fluid = p.Type .== Fluid
        lower = fluid .& (getindex.(p.Position,2) .< 0.2) .&
            (getindex.(p.Position,1) .> 0.1) .& (getindex.(p.Position,1) .< 0.5)
        println((;name,result.performance,result.diagnostics.iisph,
            density_extrema=extrema(p.Density[fluid]),max_speed=maximum(norm,p.Velocity[fluid]),
            mean_lower_pressure=any(lower) ? mean(p.Pressure[lower]) : NaN,
            std_lower_pressure=any(lower) ? std(p.Pressure[lower]) : NaN,
            max_pressure=maximum(p.Pressure),
            x_extent=extrema(getindex.(p.Position[fluid],1))))
    finally
        isopen(case.SimLogger.LoggerIo) && close(case.SimLogger.LoggerIo)
    end
end
println("Output: ",save)
