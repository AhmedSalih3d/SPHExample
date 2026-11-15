module PauseDiagnostic
include(joinpath(@__DIR__,"dual_compare.jl"))
using .Test06DualComparison, SPHExampleGPU, StaticArrays, CSV, TOML
const C = Test06DualComparison

# Smoothly stop over 0.25 s, hold for 2 s, then smoothly resume the same path.
function paused_trajectory()
    original = C.body_trajectory()
    times = collect(0.:0.001:8.)
    positions,velocities = SVector{2,Float64}[],SVector{2,Float64}[]
    for t in times
        if t <= 1.75
            s,q = t,1.
        elseif t <= 2.
            z = t-1.75
            s,q = 1.75+z/2+0.25sin(π*z/0.25)/(2π),(1+cos(π*z/0.25))/2
        elseif t <= 4.
            s,q = 1.875,0.
        elseif t <= 4.25
            z = t-4.
            s,q = 1.875+z/2-0.25sin(π*z/0.25)/(2π),(1-cos(π*z/0.25))/2
        else
            s,q = t-2.25,1.
        end
        x,v,_ = trajectory_state(original,s)
        push!(positions,x); push!(velocities,q*v)
    end
    return SampledTrajectory(times,positions; velocities)
end

function run()
    results = mkpath(joinpath(@__DIR__,"results","pause_diagnostic"))
    if !isfile(joinpath(results,"settings.toml"))
        open(joinpath(results,"settings.toml"),"w") do io
            TOML.print(io,Dict("reynolds"=>100,"sound_speeds"=>[40,160],"cfl"=>0.1,"dp"=>0.025,
                "duration"=>8.,"output_interval"=>0.01,"deceleration"=>[1.75,2.],"hold"=>[2.,4.],
                "restart"=>[4.,4.25],"motion"=>"smooth time mapping of original Hermite trajectory",
                "boundary"=>"DBC","divergence_cleaning"=>false,"edge_damping"=>false); sorted=true)
        end
    end
    for speed in (40,160)
        id = "re100_c$(speed)_pause"
        target = mkpath(joinpath(results,id))
        isfile(joinpath(target,"native.toml")) && continue
        config = merge(C.config(id; sound_speed=speed),
            (; trajectory=paused_trajectory(),duration=8.,interval=0.01,cfl=0.1))
        println("Pause diagnostic: ",id); flush(stdout)
        info = C.native_run(config,joinpath(@__DIR__,"runs","pause_diagnostic",id); results=target)
        C.save_info(joinpath(target,"native.toml"),info)
    end
end
end
if abspath(PROGRAM_FILE) == @__FILE__
    PauseDiagnostic.run()
end
