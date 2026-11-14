include(joinpath(@__DIR__,"dual_compare.jl"))
using .Test06DualComparison
using SPHExampleGPU, CSV, TOML, Statistics, Printf

function motion_diagnostics()
    C = Test06DualComparison
    data = C.read_reference(joinpath(C.REFERENCE_DIR,"Motion_Body.dat"))
    original, integrated = C.body_trajectory(), C.integrated_trajectory()
    rows = NamedTuple[]
    for time in 0:0.001:8
        x,v,a = trajectory_state(original,time)
        sx,sv,sa = trajectory_state(integrated,time)
        index = min(searchsortedlast(data[:,1],time),size(data,1)-1)
        f = (time-data[index,1])/(data[index+1,1]-data[index,1])
        acceleration = (1-f)*data[index,2]+f*data[index+1,2]
        speed = (1-f)*data[index,3]+f*data[index+1,3]
        push!(rows,(; time,x = x[1]+1.5,velocity = v[1],acceleration = a[1],
            integrated_x = sx[1]+1.5,integrated_velocity = sv[1],integrated_acceleration = sa[1],
            reference_velocity = speed,reference_acceleration = acceleration))
    end
    mkpath(C.RESULTS)
    CSV.write(joinpath(C.RESULTS,"motion.csv"),rows)
    late = filter(row -> 2 <= row.time <= 7,rows)
    info = Dict("late_hermite_acceleration_rms"=>sqrt(mean(row -> row.acceleration^2,late)),
        "late_hermite_acceleration_max"=>maximum(row -> abs(row.acceleration),late),
        "late_integrated_acceleration_rms"=>sqrt(mean(row -> row.integrated_acceleration^2,late)),
        "late_reference_acceleration_max"=>maximum(row -> abs(row.reference_acceleration),late),
        "late_hermite_velocity_error_rms"=>sqrt(mean(row -> (row.velocity-row.reference_velocity)^2,late)),
        "integrated_position_difference_max"=>maximum(row -> abs(row.x-row.integrated_x),rows))
    open(joinpath(C.RESULTS,"motion.toml"),"w") do io
        TOML.print(io,info; sorted = true)
    end
    println(info)
    for (speed,directory) in ((40,"validation"),(80,"sound_speed_80"),(160,"sound_speed_160"))
        csv = CSV.File(joinpath(@__DIR__,"results",directory,"Re100_n060.csv"))
        ref = C.read_reference(joinpath(C.REFERENCE_DIR,"Force_Re100.dat"))
        metrics,_ = C.drag_metrics(Float64.(csv.time),Float64.(csv.cd_pressure),Float64.(csv.cd_viscous),ref)
        @printf("c=%d: pressure dominant %.3f Hz, box fundamental %.3f Hz, highpass RMS %.5f\n",
            speed,metrics["pressure_dominant_hz"],speed/20,metrics["pressure_highpass_rms"])
    end
end

motion_diagnostics()
