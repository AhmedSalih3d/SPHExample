using SPHExampleGPU,CSV,StaticArrays,LinearAlgebra

function pelamis_validation(case,directory;reference_directory = nothing)
    rows = collect(CSV.File(joinpath(directory,"Pelamis2D_Floating.csv")))
    times = sort!(unique(r.Time for r in rows))
    initial = [sum(SVector{2,Float64}.(g.Particles.Position))/length(g.Particles) for g in case.geometry if g.Type == Floating]
    max_joint,max_heave,max_pitch,max_vertical_reaction = 0.,0.,0.,0.
    positions = Vector{SVector{2,Float64}}[]
    angles = Vector{Float64}[]
    for time in times
        frame = sort(filter(r -> r.Time == time,rows);by = r -> r.Body)
        state = (;center = [SVector(r[Symbol("Center:0")],r[Symbol("Center:1")]) for r in frame],angle = [Float64(r.Angle) for r in frame])
        max_joint = max(max_joint,maximum(e.translation_error for e in rigid_joint_errors(case.dynamics,case.geometry,state)))
        max_heave = max(max_heave,maximum(abs(state.center[b][2]-initial[b][2]) for b in 1:4))
        max_pitch = max(max_pitch,maximum(abs,state.angle))
        max_vertical_reaction = max(max_vertical_reaction,abs(sum(r[Symbol("ConstraintForce:1")] for r in frame))/sum(abs(r[Symbol("GravityForce:1")]) for r in frame))
        push!(positions,state.center)
        push!(angles,state.angle)
    end
    metrics = Dict{String,Any}("frames" => length(times),"max_joint_error_m" => max_joint,
        "max_heave_m" => max_heave,"max_pitch_deg" => rad2deg(max_pitch),
        "max_vertical_joint_force_fraction" => max_vertical_reaction)
    if reference_directory !== nothing
        reference = [collect(CSV.File(joinpath(reference_directory,"floating_mk$(mk).csv"))) for mk in 10:13]
        difference,pitch_difference = Float64[],Float64[]
        for b in 1:4
            ts = [Float64(r[Symbol("time [s]")]) for r in reference[b]]
            for (i,time) in enumerate(times)
                j = clamp(searchsortedlast(ts,time),1,length(ts)-1)
                fraction = clamp((time-ts[j])/(ts[j+1]-ts[j]),0.,1.)
                sample(name) = (1-fraction)*reference[b][j][Symbol(name)]+fraction*reference[b][j+1][Symbol(name)]
                push!(difference,norm(positions[i][b]-SVector(sample("center.x [m]"),sample("center.z [m]"))))
                push!(pitch_difference,-rad2deg(angles[i][b])-sample("pitch [deg]"))
            end
        end
        metrics["body_center_rms_difference_m"] = sqrt(sum(abs2,difference)/length(difference))
        metrics["body_center_max_difference_m"] = maximum(difference)
        metrics["pitch_rms_difference_deg"] = sqrt(sum(abs2,pitch_difference)/length(pitch_difference))
        metrics["pitch_max_difference_deg"] = maximum(abs,pitch_difference)
        paddle = CSV.File(joinpath(reference_directory,"WavePaddle_mkb0010.csv");header = 18,delim = ';')
        metrics["piston_max_displacement_difference_m"] = maximum(abs(Base.invokelatest(case.piston_state,r[Symbol("Time [s]")])[1]-r[Symbol("Position [m]")]) for r in paddle)
    end
    return metrics
end
