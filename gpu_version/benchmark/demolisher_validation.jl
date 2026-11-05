using SPHExampleGPU, CSV, StaticArrays, LinearAlgebra

function demolisher_validation(case, directory)
    rows = collect(CSV.File(joinpath(directory,"Demolisher3D_Floating.csv")))
    times = sort!(unique(row.Time for row in rows))
    initial = [sum(SVector{3,Float64}.(g.Particles.Position))/length(g.Particles)
        for g in case.geometry if g.Type == Floating]
    max_joint, max_speed, max_growth, min_height, penetration = 0.,0.,0.,Inf,0.
    first_motion = nothing
    resting_drift = 0.
    max_block_travel = zeros(16)
    final_block_travel = zeros(16)
    first_energy = nothing
    bodies = filter(g -> g.Type == Floating,case.geometry)
    mass = [Float64(case.constants.m₀)*g.Floating.RelativeWeight*length(g.Particles) for g in bodies]
    inertia = [Float64(case.constants.m₀)*bodies[b].Floating.RelativeWeight*
        sum(dot(r,r)*one(SMatrix{3,3,Float64,9})-r*r' for r in SVector{3,Float64}.(bodies[b].Particles.Position) .- Ref(initial[b])) for b in 1:21]
    for time in times
        frame = sort(filter(r -> r.Time == time,rows); by = r -> r.Body)
        state = (; center = [SVector(ntuple(d -> r[Symbol("Center:",d-1)],3)) for r in frame],
            velocity = [SVector(ntuple(d -> r[Symbol("Velocity:",d-1)],3)) for r in frame],
            orientation = [normalize(SVector(ntuple(d -> r[Symbol("Orientation:",d-1)],4))) for r in frame],
            omega = [SVector(ntuple(d -> r[Symbol("Omega:",d-1)],3)) for r in frame])
        max_joint = max(max_joint,maximum(e.translation_error for e in rigid_joint_errors(case.dynamics,case.geometry,state)))
        max_speed = max(max_speed,maximum(norm,state.velocity))
        energy = 0.
        for b in 1:21
            local_omega = SPHExampleGPU.GPUFloating.rotate_vector(SPHExampleGPU.GPUFloating.quaternion_conjugate(state.orientation[b]),state.omega[b])
            energy += mass[b]*(dot(state.velocity[b],state.velocity[b])/2+Float64(case.constants.g)*state.center[b][3]) + dot(local_omega,inertia[b]*local_omega)/2
        end
        first_energy === nothing && (first_energy = energy)
        max_growth = max(max_growth,(energy-first_energy)/abs(first_energy))
        for block in 1:16
            b = block+5
            travel = norm((state.center[b]-initial[b])[1:2])
            max_block_travel[block] = max(max_block_travel[block],travel)
            final_block_travel[block] = travel
            time <= 0.5 && (resting_drift = max(resting_drift,travel))
            first_motion === nothing && norm(state.velocity[b][1:2]) > 0.3 && (first_motion = Float64(time))
            q = state.orientation[b]
            for i in (-1,1), j in (-1,1), k in (-1,1)
                corner = state.center[b]+SPHExampleGPU.GPUFloating.rotate_vector(q,SVector(i,j,k)*0.25)
                min_height = min(min_height,corner[3])
            end
            local_ball = SPHExampleGPU.GPUFloating.rotate_vector(SPHExampleGPU.GPUFloating.quaternion_conjugate(q),state.center[5]-state.center[b])
            separation = norm(max.(abs.(local_ball).-0.25,0.))-0.35
            penetration = max(penetration,-separation)
        end
    end
    return (; frames = length(times), max_joint_error_m = max_joint, max_body_speed_m_s = max_speed,
        max_relative_energy_growth = max_growth, minimum_block_corner_height_m = min_height,
        max_sphere_box_overlap_m = penetration, first_block_horizontal_motion_s = first_motion,
        early_block_horizontal_drift_m = resting_drift,
        displaced_blocks = count(>(0.1),final_block_travel), max_block_horizontal_travel_m = maximum(max_block_travel))
end

function demolisher_dual_validation(directory)
    blocks = [collect(CSV.File(joinpath(directory,"floating_mk$(mk).csv"))) for mk in 61:76]
    motions = Float64[]
    travel = Float64[]
    for rows in blocks
        initial = SVector(rows[1][Symbol("center.x [m]")],rows[1][Symbol("center.y [m]")])
        push!(travel,norm(SVector(rows[end][Symbol("center.x [m]")],rows[end][Symbol("center.y [m]")])-initial))
        first_motion = findfirst(r -> hypot(r[Symbol("fvel.x [m/s]")],r[Symbol("fvel.y [m/s]")]) > 0.3,rows)
        first_motion === nothing || push!(motions,rows[first_motion][Symbol("time [s]")])
    end
    return (; displaced_blocks = count(>(0.1),travel), first_block_horizontal_motion_s = isempty(motions) ? nothing : minimum(motions))
end
