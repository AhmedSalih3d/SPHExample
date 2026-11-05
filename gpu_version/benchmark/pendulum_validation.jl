using CSV, StaticArrays, LinearAlgebra
include("pendulum_reference.jl")

function pendulum_validation(case, directory; reference = true)
    rows = collect(CSV.File(joinpath(directory,"Pendulum3D_Floating.csv")))
    times = sort!(unique(row.Time for row in rows))
    properties = pendulum_properties(case)
    planar = reference ? pendulum_planar_reference(case,times) : nothing
    max_joint, max_plane, max_reference, energy_error = 0.,0.,0.,0.
    initial_energy = sum(properties.mass[b] * properties.g * properties.center[b][3] for b in 1:4)
    energy_scale = sum(properties.gravity)
    for (index,time) in enumerate(times)
        frame = sort(filter(row -> row.Time == time, rows);by = row -> row.Body)
        state = (;
            center = [SVector(ntuple(d -> row[Symbol("Center:",d-1)],3)) for row in frame],
            velocity = [SVector(ntuple(d -> row[Symbol("Velocity:",d-1)],3)) for row in frame],
            orientation = [SVector(ntuple(d -> row[Symbol("Orientation:",d-1)],4)) for row in frame],
            omega = [SVector(ntuple(d -> row[Symbol("Omega:",d-1)],3)) for row in frame])
        max_joint = max(max_joint,maximum(e.translation_error for e in rigid_joint_errors(case.dynamics,case.geometry,state)))
        max_plane = max(max_plane,maximum(abs(c[1]-5) for c in state.center))
        planar === nothing || (max_reference = max(max_reference,maximum(norm.(state.center-planar[index]))))
        energy = 0.
        for b in 1:4
            q = normalize(state.orientation[b])
            local_omega = SPHExampleGPU.GPUFloating.rotate_vector(SPHExampleGPU.GPUFloating.quaternion_conjugate(q),state.omega[b])
            energy += properties.mass[b] * (dot(state.velocity[b],state.velocity[b])/2 + properties.g * state.center[b][3]) +
                dot(local_omega,properties.inertia[b]*local_omega)/2
        end
        energy_error = max(energy_error,abs(energy-initial_energy)/energy_scale)
    end
    return (; frames=length(times), max_joint_error_m=max_joint, max_plane_error_m=max_plane,
        max_planar_reference_error_m=planar === nothing ? nothing : max_reference,
        max_relative_energy_error=energy_error)
end

function pendulum_dual_validation(case, directory)
    rows = [collect(CSV.File(joinpath(directory,"floating_mk$(mk).csv"))) for mk in 10:13]
    times = [Float64(row[Symbol("time [s]")]) for row in rows[1]]
    reference = pendulum_planar_reference(case,times)
    error = maximum(norm(SVector(row[Symbol("center.x [m]")],row[Symbol("center.y [m]")],
        row[Symbol("center.z [m]")]) - reference[i][b]) for b in 1:4 for (i,row) in enumerate(rows[b]))
    return (; frames = length(times), max_planar_reference_error_m = error)
end
