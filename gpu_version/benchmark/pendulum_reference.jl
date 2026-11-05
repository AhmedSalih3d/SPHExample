# Independent planar four-link equations. This reference does not call the
# CUDA joint solver and uses a small-step RK4 integrator with no new dependency.
using StaticArrays, LinearAlgebra

function pendulum_properties(case)
    bodies = filter(g -> g.Type == Floating, case.geometry)
    points = [SVector{3,Float64}.(g.Particles.Position) for g in bodies]
    center = [sum(p) / length(p) for p in points]
    particle_mass = [Float64(case.constants.m₀) * Float64(g.Floating.RelativeWeight) for g in bodies]
    mass = particle_mass .* length.(points)
    inertia = [particle_mass[b] * sum(dot(r,r) * one(SMatrix{3,3,Float64,9}) - r * r' for r in points[b] .- Ref(center[b])) for b in 1:4]
    offsets = [center[b][2] - (4 + b) for b in 1:4]
    arms = SMatrix{4,4,Float64}(ntuple(16) do index
        body, joint = mod1(index,4), (index-1)÷4+1
        joint < body ? 1.0 : joint == body ? offsets[body] : 0.0
    end)
    coupling = arms' * Diagonal(SVector{4}(mass)) * arms
    gravity = Float64(case.constants.g) * arms' * SVector{4}(mass)
    return (; mass, center, inertia, arms, coupling, gravity, g = Float64(case.constants.g))
end

function pendulum_planar_rhs(state, properties)
    theta = SVector{4}(state[1:4])
    omega = SVector{4}(state[5:8])
    matrix = SMatrix{4,4,Float64}(ntuple(16) do index
        i, j = mod1(index,4), (index-1)÷4+1
        properties.coupling[i,j] * cos(theta[i]-theta[j]) + (i == j ? properties.inertia[i][1,1] : 0.0)
    end)
    force = SVector{4}(ntuple(4) do i
        properties.gravity[i] * sin(theta[i]) +
            sum(properties.coupling[i,j] * sin(theta[i]-theta[j]) * omega[j]^2 for j in 1:4)
    end)
    acceleration = -(matrix \ force)
    return SVector{8}(omega..., acceleration...)
end

function pendulum_planar_positions(state, properties)
    return [SVector(5.0,
        5 + sum(properties.arms[b,j] * sin(state[j]) for j in 1:4),
        5 - sum(properties.arms[b,j] * cos(state[j]) for j in 1:4)) for b in 1:4]
end

function pendulum_planar_reference(case, times; timestep = 0.00005)
    properties = pendulum_properties(case)
    state = SVector{8,Float64}(pi/2, pi/2, pi/2, pi/2, 0, 0, 0, 0)
    time = 0.0
    positions = Vector{SVector{3,Float64}}[]
    for target in times
        while time < target
            dt = min(timestep, target-time)
            k1 = pendulum_planar_rhs(state,properties)
            k2 = pendulum_planar_rhs(state+dt*k1/2,properties)
            k3 = pendulum_planar_rhs(state+dt*k2/2,properties)
            k4 = pendulum_planar_rhs(state+dt*k3,properties)
            state += dt * (k1+2k2+2k3+k4) / 6
            time = dt == target-time ? target : time+dt
        end
        push!(positions,pendulum_planar_positions(state,properties))
    end
    return positions
end
