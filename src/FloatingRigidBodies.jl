module FloatingRigidBodies

using StaticArrays
using ..SimulationGeometry

export FloatingBody, construct_floating_bodies, update_floating_bodies!

struct FloatingBody{D,T}
    indices::Vector{Int}
    relpos::Vector{SVector{D,T}}
    mass::T
    velocity::SVector{D,T}
    center::SVector{D,T}
end

function construct_floating_bodies(SimGeometry::Vector{Geometry{D,T}},
                                   SimParticles) where {D,T}
    bodies = FloatingBody{D,T}[]
    for geom in SimGeometry
        if geom.Type == Floating
            idx = findall(i -> (SimParticles.Type[i] == Floating &&
                                SimParticles.GroupMarker[i] == geom.GroupMarker),
                            eachindex(SimParticles.Type))
            rel = [SimParticles.Position[i] - geom.COG for i in idx]
            vel = zeros(SVector{D,T})
            push!(bodies, FloatingBody{D,T}(idx, rel, geom.Mass, vel, geom.COG))
        end
    end
    return bodies
end

function update_floating_bodies!(bodies::Vector{FloatingBody{D,T}},
                                 SimParticles, dt) where {D,T}
    Position     = SimParticles.Position
    Velocity     = SimParticles.Velocity
    Acceleration = SimParticles.Acceleration
    for body in bodies
        n = length(body.indices)
        acc = zero(SVector{D,T})
        for i in body.indices
            acc += Acceleration[i]
        end
        acc /= n
        body.velocity += acc * dt
        body.center   += body.velocity * dt
        for (pidx, rel) in zip(body.indices, body.relpos)
            Position[pidx]     = body.center + rel
            Velocity[pidx]     = body.velocity
            Acceleration[pidx] = acc
        end
    end
    return nothing
end

end # module
