# Included inside GPUFloating. Loads augment the existing predictor/corrector;
# the joint solver continues to enforce only the bilateral constraints.
struct DeviceRigidTorsion{T}
    a::Int32
    b::Int32
    axis::SVector{3,T}
    stiffness::T
    damping::T
    rest_angle::T
end

function attach_rigid_torsions(f,specification)
    T,D = eltype(f.mass),floating_dimension(f.center)
    body,inertia = Array(f.body),Array(f.inertia_inv)
    links = DeviceRigidTorsion{T}[]
    stiffness,damping = zeros(Float64,Int(f.nbodies)),zeros(Float64,Int(f.nbodies))
    for j in specification.joints
        iszero(j.stiffness) && iszero(j.damping) && continue
        a,b = body[j.body_a],j.body_b == 0 ? Int32(0) : body[j.body_b]
        push!(links,DeviceRigidTorsion(a,b,SVector{3,T}(j.axis),T(j.stiffness),T(j.damping),T(j.rest_angle)))
        all(isfinite,(T(j.stiffness),T(j.damping),T(j.rest_angle))) || throw(ArgumentError("torsional coefficients must be representable in $T"))
        for index in (a,b)
            index == 0 && continue
            # Gershgorin bound on the coupled angular stiffness/damping matrix.
            inverse = D == 2 ? Float64(inertia[index]) : sqrt(sum(abs2,inertia[index]))
            factor = b == 0 ? 1 : 2
            stiffness[index] += factor*j.stiffness*inverse
            damping[index] += factor*j.damping*inverse
        end
    end
    isempty(links) && return f
    # Resolve the elastic frequency and damping decay independently. The
    # midpoint decay stability interval ends at dt*rate = 2; retain a fourfold
    # margin instead of unnecessarily imposing the elastic factor on damping.
    elastic_rate,decay_rate = sqrt(maximum(stiffness)),maximum(damping)
    limit = min(elastic_rate > 0 ? 0.2/elastic_rate : Inf,decay_rate > 0 ? 0.5/decay_rate : Inf)
    isfinite(limit) && limit >= floatmin(T) || throw(ArgumentError("torsional timestep is not representable in $T"))
    return merge(f,(;torsions = CuArray(links),spring_dt = min(f.spring_dt,limit)))
end

@inline function torsion_moment(j,orientation,omega,turn,half,::Val{2})
    a,b = j.a,j.b
    theta_a = orientation[a]+(half ? turn[a] : zero(eltype(orientation)))
    theta_b = b == 0 ? zero(theta_a) : orientation[b]+(half ? turn[b] : zero(eltype(orientation)))
    speed = omega[a]-(b == 0 ? zero(eltype(omega)) : omega[b])
    # World Y is minus the positive native planar rotation axis.
    return -j.stiffness*(theta_a-theta_b+j.axis[2]*j.rest_angle)-j.damping*speed
end

@inline function torsion_moment(j,orientation,omega,turn,half,::Val{3})
    a,b = j.a,j.b
    qa = orientation[a]
    qb = b == 0 ? SVector(one(eltype(qa)),zero(eltype(qa)),zero(eltype(qa)),zero(eltype(qa))) : orientation[b]
    axis = rotate_vector(qa,j.axis)
    relative = quaternion_multiply(qa,quaternion_conjugate(qb))
    relative[1] < 0 && (relative = -relative)
    angle = 2atan(dot(axis,SVector(relative[2],relative[3],relative[4])),relative[1])
    speed = dot(axis,omega[a]-(b == 0 ? zero(eltype(omega)) : omega[b]))
    return (-j.stiffness*(angle-j.rest_angle)-j.damping*speed)*axis
end

function floating_torsion_kernel!(force,links,orientation,omega,turn,frozen,step,half,::Val{D},n) where D
    step_active(step) || return nothing
    b = thread_index()
    b > n && return nothing
    @inbounds begin
        frozen[b] != 0 && return nothing
        torque = zero(D == 2 ? eltype(force) : SVector{3,eltype(force)})
        for link in links
            if link.a == b || link.b == b
                moment = torsion_moment(link,orientation,omega,turn,half,Val(D))
                torque += link.a == b ? moment : -moment
            end
        end
        base = (D == 2 ? 3 : 6)*(b-1)+D
        for d in 1:(D == 2 ? 1 : 3)
            force[base+d] += floating_torque_component(torque,d)
        end
    end
    return nothing
end

function launch_floating_torsions!(f,step,half)
    hasproperty(f,:torsions) || return nothing
    D = floating_dimension(f.center)
    orientation = D == 2 ? f.angle : (half ? f.orientation_half : f.orientation)
    @cuda threads=128 blocks=cld(Int(f.nbodies),128) floating_torsion_kernel!(f.force,f.torsions,orientation,
        half ? f.omega_half : f.omega,f.turn,f.frozen,step,half,Val(D),f.nbodies)
    return nothing
end

function attach_rigid_programs(f,specification)
    result = attach_body_springs(attach_rigid_velocities(attach_rigid_torsions(f,specification),specification),specification)
    return merge(result,(;rigid_configuration = merge(result.rigid_configuration,
        (;motor_bodies = length(specification.velocities),body_springs = length(specification.springs)))))
end
