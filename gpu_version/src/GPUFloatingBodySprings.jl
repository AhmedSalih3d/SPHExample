# Moving-endpoint spring loads augment the existing evaluated body force.
struct DeviceBodySpring{D,T,TP}
    a::Int32
    b::Int32
    ra::SVector{D,TP}
    rb::SVector{D,TP}
    stiffness::T
    damping::T
    rest_length::T
end
function attach_body_springs(f,specification)
    isempty(specification.springs) && return f
    T,TP,D = eltype(f.mass),eltype(eltype(f.center)),floating_dimension(f.center)
    body,centres,mass,inertia = Array(f.body),Array(f.center),Array(f.mass),Array(f.inertia_inv)
    springs = DeviceBodySpring{D,T,TP}[]
    rates_k,rates_c = zeros(Float64,Int(f.nbodies)),zeros(Float64,Int(f.nbodies))
    for spring in specification.springs
        length(spring.attachment) == D || throw(ArgumentError("body spring dimension must match the simulation"))
        all(g -> g <= length(body) && body[g] > 0,(spring.a,spring.b)) || throw(ArgumentError("spring markers must identify floating bodies"))
        a,b = body[spring.a],body[spring.b]
        ra,rb = SVector{D,TP}(spring.attachment)-centres[a],SVector{D,TP}(spring.anchor)-centres[b]
        coefficients = T.((spring.stiffness,spring.damping,spring.rest_length))
        all(isfinite,coefficients) || throw(ArgumentError("body spring coefficients must be representable in $T"))
        push!(springs,DeviceBodySpring(a,b,ra,rb,coefficients...))
        for (index,arm) in ((a,ra),(b,rb))
            rotational = D == 2 ? Float64(inertia[index]) : norm(inertia[index])
            mobility = inv(Float64(mass[index]))+dot(arm,arm)*rotational
            rates_k[index] += 2spring.stiffness*mobility
            rates_c[index] += 2spring.damping*mobility
        end
    end
    rate = sqrt(maximum(rates_k))+maximum(rates_c)
    limit = rate > 0 ? 0.2/rate : Inf
    limit >= floatmin(T) || throw(ArgumentError("body spring timestep is not representable in $T"))
    return merge(f,(;body_springs = CuArray(springs),spring_dt = min(f.spring_dt,limit)))
end
@inline function body_spring_load(link,center,velocity,omega,orientation,turn,half,::Val{D}) where D
    a,b = link.a,link.b
    T = eltype(eltype(velocity))
    @inbounds begin
        qa = D == 2 ? orientation[a]+(half ? turn[a] : zero(eltype(orientation))) : orientation[a]
        qb = D == 2 ? orientation[b]+(half ? turn[b] : zero(eltype(orientation))) : orientation[b]
        ra,rb = spring_offset(link.ra,qa),spring_offset(link.rb,qb)
        delta = center[a]+ra-center[b]-rb
        relative = velocity[a]+angular_cross(omega[a],SVector{D,T}(ra))-velocity[b]-angular_cross(omega[b],SVector{D,T}(rb))
        load = axial_spring_force(delta,relative,link.stiffness,link.damping,link.rest_length)
    end
    return load,floating_torque(SVector{D,T}(ra),load),floating_torque(SVector{D,T}(rb),-load)
end
function body_springs_kernel!(force,links,center,velocity,omega,orientation,turn,frozen,step,half,::Val{D},n) where D
    step_active(step) || return nothing
    b = thread_index()
    b > n && return nothing
    @inbounds begin
        frozen[b] != 0 && return nothing
        load = zero(eltype(velocity))
        torque = zero(eltype(omega))
        for link in links
            if b == link.a || b == link.b
                value,ta,tb = body_spring_load(link,center,velocity,omega,orientation,turn,half,Val(D))
                load += b == link.a ? value : -value
                torque += b == link.a ? ta : tb
            end
        end
        base = (D == 2 ? 3 : 6)*(b-1)
        for d in 1:D
            force[base+d] += load[d]
        end
        for d in 1:(D == 2 ? 1 : 3)
            force[base+D+d] += floating_torque_component(torque,d)
        end
    end
    return nothing
end
function launch_body_springs!(f,step,half)
    hasproperty(f,:body_springs) || return nothing
    D = floating_dimension(f.center)
    orientation = D == 2 ? f.angle : (half ? f.orientation_half : f.orientation)
    @cuda threads=128 blocks=cld(Int(f.nbodies),128) body_springs_kernel!(f.force,f.body_springs,
        half ? f.center_half : f.center,half ? f.velocity_half : f.velocity,half ? f.omega_half : f.omega,
        orientation,f.turn,f.frozen,step,half,Val(D),f.nbodies)
    return nothing
end
