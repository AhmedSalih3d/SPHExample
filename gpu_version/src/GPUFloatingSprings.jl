# Included inside GPUFloating. Fixed anchors stay in world coordinates and
# body attachments retain their initial offset from the centre of mass.
struct DeviceSpring{D,T,TP}
    anchor::SVector{D,TP}
    offset::SVector{D,TP}
    stiffness::T
    damping::T
    rest_length::T
    coulomb_force::T
end

DeviceSpring(a::SVector{D,TP},r::SVector{D,TP},k::T,c::T,l::T) where {D,T,TP} = DeviceSpring(a,r,k,c,l,zero(T))

@inline function spring_offset(offset::SVector{2,TP},angle) where TP
    s,c = sincos(TP(angle))
    return SVector{2,TP}(c*offset[1]-s*offset[2],s*offset[1]+c*offset[2])
end
@inline spring_offset(offset::SVector{3,TP},orientation) where TP = rotate_vector(SVector{4,TP}(orientation),offset)

@inline function spring_load(spring::DeviceSpring{D,T},center,velocity,omega,orientation,stage_dt = 0,mass = 0,inertia_inverse = nothing,
        reference_velocity = velocity,reference_omega = omega) where {D,T}
    r = spring_offset(spring.offset,orientation)
    delta = center+r-spring.anchor
    point_velocity = velocity + angular_cross(omega, SVector{D,T}(r))
    force = if spring.coulomb_force > 0
        limit = T(Inf)
        length = sqrt(dot(delta,delta))
        if stage_dt > 0 && length > 0
            direction = SVector{D,T}(delta/length)
            arm = floating_torque(SVector{D,T}(r),direction)
            rotational = if D == 2
                inertia_inverse*arm^2
            else
                local_arm = rotate_vector(quaternion_conjugate(orientation),arm)
                dot(local_arm,inertia_inverse*local_arm)
            end
            mobility = inv(mass)+rotational
            reference_point_velocity = reference_velocity+angular_cross(reference_omega,SVector{D,T}(r))
            reference_speed = dot(reference_point_velocity,direction)
            limit = T(abs(reference_speed)/(stage_dt*mobility))
            iszero(dot(point_velocity,direction)) && (point_velocity = reference_point_velocity)
        end
        axial_coulomb_force(delta,point_velocity,spring.coulomb_force,limit)
    else
        axial_spring_force(delta,point_velocity,spring.stiffness,spring.damping,spring.rest_length)
    end
    return force,floating_torque(SVector{D,T}(r),force)
end

function floating_spring_kernel!(force,springs,starts,frozen,center,velocity,omega,orientation,turn,step,half,::Val{D},nbodies,mass,inertia_inverse,reference_velocity,reference_omega) where D
    step_active(step) || return nothing
    b = thread_index()
    b > nbodies && return nothing
    @inbounds begin
        frozen[b] != 0 && return nothing
        pose = D == 2 ? orientation[b]+(half ? turn[b] : zero(eltype(orientation))) : orientation[b]
        total = zero(eltype(center))
        torque = zero(D == 2 ? eltype(force) : SVector{3,eltype(force)})
        for j in starts[b]:starts[b+1]-1
            load,moment = spring_load(springs[j],center[b],velocity[b],omega[b],pose,
                step_dt(step)*(half ? 1 : 0.5),mass[b],inertia_inverse[b],reference_velocity[b],reference_omega[b])
            total += load
            torque += moment
        end
        stride = D == 2 ? 3 : 6
        base = stride*(b-1)
        for d in 1:D
            force[base+d] += total[d]
        end
        for d in 1:(D == 2 ? 1 : 3)
            force[base+D+d] += floating_torque_component(torque,d)
        end
    end
    return nothing
end

"Add spring forces and torques at the full state or predictor state, on CUDA."
function launch_floating_springs!(f,step,half::Bool)
    launch_floating_torsions!(f,step,half)
    isempty(f.springs) && return nothing
    D = floating_dimension(f.center)
    orientation = D == 2 ? f.angle : (half ? f.orientation_half : f.orientation)
    @cuda threads=128 blocks=cld(Int(f.nbodies),128) floating_spring_kernel!(f.force,f.springs,f.spring_starts,f.frozen,
        half ? f.center_half : f.center,half ? f.velocity_half : f.velocity,half ? f.omega_half : f.omega,
        orientation,f.turn,step,half,Val(D),f.nbodies,f.mass,f.inertia_inv,f.velocity,f.omega)
    return nothing
end

"Download the accepted spring endpoints, extension and force for output only."
function floating_spring_state(f)
    isempty(f.springs) && return NamedTuple[]
    springs,starts = Array(f.springs),Array(f.spring_starts)
    state = floating_state(f)
    D = floating_dimension(f.center)
    records = NamedTuple[]
    for b in 1:Int(f.nbodies), j in starts[b]:starts[b+1]-1
        spring = springs[j]
        orientation = D == 2 ? state.angle[b] : state.orientation[b]
        point = state.center[b]+spring_offset(spring.offset,orientation)
        force,torque = spring_load(spring,state.center[b],state.velocity[b],state.omega[b],orientation)
        length = sqrt(sum(abs2,point-spring.anchor))
        push!(records,(;body=b,spring=j,anchor=spring.anchor,attachment=point,length,
            extension=length-spring.rest_length,force,torque))
    end
    return records
end
