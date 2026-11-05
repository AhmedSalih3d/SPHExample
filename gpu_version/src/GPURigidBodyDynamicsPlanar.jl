# Planar bilateral joints use the existing 2D floating predictor/corrector.
# Public joint coordinates are world (x,y,z); the simulation plane is (x,z).
struct DevicePlanarRigidJoint{T,TP}
    kind::UInt8
    a::Int32
    b::Int32
    ra::SVector{2,TP}
    rb::SVector{2,TP}
    axis::SVector{2,T}
end

@inline planar_point(v) = SVector(v[1],v[3])
@inline planar_perpendicular(v) = SVector(-v[2],v[1])
@inline planar_cross(a,b) = a[1]*b[2]-a[2]*b[1]
@inline function planar_rotate(angle, r)
    s,c = sincos(angle)
    return SVector(c*r[1]-s*r[2],s*r[1]+c*r[2])
end

function attach_planar_rigid_dynamics(f,specification,particles,kernel;gravity = 0)
    f.active || throw(ArgumentError("native multibody dynamics needs at least one floating body"))
    f.constrained && throw(ArgumentError("use native joints instead of per-body FloatingConstraints in multibody mode"))
    specification.contacts === nothing || throw(ArgumentError("2D native multibody dynamics supports bilateral joints; rigid contacts currently require 3D"))
    T,TP = eltype(f.mass),eltype(eltype(f.center))
    solid_only = all(!=(Fluid),particles.Type)
    specification.max_timestep === nothing || solid_only || throw(ArgumentError("native MaxTimeStep requires a case with no fluid; coupled fluid retains the SPH timestep"))
    specification.max_timestep === nothing || (isfinite(T(specification.max_timestep)) && T(specification.max_timestep) > 0) ||
        throw(ArgumentError("MaxTimeStep must be positive and representable in $T"))
    sort(Int.(particles.ID)) == collect(1:length(particles)) || throw(ArgumentError("native rigid dynamics requires dense IDs 1:N"))
    centres,body = Array(f.center),Array(f.body)
    joints = DevicePlanarRigidJoint{T,TP}[]
    for joint in specification.joints
        iszero(joint.attachment[2]) && iszero(joint.anchor[2]) || throw(ArgumentError("planar joint points must have world y = 0; use world (x,0,z) coordinates"))
        if joint.kind == 1
            abs(joint.axis[1])+abs(joint.axis[3]) <= 16eps(Float64) || throw(ArgumentError("a planar hinge axis must be parallel to world Y"))
        elseif joint.kind == 2
            iszero(joint.axis[2]) || throw(ArgumentError("a planar sliding direction must lie in the world X-Z plane"))
        end
        a = joint.body_a <= length(body) ? body[joint.body_a] : Int32(0)
        b = joint.body_b == 0 ? Int32(0) : joint.body_b <= length(body) ? body[joint.body_b] : Int32(0)
        a != 0 && (joint.body_b == 0 || b != 0) || throw(ArgumentError("joint markers must identify floating groups (or zero for world)"))
        axis = joint.kind == 2 ? SVector{2,T}(planar_point(joint.axis)) : SVector{2,T}(1,0)
        push!(joints,DevicePlanarRigidJoint(joint.kind,a,b,SVector{2,TP}(planar_point(joint.attachment))-centres[a],
            b == 0 ? SVector{2,TP}(planar_point(joint.anchor)) : SVector{2,TP}(planar_point(joint.anchor))-centres[b],axis))
    end
    joint_order,joint_starts,joint_colours = rigid_edge_colours([(Int(j.a),Int(j.b)) for j in joints])
    references = zeros(SVector{2,TP},length(particles))
    radii = zeros(T,Int(f.nbodies))
    for i in eachindex(particles.ID)
        b = body[particles.GroupMarker[i]]
        if b != 0
            references[particles.ID[i]] = particles.Position[i]-centres[b]
            radii[b] = max(radii[b],T(sqrt(sum(abs2,references[particles.ID[i]]))))
        end
    end
    mechanics = (; joints = CuArray(joints),joint_order,joint_starts,joint_colours,references = CuArray(references),
        angle = similar(f.angle),old_angle = similar(f.angle),free_velocity = similar(f.velocity),free_omega = similar(f.omega),
        half_impulse = CUDA.zeros(SVector{2,T},Int(f.nbodies)),half_angular_impulse = CUDA.zeros(T,Int(f.nbodies)),
        reaction_force = CUDA.zeros(SVector{2,T},Int(f.nbodies)),reaction_torque = CUDA.zeros(T,Int(f.nbodies)),
        iterations = Int32(specification.iterations),has_contacts = false,solid_only,radii = CuArray(radii),
        parameters = (; distance = zero(T)),maximum_timestep = something(specification.max_timestep,0.))
    configuration = (; backend = :native_cuda,joints = length(joints),bodies = Int(f.nbodies),dimensions = 2,
        iterations = specification.iterations,contacts = false,precision = nameof(T),position_precision = nameof(TP),
        solid_interactions = :native_constraints,max_timestep = specification.max_timestep)
    gravity_force = CuArray([SVector{2,T}(0,-T(gravity)*mass) for mass in Array(f.mass)])
    return attach_rigid_torsions(merge(f,(;mechanics,rigid_configuration = configuration,constraint_force = mechanics.reaction_force,
        constraint_torque = mechanics.reaction_torque,gravity_force)),specification)
end

@inline planar_inverse_inertia(f,b,step) = b == 0 || rigid_mobility(f,b,step) == 0 ? zero(eltype(f.mass)) : @inbounds(f.inertia_inv[b])
@inline planar_arm(f,angle,b,r) = b == 0 ? zero(SVector{2,eltype(f.mass)}) : planar_rotate(@inbounds(angle[b]),SVector{2,eltype(f.mass)}(r))
@inline planar_velocity(v,w,b,r) = b == 0 ? zero(eltype(v)) : @inbounds(v[b]+w[b]*planar_perpendicular(r))

@inline function apply_planar_impulse!(f,center,angle,velocity,omega,b,r,impulse,step,position)
    mass_inv = rigid_mobility(f,b,step)
    mass_inv == 0 && return nothing
    turn = planar_inverse_inertia(f,b,step)*planar_cross(r,impulse)
    @inbounds if position
        center[b] += eltype(center)(mass_inv*impulse)
        angle[b] += turn
    else
        velocity[b] += mass_inv*impulse
        omega[b] += turn
    end
    return nothing
end

@inline function solve_planar_joint!(f,center,angle,velocity,omega,joint,step,position)
    T = eltype(f.mass)
    a,b = joint.a,joint.b
    ra,rb = planar_arm(f,angle,a,joint.ra),planar_arm(f,angle,b,joint.rb)
    error = if position
        @inbounds SVector{2,T}(center[a]+ra-(b == 0 ? joint.rb : center[b]+rb))
    else
        planar_velocity(velocity,omega,a,ra)-planar_velocity(velocity,omega,b,rb)
    end
    iszero(error) && return nothing
    ma,mb = rigid_mobility(f,a,step),rigid_mobility(f,b,step)
    ia,ib = planar_inverse_inertia(f,a,step),planar_inverse_inertia(f,b,step)
    impulse = zero(error)
    if joint.kind == 2
        normal = planar_perpendicular(joint.axis)
        mobility = (ma+mb)*dot(normal,normal)+ia*planar_cross(ra,normal)^2+ib*planar_cross(rb,normal)^2
        mobility > 0 && (impulse = -(dot(error,normal)/mobility)*normal)
    else
        pa,pb = planar_perpendicular(ra),planar_perpendicular(rb)
        mobility = (ma+mb)*one(SMatrix{2,2,T,4})+ia*(pa*pa')+ib*(pb*pb')
        det(mobility) > 0 && (impulse = -(mobility\error))
    end
    apply_planar_impulse!(f,center,angle,velocity,omega,a,ra,impulse,step,position)
    apply_planar_impulse!(f,center,angle,velocity,omega,b,rb,-impulse,step,position)
    return nothing
end

@inline function planar_joint_iterations!(f,d,center,velocity,omega,step,position)
    thread = threadIdx().x
    for iteration in 1:d.iterations,colour in 1:d.joint_colours
        @inbounds index,stop = d.joint_starts[colour]+thread-1,d.joint_starts[colour+1]
        while index < stop
            @inbounds solve_planar_joint!(f,center,d.angle,velocity,omega,d.joints[d.joint_order[index]],step,position)
            index += blockDim().x
        end
        sync_threads()
    end
    return nothing
end

function planar_prepare_kernel!(old_angle,angle,step,n)
    step_active(step) || return nothing
    b = thread_index()
    @inbounds b <= n && (old_angle[b] = angle[b])
    return nothing
end

function planar_project_kernel!(f,d,center,velocity,omega,step,::Val{Final},::Val{Initial}) where {Final,Initial}
    step_active(step) || return nothing
    b = threadIdx().x
    while b <= f.nbodies
        @inbounds begin
            d.angle[b] = Final || f.frozen[b] != 0 ? f.angle[b] : f.angle[b]+f.turn[b]
            d.free_velocity[b],d.free_omega[b] = velocity[b],omega[b]
        end
        b += blockDim().x
    end
    sync_threads()
    Initial || planar_joint_iterations!(f,d,center,velocity,omega,step,true)
    planar_joint_iterations!(f,d,center,velocity,omega,step,false)
    b = threadIdx().x
    while b <= f.nbodies
        @inbounds if !Initial
            impulse = f.mass[b]*(velocity[b]-d.free_velocity[b])
            angular = f.inertia[b]*(omega[b]-d.free_omega[b])
            if Final
                d.reaction_force[b] = (impulse+d.half_impulse[b])/eltype(f.mass)(step_dt(step))
                d.reaction_torque[b] = (angular+d.half_angular_impulse[b])/eltype(f.mass)(step_dt(step))
                f.angle[b] = d.angle[b]
            else
                d.half_impulse[b],d.half_angular_impulse[b] = 2impulse,2angular
                f.velocity[b] += 2(velocity[b]-d.free_velocity[b])
                f.omega[b] += 2(omega[b]-d.free_omega[b])
            end
            f.turn[b] = d.angle[b]-d.old_angle[b]
            s,c = sincos(f.turn[b])
            f.rotation[b] = SVector(s,c)
        end
        b += blockDim().x
    end
    return nothing
end

planar_device(f) = (; f.nbodies,f.mass,f.inertia,f.inertia_inv,f.pause,f.frozen,f.velocity,f.omega,f.angle,f.turn,f.rotation)

function prepare_planar_rigid_stage!(f,step)
    @cuda threads=128 blocks=cld(Int(f.nbodies),128) planar_prepare_kernel!(f.mechanics.old_angle,f.angle,step,f.nbodies)
    return nothing
end

function solve_planar_rigid_stage!(f,step,final)
    center,velocity,omega = final ? (f.center,f.velocity,f.omega) : (f.center_half,f.velocity_half,f.omega_half)
    @cuda threads=128 blocks=1 planar_project_kernel!(planar_device(f),f.mechanics,center,velocity,omega,step,Val(final),Val(false))
    return nothing
end

function initialize_planar_rigid_velocities!(f,time)
    @cuda threads=128 blocks=1 planar_project_kernel!(planar_device(f),f.mechanics,f.center,f.velocity,f.omega,
        HostStep(zero(eltype(f.mass)),Float64(time)),Val(true),Val(true))
    return nothing
end

function planar_particles_kernel!(positions,velocities,ids,types,groups,indices,body,references,center,angle,velocity,omega,step,pos_cells,cells,grid,support)
    step_active(step) || return nothing
    t = thread_index()
    t > length(indices) && return nothing
    @inbounds begin
        i = indices[t]
        types[i] == Frozen && return nothing
        b = body[groups[i]]
        r = planar_rotate(angle[b],SVector{2,eltype(omega)}(references[ids[i]]))
        x = center[b]+r
        positions[i] = x
        velocities[i] = velocity[b]+omega[b]*planar_perpendicular(r)
        store_pos_cell!(pos_cells,i,x,cells,grid,support)
    end
    return nothing
end

function place_planar_rigid_particles!(f,positions,velocities,ids,types,groups,step,final;pos_cells,cells,grid,support)
    ids === nothing && throw(ArgumentError("native rigid particle placement requires stable particle IDs"))
    center,velocity,omega = final ? (f.center,f.velocity,f.omega) : (f.center_half,f.velocity_half,f.omega_half)
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(length(f.indices),ELEMENTWISE_THREADS) planar_particles_kernel!(positions,velocities,ids,types,groups,
        f.indices,f.body,f.mechanics.references,center,f.mechanics.angle,velocity,omega,step,pos_cells,cells,grid,support)
    return nothing
end

function planar_rigid_joint_errors(specification,geometry,state)
    bodies = filter(g -> g.Type == Floating,geometry)
    mapping = Dict(g.GroupMarker => b for (b,g) in enumerate(bodies))
    initial = initial_body_center.(bodies)
    return [let a = mapping[j.body_a],b = j.body_b == 0 ? 0 : mapping[j.body_b]
        pa = state.center[a]+planar_rotate(state.angle[a],planar_point(j.attachment)-initial[a])
        pb = b == 0 ? planar_point(j.anchor) : state.center[b]+planar_rotate(state.angle[b],planar_point(j.anchor)-initial[b])
        delta = pa-pb
        j.kind == 2 && (axis = planar_point(j.axis); delta -= dot(delta,axis)*axis)
        (;joint = i,translation_error = norm(delta),rotation_error = 0.)
    end for (i,j) in enumerate(specification.joints)]
end
