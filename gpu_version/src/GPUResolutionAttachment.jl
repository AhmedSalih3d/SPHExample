# The limits remain in reference coordinates. Only two small poses change on
# the device; all ownership and interface queries use the same accepted pose.
struct AttachedResolutionLimits{L,P}
    reference::L
    frames::P
end
Adapt.@adapt_structure AttachedResolutionLimits

@inline resolution_base(limits) = limits
@inline resolution_base(limits::AttachedResolutionLimits) = limits.reference
@inline resolution_local(x,limits) = x
@inline function resolution_local(x,limits::AttachedResolutionLimits)
    pose = @inbounds limits.frames[1]
    return transpose(pose.rotation)*(x-pose.offset)
end
@inline function resolution_owned(x,limits,fine)
    local_x,base = resolution_local(x,limits),resolution_base(limits)
    return inside_box(local_x,base.fine_inner_lo,base.fine_inner_hi) == fine
end
resolution_host_limits(limits) = limits
resolution_host_limits(limits::AttachedResolutionLimits) = AttachedResolutionLimits(limits.reference,Array(limits.frames))

@inline resolution_face(limits,point,normal,dt) = (point,normal,zero(point))
@inline function resolution_face(limits::AttachedResolutionLimits,point,normal,dt)
    current,previous = @inbounds limits.frames[1],limits.frames[2]
    world = motion_position(current,point)
    # Secant velocity measures actual swept displacement, including a step
    # ending at a prescribed pause where the instantaneous velocity is zero.
    velocity = (world-motion_position(previous,point))/dt
    return world,current.rotation*normal,velocity
end

# MotionDetails geometry is the position at t=0, including when its active
# interval starts before zero. Keep unbounded durations without sampling a
# table or changing that initial reference position.
struct ResolutionTranslation <: MotionTrajectories.AbstractTrajectory
    velocity::SVector{2,Float64}
    start::Float64
    finish::Float64
end
PrescribedMotions.device_trajectory(m::ResolutionTranslation) = m
@inline function MotionTrajectories.trajectory_state(m::ResolutionTranslation,time)
    elapsed = clamp(time,m.start,m.finish)-clamp(0.0,m.start,m.finish)
    velocity = m.start <= time < m.finish ? m.velocity : zero(m.velocity)
    return (elapsed*m.velocity,velocity,zero(velocity))
end

function resolution_attachment_config(config,geometry,rotation,motion,coarse_kernel)
    marker = config.follow
    marker === nothing && return nothing
    bodies = filter(g -> g.GroupMarker == marker,config.fine_geometry)
    length(bodies) == 1 && only(bodies).Type in (Moving,Floating) ||
        throw(ArgumentError("follow must identify exactly one Moving or Floating group in fine_geometry"))
    body = only(bodies)
    isempty(body.Particles) && throw(ArgumentError("the followed fine body must contain particles"))
    for g in (geometry...,config.fine_geometry...)
        if g.GroupMarker == marker
            g.Type == body.Type || throw(ArgumentError("the followed GroupMarker must be exclusive to its body in both domains"))
        else
            g.Type in (Fixed,Fluid) && g.Motion === nothing && g.Floating === nothing ||
                throw(ArgumentError("attached refinement supports one moving/floating body; other groups must be fixed walls or fluid"))
        end
    end
    h = Float64(coarse_kernel.h)
    limits = resolution_limits(config.region,h,h/config.region.ratio)
    lo,hi = limits.coarse_inner_lo .+ coarse_kernel.H,limits.coarse_inner_hi .- coarse_kernel.H
    all(hi .> lo) && all(x -> inside_box(x,lo,hi),body.Particles.Position) ||
        throw(ArgumentError("the followed body must fit inside the coarse exclusion rectangle with one coarse support radius of clearance; enlarge the refinement box"))
    if body.Type == Floating
        body.Floating !== nothing && body.Motion === nothing && motion === nothing && rotation === nothing ||
            throw(ArgumentError("a followed floating body requires FloatingDetails and cannot also have prescribed motion"))
        return (;marker,motion=nothing)
    end
    body.Floating === nothing || throw(ArgumentError("a Moving body cannot have FloatingDetails"))
    count(!isnothing,(body.Motion,rotation,motion)) == 1 ||
        throw(ArgumentError("a followed Moving body needs exactly one of MotionDetails, SimMotion or SimRotation"))
    if body.Motion !== nothing
        m = body.Motion
        m.MoveParticles || throw(ArgumentError("follow requires physical motion; MoveParticles=false is a velocity-only wall"))
        isfinite(m.StartTime) && !isnan(m.Duration) && m.Duration >= 0 &&
            isfinite(m.Velocity) && all(isfinite,m.Direction) ||
            throw(ArgumentError("followed MotionDetails needs finite velocity/direction/start and a nonnegative duration"))
        trajectory = ResolutionTranslation(SVector{2,Float64}(m.Direction)*m.Velocity,
            Float64(m.StartTime),Float64(m.StartTime)+Float64(m.Duration))
        motion = PrescribedMotion(marker;translation=trajectory)
    elseif rotation !== nothing
        rotation.group_marker == marker || throw(ArgumentError("SimRotation must target the followed body"))
        motion = PrescribedMotion(marker;center=rotation.center,
            rotation=SampledTrajectory(rotation.times,rotation.angles))
    end
    system = motion isa PrescribedMotion ? MotionSystem(motion) : motion
    system isa MotionSystem || throw(ArgumentError("SimMotion must be a PrescribedMotion or MotionSystem"))
    all(n -> n.group_marker === nothing || n.group_marker == marker,system.motions) &&
        count(n -> n.group_marker == marker,system.motions) == 1 ||
        throw(ArgumentError("SimMotion must target only the followed body (virtual parents are supported)"))
    length(first(system.motions).center) == 2 || throw(ArgumentError("attached refinement requires 2D motion"))
    return (;marker,motion=system)
end

function identity_resolution_pose()
    r,z,v = one(SMatrix{2,2,Float64,4}),zero(SMatrix{2,2,Float64,4}),zero(SVector{2,Float64})
    return MotionPose(r,v,z,v,z,v)
end

make_resolution_attachment(::Nothing,args...;kwargs...) = nothing
function make_resolution_attachment(config,geometry,host,constants;time)
    selected = findall(==(config.marker),host.GroupMarker)
    selected = selected[sortperm(host.ID[selected])]
    slots = zeros(Int32,length(host))
    slots[selected] = Int32.(eachindex(selected))
    # MotionDetails has already been translated into an absolute trajectory.
    plain_geometry = [SPHGeometry{2,typeof(constants.dx)}(;Particles=g.Particles,
        Type=g.Type,GroupMarker=g.GroupMarker,Floating=g.Floating) for g in geometry]
    motion = prepare_prescribed_motion(config.motion,plain_geometry,host)
    floating = motion === nothing ? FloatingArrays(geometry,host,constants;position_type=Float64,time) : nothing
    motion_slot = motion === nothing ? 0 : findfirst(n -> n.group_marker == config.marker,config.motion.motions)
    # Match FloatingArrays' summation order even with user-supplied IDs.
    center = sum(host.Position[sort(selected)])/length(selected)
    return (;slots=CuArray(slots),scratch=CUDA.zeros(Int32,length(host)),indices=CuArray(Int32.(selected)),
        reference=CuArray(host.Position[selected]),ghosts=CuArray(host.GhostPoints[selected]),
        normals=CuArray(host.GhostNormals[selected]),motion,motion_slot=Int32(motion_slot),floating,
        center_reference=center,pose=CuArray([identity_resolution_pose()]))
end

predictor_attachment(::Nothing) = nothing
predictor_attachment(a) = merge(a,(;slots=copy(a.slots),scratch=similar(a.scratch),indices=copy(a.indices)))

function resolution_body_indices_kernel!(slots,sorted,indices,starts,step)
    GPUStepState.step_active(step) || return nothing
    i = thread_index()
    @inbounds if i <= starts[end]
        slot = sorted[i]
        slots[i] = slot
        slot == 0 || (indices[slot] = i)
    elseif i <= step.count[1]
        slots[i] = 0
    end
    return nothing
end

function resolution_body_predictor_indices_kernel!(slots,indices,source,step,n)
    GPUStepState.step_active(step) || return nothing
    i = thread_index()
    @inbounds if i <= GPUStepState.step_particle_count(step,n)
        slot = source[i]
        slots[i] = slot
        slot == 0 || (indices[slot] = i)
    end
    return nothing
end

function resolution_motion_pose_kernel!(pose,motion,slot,step,fraction)
    GPUStepState.step_active(step) || return nothing
    sample_nodes!(motion.nodes,motion.poses,Float64(GPUStepState.step_time(step)+fraction*GPUStepState.step_dt(step)),0,1)
    @inbounds pose[1] = motion.poses[slot]
    return nothing
end

function resolution_floating_pose_kernel!(poses,center,velocity,omega,angle,turn,reference,step,half)
    GPUStepState.step_active(step) || return nothing
    @inbounds begin
        theta = Float64(angle[1]) + (half ? Float64(turn[1]) : 0.0)
        s,c = sincos(theta)
        r = SMatrix{2,2,Float64,4}(c,s,-s,c)
        dr = Float64(omega[1])*SMatrix{2,2,Float64,4}(-s,c,-c,-s)
        z,v = zero(r),zero(reference)
        poses[1] = MotionPose(r,SVector{2,Float64}(center[1])-r*reference,
            dr,SVector{2,Float64}(velocity[1])-dr*reference,z,v)
    end
    return nothing
end

function sample_resolution_body!(a,step,fraction)
    if a.motion !== nothing
        @cuda threads=1 resolution_motion_pose_kernel!(a.pose,a.motion,a.motion_slot,step,Float64(fraction))
    else
        f = a.floating
        half = fraction == 0.5
        @cuda threads=1 resolution_floating_pose_kernel!(a.pose,half ? f.center_half : f.center,
            half ? f.velocity_half : f.velocity,half ? f.omega_half : f.omega,f.angle,f.turn,
            a.center_reference,step,half)
    end
    return nothing
end

function resolution_body_particles_kernel!(x,v,ghosts,normals,indices,reference,reference_ghosts,reference_normals,poses,step,n)
    GPUStepState.step_active(step) || return nothing
    slot = thread_index()
    @inbounds if slot <= n
        i,pose = indices[slot],poses[1]
        x[i] = motion_position(pose,reference[slot])
        v[i] = motion_velocity(pose,reference[slot])
        if !iszero(reference_ghosts[slot])
            ghosts[i] = motion_position(pose,reference_ghosts[slot])
            normals[i] = pose.rotation*reference_normals[slot]
        end
    end
    return nothing
end

function place_resolution_body!(domain,step)
    a,p = domain.attachment,domain.particles
    n = length(a.indices)
    @cuda threads=128 blocks=cld(n,128) resolution_body_particles_kernel!(p.Position,p.Velocity,p.GhostPoints,p.GhostNormals,
        a.indices,a.reference,a.ghosts,a.normals,a.pose,step,Int32(n))
    return nothing
end

function resolution_floating_dt_kernel!(f,reference,limits,step,spacing,gravity)
    GPUStepState.step_active(step) || return nothing
    @inbounds begin
        GPUStepState.step_time(step) < f.pause[1] && return nothing
        lo,hi = limits.fine_outer_lo,limits.fine_outer_hi
        radius = norm(max.(abs.(lo-reference),abs.(hi-reference)))
        speed = norm(f.velocity[1])+abs(f.omega[1])*radius
        acceleration = norm(SVector(f.force[1],f.force[2])/f.mass[1]+SVector(0.0,-gravity)) +
            (abs(f.force[3]/f.inertia[1])+f.omega[1]^2)*radius
        distance = 0.25spacing
        denominator = speed+sqrt(speed^2+2acceleration*distance)
        if !isfinite(denominator)
            step.i[1] = RESOLUTION_MOTION_ERROR
        elseif denominator > 0
            step.f[1] = min(step.f[1],2distance/denominator)
        end
    end
    return nothing
end

function update_resolution_floating!(domain,step,final;limits=nothing)
    a,p = domain.attachment,domain.particles
    (a === nothing || a.floating === nothing) && return nothing
    f = merge(a.floating,(;indices=a.indices))
    launch_floating_forces!(f,p.Acceleration,p.Position,p.Type,p.GroupMarker,
        final ? f.center_half : f.center,domain.constants.m₀,step)
    if !final
        device = (;f.pause,f.velocity,f.omega,f.force,f.mass,f.inertia)
        @cuda threads=1 resolution_floating_dt_kernel!(device,a.center_reference,limits.reference,step,
            Float64(domain.constants.dx),Float64(domain.constants.g))
    end
    launch_floating_update!(f,ResolutionArithmeticStep{eltype(p.Density)}(step),domain.constants.g,final)
    return nothing
end

@inline function resolution_frame_travel(a,b,limits)
    lo,hi = limits.fine_outer_lo,limits.fine_outer_hi
    travel = 0.0
    for x in (SVector(lo[1],lo[2]),SVector(lo[1],hi[2]),SVector(hi[1],lo[2]),SVector(hi[1],hi[2]))
        travel = max(travel,norm(motion_position(a,x)-motion_position(b,x)))
    end
    return travel
end

@inline function resolution_pose_travel_bound(pose,limits,dt)
    lo,hi = limits.fine_outer_lo,limits.fine_outer_hi
    travel = 0.0
    for x in (SVector(lo[1],lo[2]),SVector(lo[1],hi[2]),SVector(hi[1],lo[2]),SVector(hi[1],hi[2]))
        acceleration = pose.rotation_acceleration*x+pose.offset_acceleration
        travel = max(travel,dt*norm(motion_velocity(pose,x))+0.5dt^2*norm(acceleration))
    end
    return travel
end

function resolution_frame_kernel!(frames,pose,limits,step,max_travel,initial)
    GPUStepState.step_active(step) || return nothing
    @inbounds begin
        current = pose[1]
        if !all(isfinite,current.offset) || !all(isfinite,current.rotation) ||
                !all(isfinite,current.offset_rate) || !all(isfinite,current.rotation_rate) ||
                (!initial && resolution_frame_travel(current,frames[1],limits) > max_travel)
            step.i[1] = RESOLUTION_MOTION_ERROR
            return nothing
        end
        frames[2] = initial ? current : frames[1]
        frames[1] = current
    end
    return nothing
end

# Probe the entire rectangle, not only body particles: rotational travel grows
# with distance from the pivot. A discontinuous trajectory is rejected if
# shrinking the step cannot resolve it.
function resolution_motion_dt_kernel!(frames,motion,slot,limits,step,spacing)
    GPUStepState.step_active(step) || return nothing
    @inbounds begin
        start = frames[1]
        dt = step.f[1]
        for _ in 1:32
            sample_nodes!(motion.nodes,motion.poses,step.f[2]+dt,0,1)
            finish = motion.poses[slot]
            sample_nodes!(motion.nodes,motion.poses,step.f[2]+dt/2,0,1)
            half = motion.poses[slot]
            travel = max(resolution_frame_travel(start,half,limits),resolution_frame_travel(start,finish,limits),
                resolution_pose_travel_bound(start,limits,dt),resolution_pose_travel_bound(half,limits,dt),
                resolution_pose_travel_bound(finish,limits,dt))
            if isfinite(travel) && travel <= 0.25spacing
                step.f[1] = dt
                return nothing
            end
            dt /= 2
        end
        step.i[1] = RESOLUTION_MOTION_ERROR
    end
    return nothing
end

function initialize_resolution_attachment!(fine,limits,state)
    a = fine.attachment
    a === nothing && return nothing
    step = ResolutionStep(state,fine,0)
    sample_resolution_body!(a,step,0.0)
    place_resolution_body!(fine,step)
    @cuda threads=1 resolution_frame_kernel!(limits.frames,a.pose,limits.reference,step,Inf,true)
    return nothing
end
