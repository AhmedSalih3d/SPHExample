module PrescribedMotions

using Adapt
using CUDA
using StaticArrays
using ..MotionTrajectories
using ..MotionTrajectories: AbstractTrajectory, ConstantTrajectory, motion_value,
    checked_axis, rotation_matrix, cross_matrix
using ..SimulationGeometry: Moving
using ..GPUStepState: step_active, step_time, step_dt
using ..GPUKernels: store_pos_cell!
import ..PrescribedRotation: sample_rotation!, launch_rotation!, update_rotation_indices!
using ..PrescribedRotation: prepare_rotation, inverse_permutation_kernel!, rotation_indices_kernel!

export PrescribedMotion, MotionSystem, motion_state, motion_position, motion_velocity,
       motion_acceleration, load_translation_trajectory, load_rotation_trajectory

"""
    PrescribedMotion(id; group_marker = id, parent = nothing,
                     translation = nothing, rotation = nothing,
                     center = (0,0), axis = (0,0,1))

Absolute rigid motion of a `Moving` group in 2D or 3D. Translation is a vector
trajectory in metres; rotation is a scalar trajectory in radians. Dimension
comes from `center`. An arbitrary oriented axis is supported in 3D. The local
transform is `center + translation(t) + R(t)*(x_reference-center)`.

`parent` names another motion's ID. All zero-pose reference geometry and centers
share a common coordinate frame. The child's transform acts first, followed
by its parent's transform, recursively; translation directions and rotation
axes therefore move with the parent. Set `group_marker = nothing` for a virtual
parent that has no particles. A `MotionSystem` validates and orders the tree.
Pass it as `SimMotion` to `RunSimulation` (2D/3D DBC). Periodic cases allow
independent translations that leave the periodic coordinate unchanged;
rotations and parent hierarchies still require a nonperiodic case.
"""
struct PrescribedMotion{D,T,A}
    id::Int
    group_marker::Union{Nothing,Int}
    parent::Union{Nothing,Int}
    translation::T
    rotation::A
    center::SVector{D,Float64}
    axis::SVector{3,Float64}
end
Adapt.@adapt_structure PrescribedMotion

function PrescribedMotion(id::Integer; group_marker = id, parent = nothing,
        translation = nothing, rotation = nothing, center = (0,0), axis = (0,0,1))
    D = length(center)
    D in (2,3) || throw(DimensionMismatch("motion center needs two or three components"))
    c = SVector{D,Float64}(center)
    all(isfinite, c) || throw(ArgumentError("motion center must be finite"))
    tr = translation === nothing ? ConstantTrajectory(zero(c)) : translation
    rot = rotation === nothing ? ConstantTrajectory(0.0) : rotation
    tr isa AbstractTrajectory && rot isa AbstractTrajectory || throw(ArgumentError("motion requires trajectory objects"))
    typeof(first(trajectory_state(tr, 0.0))) == typeof(c) || throw(DimensionMismatch("translation dimension must match center"))
    first(trajectory_state(rot, 0.0)) isa Float64 || throw(ArgumentError("rotation trajectory must be scalar"))
    marker = group_marker === nothing ? nothing : Int(group_marker)
    marker === nothing || marker > 0 || throw(ArgumentError("motion group markers must be positive"))
    return PrescribedMotion(Int(id), marker, parent === nothing ? nothing : Int(parent), tr, rot, c, checked_axis(axis, D))
end

"""
    MotionSystem(motions)

Validate unique IDs/group markers, dimensions, parent references and cycles,
then order parents before children. Input order does not change semantics.
"""
struct MotionSystem{M}
    motions::M
end

function MotionSystem(motions::Union{Tuple,AbstractVector})
    nodes = collect(motions)
    !isempty(nodes) && all(n -> n isa PrescribedMotion, nodes) || throw(ArgumentError("a motion system needs PrescribedMotion nodes"))
    ids = [n.id for n in nodes]
    length(unique(ids)) == length(ids) || throw(ArgumentError("motion IDs must be unique"))
    markers = [n.group_marker for n in nodes if n.group_marker !== nothing]
    length(unique(markers)) == length(markers) || throw(ArgumentError("each particle group may have only one motion"))
    D = length(first(nodes).center)
    all(n -> length(n.center) == D, nodes) || throw(DimensionMismatch("all motions must have the same dimension"))
    all(n -> n.parent === nothing || n.parent in ids, nodes) || throw(ArgumentError("motion parent ID is absent"))
    ordered, remaining = eltype(nodes)[], copy(nodes)
    while !isempty(remaining)
        i = findfirst(n -> n.parent === nothing || any(p -> p.id == n.parent, ordered), remaining)
        i === nothing && throw(ArgumentError("motion hierarchy contains a cycle"))
        push!(ordered, popat!(remaining, i))
    end
    return MotionSystem{typeof(Tuple(ordered))}(Tuple(ordered))
end
MotionSystem(motion::PrescribedMotion) = MotionSystem((motion,))

# Absolute poses retain their original reference positions. A transverse
# translation never crosses a periodic face, so wrapping cannot invalidate
# that reference. Reject other periodic motions until they have that support.
function validate_periodic_motion(system, periodicity)
    (system === nothing || periodicity === nothing) && return nothing
    nodes = system isa PrescribedMotion ? (system,) : system.motions
    for node in nodes
        node.parent === nothing && node.rotation isa ConstantTrajectory && iszero(node.rotation.value) &&
            zero_axial_translation(node.translation, periodicity.axis) ||
            throw(ArgumentError("Periodic SimMotion requires independent translations with zero displacement along the periodic axis; rotations and parent hierarchies are unsupported"))
    end
    return nothing
end

zero_axial_translation(trajectory, axis) = false
zero_axial_translation(h::ConstantTrajectory, axis) = iszero(h.value[axis])
zero_axial_translation(h::SampledTrajectory, axis) =
    all(x -> iszero(x[axis]), h.values) && all(v -> iszero(v[axis]), h.velocities)
zero_axial_translation(h::MotionSequence, axis) =
    all(x -> iszero(x[axis]), h.offsets) &&
    all(s -> iszero(s.velocity[axis]) && iszero(s.acceleration[axis]) && iszero(s.amplitude[axis]), h.segments)
zero_axial_translation(h::TrajectoryWindow, axis) =
    iszero(h.finish_offset[axis]) && zero_axial_translation(h.trajectory, axis)

# An affine transform and its exact first two time derivatives. Keeping the
# derivatives avoids frame ambiguities when composing angular/linear motion.
struct MotionPose{D,L}
    rotation::SMatrix{D,D,Float64,L}
    offset::SVector{D,Float64}
    rotation_rate::SMatrix{D,D,Float64,L}
    offset_rate::SVector{D,Float64}
    rotation_acceleration::SMatrix{D,D,Float64,L}
    offset_acceleration::SVector{D,Float64}
end

@inline motion_state(motion::PrescribedMotion{D}, time) where {D} = local_motion_state(motion, time, Val(D))

@inline function local_motion_state(motion, time, ::Val{D}) where {D}
    x, v, a = trajectory_state(motion.translation, time)
    angle, omega, alpha = trajectory_state(motion.rotation, time)
    r = rotation_matrix(motion.axis, angle, Val(D))
    k = cross_matrix(motion.axis, Val(D))
    dr = omega*(k*r)
    ddr = alpha*(k*r) + omega*omega*(k*(k*r))
    c = motion.center
    return MotionPose(r, c + x - r*c, dr, v - dr*c, ddr, a - ddr*c)
end

@inline function compose_pose(parent::MotionPose, child::MotionPose)
    p, c = parent, child
    return MotionPose(p.rotation*c.rotation, p.rotation*c.offset + p.offset,
        p.rotation_rate*c.rotation + p.rotation*c.rotation_rate,
        p.rotation_rate*c.offset + p.rotation*c.offset_rate + p.offset_rate,
        p.rotation_acceleration*c.rotation + 2p.rotation_rate*c.rotation_rate + p.rotation*c.rotation_acceleration,
        p.rotation_acceleration*c.offset + 2p.rotation_rate*c.offset_rate + p.rotation*c.offset_acceleration + p.offset_acceleration)
end

"""
    motion_state(system, id, time)

World affine pose and its first two derivatives, including every parent.
`motion_state(node, time)` instead samples just that node's local transform.
"""
function motion_state(system::MotionSystem, id::Integer, time)
    poses = MotionPose[]
    for node in system.motions
        pose = motion_state(node, time)
        if node.parent !== nothing
            p = findfirst(n -> n.id == node.parent, system.motions)
            pose = compose_pose(poses[p], pose)
        end
        push!(poses, pose)
        node.id == id && return pose
    end
    throw(ArgumentError("motion ID $id is absent"))
end

"""
    motion_position(motion, reference, time)
    motion_position(system, id, reference, time)

Place a point from the zero-pose reference geometry. A single motion samples
its local transform; a system includes the complete parent hierarchy. Suitable
for a measurement probe's host `trajectory(time)` callback.
"""
@inline motion_position(pose::MotionPose, reference) = pose.rotation*reference + pose.offset

"""
    motion_velocity(motion, reference, time)
    motion_velocity(system, id, reference, time)

Exact first derivative of `motion_position`, including angular wall velocity
and parent-frame motion. `reference` remains the point's zero-pose position.
"""
@inline motion_velocity(pose::MotionPose, reference) = pose.rotation_rate*reference + pose.offset_rate

"""
    motion_acceleration(motion, reference, time)
    motion_acceleration(system, id, reference, time)

Exact second derivative of `motion_position` on each smooth trajectory
interval, including rotational and parent-frame cross terms. Impulses at
discontinuous interval rates are not represented.
"""
@inline motion_acceleration(pose::MotionPose, reference) = pose.rotation_acceleration*reference + pose.offset_acceleration
for f in (:motion_position, :motion_velocity, :motion_acceleration)
    @eval begin
        $f(motion::PrescribedMotion, reference, time) = $f(motion_state(motion, time), motion_value(reference))
        $f(system::MotionSystem, id::Integer, reference, time) = $f(motion_state(system, id, time), motion_value(reference))
    end
end

include("PrescribedMotionFiles.jl")
include("PrescribedMotionGPU.jl")

end
