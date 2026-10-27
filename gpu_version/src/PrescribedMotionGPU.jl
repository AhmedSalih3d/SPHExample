# Host-only optional IDs stay out of kernel arguments; Union{Nothing,Int}
# fields are not bitstypes even when their current value is an integer.
struct DeviceMotion{D,T,A}
    translation::T
    rotation::A
    center::SVector{D,Float64}
    axis::SVector{3,Float64}
end
Adapt.@adapt_structure DeviceMotion
@inline motion_state(motion::DeviceMotion{D}, time) where {D} = local_motion_state(motion, time, Val(D))

device_trajectory(h::ConstantTrajectory) = h
device_trajectory(h::SampledTrajectory) = SampledTrajectory(CuArray(h.times), CuArray(h.values), CuArray(h.velocities), h.cubic)
device_trajectory(h::MotionSequence) = MotionSequence(CuArray(h.times), CuArray(h.segments), CuArray(h.offsets))
device_trajectory(h::CircularTrajectory) = CircularTrajectory(device_trajectory(h.angles), h.reference, h.center, h.axis)
device_trajectory(h::TrajectoryWindow) = TrajectoryWindow(device_trajectory(h.trajectory), h.start_time, h.end_time, h.finish_offset)

struct PreparedMotionSystem{N,I,V,P,G,O}
    nodes::N
    indices::I
    reference::V
    poses::P
    markers::G
    owners::O
end
Adapt.@adapt_structure PreparedMotionSystem

prepare_prescribed_motion(::Nothing, geometry, particles) = nothing
prepare_prescribed_motion(motion::PrescribedMotion, geometry, particles) = prepare_prescribed_motion(MotionSystem(motion), geometry, particles)
function prepare_prescribed_motion(system::MotionSystem, geometry, particles)
    D = length(first(system.motions).center)
    all(p -> length(p) == D, particles.Position) || throw(DimensionMismatch("prescribed motion dimension differs from particles"))
    selected, owners, markers = Int[], Int32[], Int[]
    for (slot, node) in enumerate(system.motions)
        marker = node.group_marker
        marker === nothing && continue
        indices = findall(==(marker), particles.GroupMarker)
        isempty(indices) && throw(ArgumentError("motion GroupMarker $marker is absent"))
        all(==(Moving), particles.Type[indices]) || throw(ArgumentError("prescribed motion requires exclusively Moving groups"))
        all(g -> g.GroupMarker != marker || g.Motion === nothing, geometry) ||
            throw(ArgumentError("prescribed motion and MotionDetails cannot prescribe the same group"))
        indices = indices[sortperm(particles.ID[indices])]
        append!(selected, indices)
        append!(owners, fill(Int32(slot), length(indices)))
        append!(markers, fill(marker, length(indices)))
    end
    isempty(selected) && throw(ArgumentError("motion system has no particle groups"))
    nodes = map(system.motions) do node
        parent = node.parent === nothing ? 0 : findfirst(n -> n.id == node.parent, system.motions)
        local_node = DeviceMotion(device_trajectory(node.translation), device_trajectory(node.rotation), node.center, node.axis)
        (; motion = local_node, parent = Int32(parent))
    end
    initial_pose = motion_state(first(system.motions), 0.0)
    return PreparedMotionSystem(nodes, CuArray(Int32.(selected)),
        CuArray(SVector{D,Float64}.(particles.Position[selected])),
        CuArray(fill(initial_pose, 3length(nodes))), CuArray(markers), CuArray(owners))
end

@inline sample_nodes!(::Tuple{}, poses, time, offset, slot) = nothing
@inline function sample_nodes!(nodes::Tuple, poses, time, offset, slot)
    node = first(nodes)
    pose = motion_state(node.motion, time)
    @inbounds begin
        if node.parent != 0
            pose = compose_pose(poses[offset + node.parent], pose)
        end
        poses[offset + slot] = pose
    end
    sample_nodes!(Base.tail(nodes), poses, time, offset, slot + 1)
    return nothing
end

@inline function sample_rotation!(motion::PreparedMotionSystem, step, fraction, slot)
    time = Float64(step_time(step) + oftype(step_dt(step), fraction)*step_dt(step))
    sample_nodes!(motion.nodes, motion.poses, time, (slot-1)*length(motion.nodes), 1)
    return nothing
end

function motion_coefficients_kernel!(motion, step, fraction)
    step_active(step) || return nothing
    sample_rotation!(motion, step, fraction, 1)
    return nothing
end

function prescribed_motion_kernel!(position, velocity, types, groups, motion, step,
        pos_cells, cell_ids, grid, H, coefficient_slot, n)
    step_active(step) || return nothing
    slot = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    @inbounds if slot <= n
        i = motion.indices[slot]
        if types[i] == Moving && groups[i] == motion.markers[slot]
            pose = motion.poses[(coefficient_slot-1)*length(motion.nodes) + motion.owners[slot]]
            x = motion.reference[slot]
            position[i] = motion_position(pose, x)
            velocity[i] = motion_velocity(pose, x)
            store_pos_cell!(pos_cells, i, position[i], cell_ids, grid, H)
        end
    end
    return nothing
end

function launch_rotation!(motion::PreparedMotionSystem, position, velocity, types, groups, ids, step;
        fraction = 0.0, pos_cells = nothing, cell_ids = nothing, grid = nothing, H = nothing,
        prepared::Bool = false, coefficient_slot::Int = 1)
    n = length(motion.indices)
    if !prepared
        @cuda threads=1 blocks=1 motion_coefficients_kernel!(motion, step, Float64(fraction))
    end
    @cuda threads=256 blocks=cld(n,256) prescribed_motion_kernel!(position, velocity, types, groups,
        motion, step, pos_cells, cell_ids, grid, H, prepared ? coefficient_slot : 1, n)
    return nothing
end

function update_rotation_indices!(motion::PreparedMotionSystem, workspace)
    n = length(workspace.Perm)
    @cuda threads=256 blocks=cld(n,256) inverse_permutation_kernel!(workspace.CellIDScratch, workspace.Perm, n)
    m = length(motion.indices)
    @cuda threads=256 blocks=cld(m,256) rotation_indices_kernel!(motion.indices, workspace.CellIDScratch, m)
    return nothing
end

# Reuse the established solver motion lifecycle, including sorting, stage
# sampling and capture, without changing RotationMotion's representation.
@inline function sample_rotation!(motions::Tuple, step, fraction, slot)
    foreach(m -> sample_rotation!(m, step, fraction, slot), motions)
    return nothing
end
function launch_rotation!(motions::Tuple, args...; kwargs...)
    foreach(m -> launch_rotation!(m, args...; kwargs...), motions)
    return nothing
end
function update_rotation_indices!(motions::Tuple, workspace)
    foreach(m -> update_rotation_indices!(m, workspace), motions)
    return nothing
end

function combine_motions(legacy, system, geometry, particles)
    system = system isa PrescribedMotion ? MotionSystem(system) : system
    if legacy !== nothing && system !== nothing
        any(n -> n.group_marker == legacy.group_marker, system.motions) &&
            throw(ArgumentError("SimRotation and SimMotion cannot prescribe the same group"))
    end
    a = prepare_rotation(legacy, geometry, particles)
    b = prepare_prescribed_motion(system, geometry, particles)
    return a === nothing ? b : b === nothing ? a : (a,b)
end
