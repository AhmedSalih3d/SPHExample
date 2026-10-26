module PrescribedRotation

using CUDA
using StaticArrays
using ..SimulationGeometry: Moving
using ..GPUStepState: step_active, step_time, step_dt
using ..GPUKernels: store_pos_cell!

export RotationMotion, rotation_state, rotation_position

"""
    RotationMotion(group_marker; times, angles, center = (0, 0))

Prescribe absolute 2D rotation of a `Moving` particle group. Angles are in
radians, positive counterclockwise, about a fixed world-coordinate center.
Input particle positions describe the zero-angle geometry. Angles interpolate
linearly; angular velocity is the slope of the same interval. Outside the
table the endpoint angle is held and angular velocity is zero. Times must be
finite, nonnegative and strictly increasing, with at least two samples.

Pass as `SimRotation` to `RunSimulation`. Currently supports nonperiodic 2D
DBC cases. The selected group cannot also have `MotionDetails`. Device motion
uses stable reference slots and absolute reference positions, so sorting and
long runs do not accumulate incremental rotation drift.
"""
struct RotationMotion
    group_marker::Int
    times::Vector{Float64}
    angles::Vector{Float64}
    center::SVector{2,Float64}
end

function RotationMotion(group_marker::Integer; times, angles, center = (0, 0))
    ts, theta = Float64.(collect(times)), Float64.(collect(angles))
    length(ts) == length(theta) || throw(DimensionMismatch("rotation angles must match times"))
    length(ts) >= 2 && all(isfinite, ts) && first(ts) >= 0 && all(>(0), diff(ts)) ||
        throw(ArgumentError("rotation requires at least two finite, nonnegative, strictly increasing times"))
    all(isfinite, theta) || throw(ArgumentError("rotation angles must be finite"))
    all(isfinite, diff(theta) ./ diff(ts)) || throw(ArgumentError("rotation interval velocities must be finite"))
    length(center) == 2 || throw(DimensionMismatch("rotation center requires two components"))
    c = SVector{2,Float64}(center)
    all(isfinite, c) || throw(ArgumentError("rotation center must be finite"))
    return RotationMotion(Int(group_marker), ts, theta, c)
end

"""
    rotation_state(motion, time) -> (angle, angular_velocity)

Sample the prescribed angle and its piecewise constant derivative. At the
last endpoint the motion stops. The same sampler is used on host and device.
"""
@inline function rotation_state(motion, time)
    times, angles = motion.times, motion.angles
    @inbounds begin
        time < times[1] && return (angles[1], 0.0)
        time >= times[end] && return (angles[end], 0.0)
        lo, hi = 1, length(times) - 1
        while lo < hi
            mid = (lo + hi + 1) >>> 1
            if times[mid] <= time
                lo = mid
            else
                hi = mid - 1
            end
        end
        omega = (angles[lo+1] - angles[lo]) / (times[lo+1] - times[lo])
        return (angles[lo] + omega * (time - times[lo]), omega)
    end
end

@inline function rotated_position(reference, center, angle)
    s, c = sincos(angle)
    r = reference - center
    return center + SVector(c*r[1] - s*r[2], s*r[1] + c*r[2])
end

"""
    rotation_position(motion, reference_position, time)

Position of a point attached to the zero-angle geometry. Suitable for moving
pressure/velocity probes via `MeasurementProbe(...; trajectory = t -> ...)`.
"""
@inline function rotation_position(motion, reference_position, time)
    angle, _ = rotation_state(motion, time)
    return rotated_position(SVector{2,Float64}(reference_position), motion.center, angle)
end

prepare_rotation(::Nothing, geometry, particles) = nothing
function prepare_rotation(motion::RotationMotion, geometry, particles)
    all(p -> length(p) == 2, particles.Position) || throw(ArgumentError("prescribed rotation requires 2D particles"))
    selected = findall(==(motion.group_marker), particles.GroupMarker)
    isempty(selected) && throw(ArgumentError("rotation GroupMarker $(motion.group_marker) is absent"))
    all(==(Moving), particles.Type[selected]) || throw(ArgumentError("rotation requires an exclusively Moving group"))
    all(g -> g.GroupMarker != motion.group_marker || g.Motion === nothing, geometry) ||
        throw(ArgumentError("rotation and MotionDetails cannot prescribe the same group"))
    order = sortperm(particles.ID[selected])
    selected = selected[order]
    return (; group_marker = motion.group_marker, center = motion.center,
        times = CuArray(motion.times), angles = CuArray(motion.angles),
        indices = CuArray(Int32.(selected)),
        coefficients = CUDA.zeros(SVector{3,Float64}, 1),
        reference = CuArray(SVector{2,Float64}.(particles.Position[selected])))
end

function inverse_permutation_kernel!(inverse, permutation, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if i <= n
        inverse[permutation[i]] = i
    end
    return nothing
end

function rotation_indices_kernel!(indices, inverse, n)
    slot = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if slot <= n
        indices[slot] = inverse[indices[slot]]
    end
    return nothing
end

update_rotation_indices!(::Nothing, workspace) = nothing
function update_rotation_indices!(motion, workspace)
    # Sorting has consumed cell IDs; borrow that scratch before compaction uses it.
    n = length(workspace.Perm)
    @cuda threads=256 blocks=cld(n, 256) inverse_permutation_kernel!(
        workspace.CellIDScratch, workspace.Perm, n)
    m = length(motion.indices)
    @cuda threads=256 blocks=cld(m, 256) rotation_indices_kernel!(
        motion.indices, workspace.CellIDScratch, m)
    return nothing
end

function rotation_coefficients_kernel!(motion, step, fraction)
    step_active(step) || return nothing
    time = Float64(step_time(step) + oftype(step_dt(step), fraction) * step_dt(step))
    angle, omega = rotation_state(motion, time)
    s, c = sincos(angle)
    @inbounds motion.coefficients[1] = SVector(s, c, omega)
    return nothing
end

function rotation_kernel!(position, velocity, types, groups, motion, step,
                          pos_cells, cell_ids, grid, H, n)
    step_active(step) || return nothing
    slot = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if slot <= n
        i = motion.indices[slot]
        if types[i] == Moving && groups[i] == motion.group_marker
            s, c, omega = motion.coefficients[1]
            r = motion.reference[slot] - motion.center
            x = motion.center + SVector(c*r[1] - s*r[2], s*r[1] + c*r[2])
            r = x - motion.center
            position[i] = x
            velocity[i] = SVector(-omega*r[2], omega*r[1])
            store_pos_cell!(pos_cells, i, position[i], cell_ids, grid, H)
        end
    end
    return nothing
end

launch_rotation!(::Nothing, args...; kwargs...) = nothing
function launch_rotation!(motion, position, velocity, types, groups, ids, step;
        fraction = 0.0, pos_cells = nothing, cell_ids = nothing, grid = nothing, H = nothing)
    n = length(motion.indices)
    n == 0 && return nothing
    @cuda threads=1 blocks=1 rotation_coefficients_kernel!(motion, step, Float64(fraction))
    @cuda threads=256 blocks=cld(n, 256) rotation_kernel!(position, velocity, types, groups,
        motion, step, pos_cells, cell_ids, grid, H, n)
    return nothing
end

end
