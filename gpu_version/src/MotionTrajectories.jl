module MotionTrajectories

using Adapt
using StaticArrays

export SampledTrajectory, KinematicSegment, SinusoidalSegment, MotionSequence, TrajectoryWindow,
       CircularTrajectory, trajectory_state

abstract type AbstractTrajectory end

motion_value(x::Real) = Float64(x)
motion_value(x) = SVector{length(x),Float64}(x)

function checked_times(times)
    ts = Float64.(collect(times))
    length(ts) >= 2 && all(isfinite, ts) && first(ts) >= 0 && all(>(0), diff(ts)) ||
        throw(ArgumentError("motion requires at least two finite, nonnegative, strictly increasing times"))
    return ts
end

"""
    SampledTrajectory(times, values; velocities = nothing)

Scalar angles (radians) or vector displacements (metres), sampled in seconds.
Linear interpolation is the default. Supplying a velocity at every keyframe
selects cubic Hermite interpolation, including its exact first two derivatives.
Intervals are right-continuous. Before the first time and at/after the last
time, hold the endpoint value with zero velocity and acceleration. No angle
wrapping is applied. Tables retain Float64 precision on the host and device.
"""
struct SampledTrajectory{T,V} <: AbstractTrajectory
    times::T
    values::V
    velocities::V
    cubic::Bool
end
Adapt.@adapt_structure SampledTrajectory

function SampledTrajectory(times, values; velocities = nothing)
    ts = checked_times(times)
    vs = motion_value.(collect(values))
    length(vs) == length(ts) || throw(DimensionMismatch("motion values must match times"))
    all(x -> typeof(x) == typeof(first(vs)) && all(isfinite, x), vs) ||
        throw(ArgumentError("motion values must be finite and have consistent dimensions"))
    slopes = velocities === nothing ? zero.(vs) : motion_value.(collect(velocities))
    length(slopes) == length(ts) || throw(DimensionMismatch("motion velocities must match times"))
    all(x -> typeof(x) == typeof(first(vs)) && all(isfinite, x), slopes) ||
        throw(ArgumentError("motion velocities must be finite and match the value dimensions"))
    all(x -> all(isfinite, x), diff(vs) ./ diff(ts)) || throw(ArgumentError("motion interval rates must be finite"))
    if velocities !== nothing
        for i in 1:(length(ts)-1)
            dt = ts[i+1] - ts[i]
            a = (3*(vs[i+1]-vs[i])/dt - 2slopes[i] - slopes[i+1]) / dt
            b = (2*(vs[i]-vs[i+1])/dt + slopes[i] + slopes[i+1]) / (dt*dt)
            all(x -> all(isfinite, x), (a,b,2a+6dt*b)) ||
                throw(ArgumentError("motion cubic coefficients and accelerations must be finite"))
        end
    end
    return SampledTrajectory(ts, vs, slopes, velocities !== nothing)
end

# Shared binary search keeps host and device knot ownership identical.
@inline function interval_index(times, time)
    lo, hi = 1, length(times) - 1
    @inbounds while lo < hi
        mid = (lo + hi + 1) >>> 1
        if times[mid] <= time
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end

"""
    trajectory_state(trajectory, time) -> (value, velocity, acceleration)

Sample a scalar or vector trajectory and the derivatives of that same curve.
Acceleration is zero outside the active interval; impulses at discontinuous
velocity changes are not represented.
"""
@inline function trajectory_state(h::SampledTrajectory, time)
    @inbounds begin
        time < h.times[1] && return (h.values[1], zero(h.values[1]), zero(h.values[1]))
        time >= h.times[end] && return (h.values[end], zero(h.values[end]), zero(h.values[end]))
        i = interval_index(h.times, time)
        dt = h.times[i+1] - h.times[i]
        tau = time - h.times[i]
        p, q = h.values[i], h.values[i+1]
        if !h.cubic
            v = (q - p) / dt
            return (p + v*tau, v, zero(v))
        end
        v, w = h.velocities[i], h.velocities[i+1]
        a = (3*(q-p)/dt - 2v - w) / dt
        b = (2*(p-q)/dt + v + w) / (dt*dt)
        return (p + tau*(v + tau*(a + tau*b)), v + tau*(2a + 3tau*b), 2a + 6tau*b)
    end
end

struct ConstantTrajectory{V} <: AbstractTrajectory
    value::V
end
Adapt.@adapt_structure ConstantTrajectory
@inline trajectory_state(h::ConstantTrajectory, time) = (h.value, zero(h.value), zero(h.value))

struct TrajectorySegment{V}
    duration::Float64
    velocity::V
    acceleration::V
    amplitude::V
    frequency::V
    phase::V
end

function checked_segment(duration, velocity, acceleration, amplitude, frequency, phase)
    d = Float64(duration)
    isfinite(d) && d > 0 || throw(ArgumentError("segment duration must be finite and positive"))
    v, a, amp, f, p = motion_value.((velocity, acceleration, amplitude, frequency, phase))
    all(x -> typeof(x) == typeof(v) && all(isfinite, x), (a, amp, f, p)) && all(isfinite, v) ||
        throw(ArgumentError("segment coefficients must be finite with matching dimensions"))
    all(>=(0), f) || throw(ArgumentError("sinusoidal frequencies must be nonnegative"))
    all(isfinite, (2pi .* f) .* d + p) || throw(ArgumentError("sinusoidal phase overflows"))
    s = TrajectorySegment(d, v, a, amp, f, p)
    all(x -> all(isfinite, x), segment_state(s, d)) || throw(ArgumentError("segment endpoint overflows"))
    return s
end

"""
    KinematicSegment(duration; velocity, acceleration = zero(velocity))

Constant-acceleration scalar or vector segment. Each segment explicitly sets
its initial velocity; use the previous endpoint velocity to continue smoothly.
A zero-velocity, zero-acceleration segment is a pause.
"""
function KinematicSegment(duration; velocity, acceleration = zero(motion_value(velocity)))
    v = motion_value(velocity)
    return checked_segment(duration, v, acceleration, zero(v), zero(v), zero(v))
end

"""
    SinusoidalSegment(duration; amplitude, frequency, phase = zero(amplitude))

Displacement `amplitude * (sin(2pi*frequency*t + phase) - sin(phase))`.
Frequency is in Hz, phase in radians; vector components may differ. The phase
offset is subtracted so that every segment starts at zero displacement.
"""
function SinusoidalSegment(duration; amplitude, frequency, phase = zero(motion_value(amplitude)))
    amp = motion_value(amplitude)
    return checked_segment(duration, zero(amp), zero(amp), amp, frequency, phase)
end

@inline function segment_state(s::TrajectorySegment, time)
    w = 2pi .* s.frequency
    phase = w .* time + s.phase
    value = s.velocity*time + s.acceleration*(time*time/2) + s.amplitude .* (sin.(phase) - sin.(s.phase))
    velocity = s.velocity + s.acceleration*time + s.amplitude .* w .* cos.(phase)
    acceleration = s.acceleration - s.amplitude .* w .* w .* sin.(phase)
    return (value, velocity, acceleration)
end

"""
    MotionSequence(segments; start_time = 0, initial_value = zero, jumps = nothing)

Ordered kinematic/sinusoidal segments with continuous accumulated displacement.
The next segment owns a shared endpoint. Before `start_time`, hold
`initial_value`; at/after the final endpoint hold the accumulated value, with
zero rates. Scalar sequences describe angles; vector sequences translations.
Optional `jumps` gives one instantaneous displacement after each segment.
Jumps take effect at the shared endpoint; their impulses are not represented
as finite velocities or accelerations. Omit them for continuous trajectories.
"""
struct MotionSequence{T,S,V} <: AbstractTrajectory
    times::T
    segments::S
    offsets::V
end
Adapt.@adapt_structure MotionSequence

function MotionSequence(segments; start_time = 0, initial_value = nothing, jumps = nothing)
    ss = collect(segments)
    isempty(ss) && throw(ArgumentError("a motion sequence needs at least one segment"))
    all(s -> s isa TrajectorySegment && typeof(s) == typeof(first(ss)), ss) ||
        throw(ArgumentError("sequence segments must have matching dimensions"))
    t = Float64(start_time)
    isfinite(t) && t >= 0 || throw(ArgumentError("sequence start time must be finite and nonnegative"))
    p = initial_value === nothing ? zero(first(ss).velocity) : motion_value(initial_value)
    typeof(p) == typeof(first(ss).velocity) && all(isfinite, p) ||
        throw(ArgumentError("initial value must be finite and match the segments"))
    shifts = jumps === nothing ? fill(zero(p), length(ss)) : motion_value.(collect(jumps))
    length(shifts) == length(ss) || throw(DimensionMismatch("jumps must match sequence segments"))
    all(x -> typeof(x) == typeof(p) && all(isfinite, x), shifts) ||
        throw(ArgumentError("sequence jumps must be finite and match segment dimensions"))
    ts, offsets = [t], [p]
    for (s, shift) in zip(ss, shifts)
        t += s.duration
        p += first(segment_state(s, s.duration)) + shift
        isfinite(t) && t > last(ts) && all(isfinite, p) || throw(ArgumentError("sequence endpoint overflows"))
        push!(ts, t)
        push!(offsets, p)
    end
    return MotionSequence(ts, ss, offsets)
end

"""
    TrajectoryWindow(trajectory; start_time = 0, end_time = Inf,
                     finish_offset = zero)

Play a trajectory in local time starting at `start_time`, stopping at
`end_time`. Hold the initial/final value with zero rates outside the window.
An optional instantaneous `finish_offset` is applied strictly after the final
time, allowing a file-driven path to finish with a relocation. No finite
velocity or acceleration is assigned to that instantaneous displacement.
"""
struct TrajectoryWindow{H,V} <: AbstractTrajectory
    trajectory::H
    start_time::Float64
    end_time::Float64
    finish_offset::V
end
Adapt.@adapt_structure TrajectoryWindow

function TrajectoryWindow(trajectory::AbstractTrajectory; start_time = 0, end_time = Inf, finish_offset = nothing)
    start, finish = Float64(start_time), Float64(end_time)
    isfinite(start) && start >= 0 && finish > start && !isnan(finish) ||
        throw(ArgumentError("trajectory window needs a finite nonnegative start and a later end"))
    value = first(trajectory_state(trajectory, 0.0))
    offset = finish_offset === nothing ? zero(value) : motion_value(finish_offset)
    typeof(offset) == typeof(value) && all(isfinite, offset) ||
        throw(ArgumentError("finish offset must be finite and match the trajectory dimension"))
    return TrajectoryWindow(trajectory, start, finish, offset)
end

@inline function trajectory_state(h::TrajectoryWindow, time)
    if time < h.start_time
        value = first(trajectory_state(h.trajectory, 0.0))
        return (value, zero(value), zero(value))
    elseif time >= h.end_time
        value = first(trajectory_state(h.trajectory, h.end_time - h.start_time))
        time > h.end_time && (value += h.finish_offset)
        return (value, zero(value), zero(value))
    end
    return trajectory_state(h.trajectory, time - h.start_time)
end

@inline function trajectory_state(h::MotionSequence, time)
    @inbounds begin
        time < h.times[1] && return (h.offsets[1], zero(h.offsets[1]), zero(h.offsets[1]))
        time >= h.times[end] && return (h.offsets[end], zero(h.offsets[end]), zero(h.offsets[end]))
        i = interval_index(h.times, time)
        p, v, a = segment_state(h.segments[i], time - h.times[i])
        return (h.offsets[i] + p, v, a)
    end
end

@inline function rotation_matrix(axis, angle, ::Val{2})
    s, c = sincos(angle * axis[3])
    return SMatrix{2,2}(c, s, -s, c)
end
@inline function cross_matrix(axis, ::Val{2})
    z = axis[3]
    return SMatrix{2,2}(0.0, z, -z, 0.0)
end
@inline function cross_matrix(axis, ::Val{3})
    x, y, z = axis
    return SMatrix{3,3}(0.0, z, -y, -z, 0.0, x, y, -x, 0.0)
end
@inline function rotation_matrix(axis, angle, ::Val{3})
    s, c = sincos(angle)
    k = cross_matrix(axis, Val(3))
    return one(k) + s*k + (1-c)*(k*k)
end

function checked_axis(axis, D)
    length(axis) == 3 || throw(DimensionMismatch("rotation axis needs three components"))
    a = SVector{3,Float64}(axis)
    scale = maximum(abs, a)
    all(isfinite, a) && scale > 0 || throw(ArgumentError("rotation axis must be finite and nonzero"))
    a /= scale
    a /= sqrt(sum(abs2, a))
    D == 3 || (iszero(a[1]) && iszero(a[2])) || throw(ArgumentError("2D rotation requires the perpendicular axis (0,0,+/-1)"))
    return a
end

"""
    CircularTrajectory(angles; reference, center = zero(reference), axis = (0,0,1))

Translation `center + R(angle)*(reference-center) - reference`. All attached
points translate by that same vector, preserving the body's orientation.
Angles are a scalar trajectory in radians. Unlike rigid rotation, circular
translation does not spin the body. Reference/center use 2D or 3D coordinates.
"""
struct CircularTrajectory{D,A} <: AbstractTrajectory
    angles::A
    reference::SVector{D,Float64}
    center::SVector{D,Float64}
    axis::SVector{3,Float64}
end
Adapt.@adapt_structure CircularTrajectory

function CircularTrajectory(angles::AbstractTrajectory; reference, center = zero(motion_value(reference)), axis = (0,0,1))
    r, c = motion_value(reference), motion_value(center)
    D = length(r)
    D in (2,3) && typeof(r) == typeof(c) || throw(DimensionMismatch("circular reference and center need two or three matching components"))
    all(isfinite, r) && all(isfinite, c) || throw(ArgumentError("circular geometry must be finite"))
    first(trajectory_state(angles, 0.0)) isa Real || throw(ArgumentError("circular angles must be scalar"))
    return CircularTrajectory(angles, r, c, checked_axis(axis, D))
end

@inline function trajectory_state(h::CircularTrajectory{D}, time) where {D}
    angle, omega, alpha = trajectory_state(h.angles, time)
    r = rotation_matrix(h.axis, angle, Val(D)) * (h.reference - h.center)
    k = cross_matrix(h.axis, Val(D))
    return (h.center + r - h.reference, omega*(k*r), alpha*(k*r) + omega*omega*(k*(k*r)))
end

end
