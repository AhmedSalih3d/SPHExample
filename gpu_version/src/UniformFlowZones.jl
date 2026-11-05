"""
    UniformFlowZone(lower, upper; direction, times, speeds,
        axis = 1, coefdt = 1000, psi = 0.7, beta = 3)

Axis-aligned uniform-flow relaxation zone, passed through `SimDamping`.
After each accepted step, blend fluid velocity towards `speed(t) * direction`.
Speed is linearly interpolated and held at the table endpoints. The normalized
double-tanh profile is one at the box centre and zero at its two `axis` faces.
Other faces strictly bound the box. `coefdt = 0` applies the profile directly;
otherwise the blend is capped at one after multiplication by `coefdt * dt`.
All histories are stored in immutable tuples for CUDA graph execution.
"""
struct UniformFlowZone{D,N} <: AbstractDampingZone{D}
    lower::SVector{D,Float64}
    upper::SVector{D,Float64}
    direction::SVector{D,Float64}
    times::NTuple{N,Float64}
    speeds::NTuple{N,Float64}
    axis::Int
    coefdt::Float64
    psi::Float64
    beta::Float64
    subtract::Float64
    divisor::Float64
end

function UniformFlowZone(lower, upper; direction, times, speeds,
        axis = 1, coefdt = 1000, psi = 0.7, beta = 3)
    D = length(lower)
    D in (2, 3) || throw(ArgumentError("uniform flow requires 2D or 3D"))
    lo, hi = damping_box_bounds((lower, upper), Val(D), "flow box")
    all(lo .< hi) || throw(ArgumentError("flow box needs positive extents"))
    length(direction) == D || throw(DimensionMismatch("direction requires $D components"))
    vector = SVector{D,Float64}(direction)
    all(isfinite, vector) && isfinite(norm(vector)) && norm(vector) > 0 ||
        throw(ArgumentError("direction must be finite and nonzero"))
    1 <= axis <= D || throw(ArgumentError("axis must be between 1 and $D"))
    ts, vs = Tuple(Float64.(times)), Tuple(Float64.(speeds))
    length(ts) >= 2 && length(ts) == length(vs) ||
        throw(ArgumentError("provide at least two matching time and speed samples"))
    all(isfinite, ts) && first(ts) >= 0 && all(>(0), diff(collect(ts))) &&
        all(isfinite, vs) || throw(ArgumentError("invalid uniform-flow history"))
    all(isfinite, (coefdt, psi, beta)) && coefdt >= 0 && psi > 0 && beta > 0 ||
        throw(ArgumentError("coefdt must be nonnegative; psi and beta must be positive"))
    subtract = tanh((1 + psi) * beta) - tanh((1 - psi) * beta)
    divisor = 2tanh(psi * beta) - subtract
    isfinite(divisor) && divisor > eps(Float64) ||
        throw(ArgumentError("degenerate relaxation profile"))
    return UniformFlowZone(lo, hi, vector / norm(vector), ts, vs, Int(axis),
        Float64(coefdt), Float64(psi), Float64(beta), subtract, divisor)
end

@inline function normalized_damping(zone::UniformFlowZone{D}, position) where D
    for d in 1:D
        zone.lower[d] < position[d] < zone.upper[d] || return 0.0
    end
    a = zone.axis
    distance = (2position[a] - zone.lower[a] - zone.upper[a]) /
        (zone.upper[a] - zone.lower[a])
    return clamp((tanh((distance + zone.psi) * zone.beta) -
        tanh((distance - zone.psi) * zone.beta) - zone.subtract) / zone.divisor, 0.0, 1.0)
end

@inline function uniform_flow_speed(zone::UniformFlowZone{D,N}, time) where {D,N}
    time <= zone.times[1] && return zone.speeds[1]
    for k in 2:N
        if time <= zone.times[k]
            fraction = (time - zone.times[k-1]) / (zone.times[k] - zone.times[k-1])
            return zone.speeds[k-1] + fraction * (zone.speeds[k] - zone.speeds[k-1])
        end
    end
    return zone.speeds[N]
end

@inline function zone_velocity(zone::UniformFlowZone, position, velocity::SVector{D,T}, dt, time) where {D,T}
    profile = normalized_damping(zone, position)
    profile == 0 && return velocity
    blend = zone.coefdt == 0 ? profile : min(1.0, profile * zone.coefdt * dt)
    target = uniform_flow_speed(zone, time) * zone.direction
    return SVector{D,T}(blend * target + (1 - blend) * velocity)
end
