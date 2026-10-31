module DampingZones

using CUDA
using StaticArrays
using LinearAlgebra: norm, dot
using ..SimulationGeometry: Fluid
using ..GPUStepState: step_active, step_dt, step_particle_count

export DampingZone, DampingBox, DampingCylinder

"""
    DampingZone(limit_min, limit_max; overlimit = 1, rate = 10, factors = ones(D))

Planar velocity absorption for fluid particles in 2D or 3D. The two points
define the direction and length of the zone. At distance `s` from its first
plane, each velocity component is multiplied by
`max(0, 1 - dt * rate * min(s / length, 1)^2 * factors[d])` after each accepted
step. Absorption extends `overlimit` beyond the second plane. Boundary,
floating and frozen particles are unchanged. Overlapping zones act in order.
Pass one zone or a tuple/vector as `SimDamping` to `RunSimulation`.
"""
abstract type AbstractDampingZone{D} end

struct DampingZone{D} <: AbstractDampingZone{D}
    origin::SVector{D,Float64}
    direction::SVector{D,Float64}
    length::Float64
    overlimit::Float64
    rate::Float64
    factors::SVector{D,Float64}
end

function damping_parameters(::Val{D}, overlimit, rate, factors) where D
    length(factors) == D || throw(DimensionMismatch("damping factors must have $D components"))
    weights = SVector{D,Float64}(factors)
    all(isfinite, weights) && all(>=(0), weights) ||
        throw(ArgumentError("damping factors must be finite and nonnegative"))
    isfinite(overlimit) && overlimit >= 0 && isfinite(rate) && rate >= 0 ||
        throw(ArgumentError("damping overlimit and rate must be finite and nonnegative"))
    return Float64(overlimit), Float64(rate), weights
end

function DampingZone(limit_min, limit_max; overlimit = 1, rate = 10, factors = ones(length(limit_min)))
    D = length(limit_min)
    D in (2, 3) || throw(ArgumentError("damping zones require 2D or 3D points"))
    length(limit_max) == D || throw(DimensionMismatch("damping points must have $D components"))
    origin, finish = SVector{D,Float64}(limit_min), SVector{D,Float64}(limit_max)
    all(isfinite, origin) && all(isfinite, finish) ||
        throw(ArgumentError("damping coordinates must be finite"))
    distance = norm(finish - origin)
    isfinite(distance) && distance > 0 || throw(ArgumentError("damping limits must be distinct with finite separation"))
    overlimit, rate, weights = damping_parameters(Val(D), overlimit, rate, factors)
    return DampingZone(origin, (finish - origin) / distance, distance, overlimit, rate, weights)
end

"""
    DampingBox(minimum_box, maximum_box; overlimit = 1, rate = 10,
        factors = ones(D), directions = :all)

Damping that ramps outward from an axis-aligned minimum-damping box to a
maximum-damping box. Each box is given as `(lower_corner, upper_corner)`.
`directions` selects active sides with `:left`, `:right`, `:front`, `:back`,
`:bottom`, and `:top`; 2D boxes use left/right/bottom/top. At corners, the
strongest active-side profile is used, matching DualSPHysics' box damping.
"""
struct DampingBox{D} <: AbstractDampingZone{D}
    minimum_min::SVector{D,Float64}
    minimum_max::SVector{D,Float64}
    maximum_min::SVector{D,Float64}
    maximum_max::SVector{D,Float64}
    directions::SVector{D,UInt8}
    overlimit::Float64
    rate::Float64
    factors::SVector{D,Float64}
end

function damping_box_bounds(bounds, ::Val{D}, name) where D
    length(bounds) == 2 ||
        throw(ArgumentError("$name must contain a lower and upper corner"))
    length(bounds[1]) == D && length(bounds[2]) == D ||
        throw(DimensionMismatch("$name corners must have $D components"))
    lower, upper = SVector{D,Float64}(bounds[1]), SVector{D,Float64}(bounds[2])
    all(isfinite, lower) && all(isfinite, upper) ||
        throw(ArgumentError("$name coordinates must be finite"))
    all(lower .<= upper) || throw(ArgumentError("$name lower corner must not exceed its upper corner"))
    return lower, upper
end

function damping_box_directions(directions, ::Val{D}) where D
    all_directions = D == 2 ?
        (:left, :right, :bottom, :top) :
        (:left, :right, :front, :back, :bottom, :top)
    active = directions === :all ? all_directions :
        directions isa Symbol ? (directions,) : Tuple(directions)
    bits = zeros(UInt8, D)
    for side in active
        axis, bit = if side === :left
            (1, 0x01)
        elseif side === :right
            (1, 0x02)
        elseif side === :bottom
            (D, 0x01)
        elseif side === :top
            (D, 0x02)
        elseif side === :front && D == 3
            (2, 0x01)
        elseif side === :back && D == 3
            (2, 0x02)
        else
            throw(ArgumentError("invalid damping box direction $(repr(side)) for $D dimensions"))
        end
        bits[axis] |= bit
    end
    any(!iszero, bits) || throw(ArgumentError("damping box requires at least one active direction"))
    return SVector{D,UInt8}(bits)
end

function DampingBox(minimum_box, maximum_box; overlimit = 1, rate = 10,
        factors = ones(length(first(minimum_box))), directions = :all)
    D = length(first(minimum_box))
    D in (2, 3) || throw(ArgumentError("damping boxes require 2D or 3D corners"))
    minimum_min, minimum_max = damping_box_bounds(minimum_box, Val(D), "minimum_box")
    maximum_min, maximum_max = damping_box_bounds(maximum_box, Val(D), "maximum_box")
    all(maximum_min .<= minimum_min) && all(minimum_max .<= maximum_max) ||
        throw(ArgumentError("maximum_box must contain minimum_box"))
    active_directions = damping_box_directions(directions, Val(D))
    for axis in 1:D
        active = active_directions[axis]
        (active & 0x01) == 0 || minimum_min[axis] > maximum_min[axis] ||
            throw(ArgumentError("minimum_box and maximum_box need positive lower-side width"))
        (active & 0x02) == 0 || maximum_max[axis] > minimum_max[axis] ||
            throw(ArgumentError("minimum_box and maximum_box need positive upper-side width"))
    end
    overlimit, rate, weights = damping_parameters(Val(D), overlimit, rate, factors)
    return DampingBox(minimum_min, minimum_max, maximum_min, maximum_max,
        active_directions, overlimit, rate, weights)
end

"""
    DampingCylinder(axis_start, axis_end, radius_min, radius_max;
        overlimit = 1, rate = 10, factors = ones(3))

Radial damping between two cylindrical radii around an infinite 3D axis.
The profile is zero at `radius_min`, reaches one at `radius_max`, and remains
one for `overlimit` beyond it.
"""
struct DampingCylinder <: AbstractDampingZone{3}
    axis_origin::SVector{3,Float64}
    axis_direction::SVector{3,Float64}
    radius_min::Float64
    radius_max::Float64
    overlimit::Float64
    rate::Float64
    factors::SVector{3,Float64}
end

function DampingCylinder(axis_start, axis_end, radius_min::Real, radius_max::Real;
        overlimit = 1, rate = 10, factors = ones(3))
    length(axis_start) == 3 && length(axis_end) == 3 ||
        throw(DimensionMismatch("cylinder axis points must have three components"))
    origin, finish = SVector{3,Float64}(axis_start), SVector{3,Float64}(axis_end)
    all(isfinite, origin) && all(isfinite, finish) ||
        throw(ArgumentError("cylinder axis coordinates must be finite"))
    axis = finish - origin
    axis_length = norm(axis)
    isfinite(axis_length) && axis_length > 0 ||
        throw(ArgumentError("cylinder axis points must be distinct"))
    isfinite(radius_min) && isfinite(radius_max) && 0 <= radius_min < radius_max ||
        throw(ArgumentError("cylinder radii must be finite with 0 <= radius_min < radius_max"))
    overlimit, rate, weights = damping_parameters(Val(3), overlimit, rate, factors)
    return DampingCylinder(origin, axis / axis_length, Float64(radius_min),
        Float64(radius_max), overlimit, rate, weights)
end

function prepare_damping(zones, ::Val{D}) where {D}
    zones === nothing && return ()
    result = if zones isa AbstractDampingZone{D}
        (zones,)
    elseif zones isa Tuple || zones isa AbstractVector
        Tuple(zones)
    else
        throw(ArgumentError("SimDamping requires damping-zone values"))
    end
    all(zone -> zone isa AbstractDampingZone{D}, result) ||
        throw(ArgumentError("SimDamping requires $D-dimensional DampingZone, DampingBox, or DampingCylinder values"))
    return result
end

@inline function normalized_damping(zone::DampingZone, position)
    distance = dot(position - zone.origin, zone.direction)
    if 0 < distance <= zone.length + zone.overlimit
        return min(distance / zone.length, 1.0)^2
    end
    return 0.0
end

@inline function normalized_damping(zone::DampingBox{D}, position::SVector{D}) where D
    inside_minimum = true
    for axis in 1:D
        bits = zone.directions[axis]
        lower_limit = (bits & 0x01) != 0 ?
            zone.maximum_min[axis] - zone.overlimit : zone.minimum_min[axis]
        upper_limit = (bits & 0x02) != 0 ?
            zone.maximum_max[axis] + zone.overlimit : zone.minimum_max[axis]
        if position[axis] < lower_limit || position[axis] > upper_limit
            return 0.0
        end
        if position[axis] < zone.minimum_min[axis] || position[axis] > zone.minimum_max[axis]
            inside_minimum = false
        end
    end
    inside_minimum && return 0.0

    profile = 0.0
    for axis in 1:D
        bits = zone.directions[axis]
        if (bits & 0x01) != 0
            distance = zone.minimum_min[axis] - position[axis]
            profile = max(profile, clamp(distance / (zone.minimum_min[axis] - zone.maximum_min[axis]), 0.0, 1.0))
        end
        if (bits & 0x02) != 0
            distance = position[axis] - zone.minimum_max[axis]
            profile = max(profile, clamp(distance / (zone.maximum_max[axis] - zone.minimum_max[axis]), 0.0, 1.0))
        end
    end
    return profile^2
end

@inline function normalized_damping(zone::DampingCylinder, position::SVector{3})
    relative = position - zone.axis_origin
    radial = relative - dot(relative, zone.axis_direction) * zone.axis_direction
    distance = sqrt(dot(radial, radial)) - zone.radius_min
    width = zone.radius_max - zone.radius_min
    if 0 < distance <= width + zone.overlimit
        return min(distance / width, 1.0)^2
    end
    return 0.0
end

@inline function damped_velocity(zones::Tuple, position, velocity::SVector{D,T}, kind, dt) where {D,T}
    kind == Fluid || return velocity
    for zone in zones
        profile = normalized_damping(zone, position)
        profile == 0 && continue
        reduction = Float64(dt) * zone.rate * profile
        velocity = SVector{D,T}(velocity .* max.(0.0, 1.0 .- reduction .* zone.factors))
    end
    return velocity
end

function damping_kernel!(zones, position, velocity, types, step, n)
    step_active(step) || return nothing
    n = step_particle_count(step,n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds velocity[i] = damped_velocity(zones, position[i], velocity[i], types[i], step_dt(step))
    end
    return nothing
end

function launch_damping!(zones::Tuple, position, velocity, types, step)
    isempty(zones) && return nothing
    n = length(types)
    n == 0 && return nothing
    @cuda threads=256 blocks=cld(n, 256) damping_kernel!(zones, position, velocity, types, step, n)
    return nothing
end

end
