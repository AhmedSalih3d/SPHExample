module DampingZones

using CUDA
using StaticArrays
using LinearAlgebra: norm, dot
using ..SimulationGeometry: Fluid
using ..GPUStepState: step_active, step_dt

export DampingZone

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
struct DampingZone{D}
    origin::SVector{D,Float64}
    direction::SVector{D,Float64}
    length::Float64
    overlimit::Float64
    rate::Float64
    factors::SVector{D,Float64}
end

function DampingZone(limit_min, limit_max; overlimit = 1, rate = 10, factors = ones(length(limit_min)))
    D = length(limit_min)
    D in (2, 3) || throw(ArgumentError("damping zones require 2D or 3D points"))
    length(limit_max) == D && length(factors) == D || throw(DimensionMismatch("damping zone points and factors must have $D components"))
    origin, finish, weights = SVector{D,Float64}(limit_min), SVector{D,Float64}(limit_max), SVector{D,Float64}(factors)
    all(isfinite, origin) && all(isfinite, finish) && all(isfinite, weights) && all(>=(0), weights) ||
        throw(ArgumentError("damping coordinates must be finite and factors finite and nonnegative"))
    distance = norm(finish - origin)
    isfinite(distance) && distance > 0 || throw(ArgumentError("damping limits must be distinct with finite separation"))
    isfinite(overlimit) && overlimit >= 0 && isfinite(rate) && rate >= 0 ||
        throw(ArgumentError("damping overlimit and rate must be finite and nonnegative"))
    return DampingZone(origin, (finish - origin) / distance, distance, Float64(overlimit), Float64(rate), weights)
end

function prepare_damping(zones, ::Val{D}) where {D}
    zones === nothing && return ()
    result = zones isa DampingZone ? (zones,) : Tuple(zones)
    all(zone -> zone isa DampingZone{D}, result) || throw(ArgumentError("SimDamping requires $D-dimensional DampingZone values"))
    return result
end

@inline function damped_velocity(zones::Tuple, position, velocity::SVector{D,T}, kind, dt) where {D,T}
    kind == Fluid || return velocity
    for zone in zones
        distance = dot(position - zone.origin, zone.direction)
        if 0 < distance <= zone.length + zone.overlimit
            reduction = Float64(dt) * zone.rate * min(distance / zone.length, 1.0)^2
            velocity = SVector{D,T}(velocity .* max.(0.0, 1.0 .- reduction .* zone.factors))
        end
    end
    return velocity
end

function damping_kernel!(zones, position, velocity, types, step, n)
    step_active(step) || return nothing
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
