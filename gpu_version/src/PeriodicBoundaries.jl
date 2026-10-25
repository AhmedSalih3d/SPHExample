module PeriodicBoundaries

using StaticArrays

export PeriodicBoundary, wrap_periodic_position

"""
    PeriodicBoundary{D}(axis; min, max, shift = ntuple(_ -> 0, D))

A single periodic axis in 2D or 3D, with physical interval `[min, max)`.
Crossing `max` translates a particle by `-(max-min)` along `axis` and by
`shift` in the transverse directions; crossing `min` applies the inverse.
For DualSPHysics `XPeriodicIncZ = 0.3` in a 2D x/height simulation, use
`PeriodicBoundary{2}(1; min, max, shift = (0, 0.3))`.

The axial shift must be zero. The interval must exceed twice the kernel
support radius (validated by the solver). Only one periodic axis is supported.
Floating bodies wrap together about their center; their diameter plus twice
the support radius must fit inside the interval.
"""
struct PeriodicBoundary{D, T <: AbstractFloat}
    axis::Int
    lower::T
    upper::T
    translation::SVector{D, T}

    function PeriodicBoundary{D, T}(axis::Integer; min, max, shift = ntuple(_ -> 0, D)) where {D, T <: AbstractFloat}
        D in (2, 3) || throw(ArgumentError("Periodic boundaries require 2D or 3D"))
        1 <= axis <= D || throw(ArgumentError("Periodic axis must be in 1:$D"))
        lo, hi = T(min), T(max)
        isfinite(lo) && isfinite(hi) && isfinite(hi - lo) && hi > lo ||
            throw(ArgumentError("Periodic min and max must be finite with max > min"))
        length(shift) == D || throw(ArgumentError("Periodic shift must have $D components"))
        delta = SVector{D, T}(shift)
        all(isfinite, delta) && iszero(delta[axis]) ||
            throw(ArgumentError("Periodic shift must be finite and transverse to its axis"))
        translation = setindex(delta, -(hi - lo), axis)
        new{D, T}(Int(axis), lo, hi, translation)
    end
end

PeriodicBoundary{D}(axis::Integer; kwargs...) where {D} = PeriodicBoundary{D, Float64}(axis; kwargs...)
Base.convert(::Type{PeriodicBoundary{D, T}}, p::PeriodicBoundary{D}) where {D, T} =
    PeriodicBoundary{D, T}(p.axis; min = p.lower, max = p.upper,
        shift = setindex(SVector{D, T}(p.translation), zero(T), p.axis))

@inline wrap_periodic_position(x, ::Nothing) = x
@inline function wrap_periodic_position(x::SVector{D, T}, p::PeriodicBoundary{D}) where {D, T}
    value = x[p.axis]
    isfinite(value) || return x # the grid's normal nonfinite validation reports this
    lo, hi = T(p.lower), T(p.upper)
    lo <= value < hi && return x
    length = hi - lo
    crossings = floor((value - lo) / length)
    result = x + crossings * SVector{D, T}(p.translation)
    # mod avoids a rounded subtraction leaving an exact upper-endpoint value.
    axial = lo + mod(value - lo, length)
    return setindex(result, axial < hi ? axial : prevfloat(hi), p.axis)
end

end
