"""
Rigid-body constraints and force links shared by simulation geometry and the
GPU floating-body integrator. These native models do not require Chrono.
"""
module Constraints

using StaticArrays
using LinearAlgebra: dot

export FloatingConstraints, LinearSpring, CoulombDamper, axial_spring_force, axial_coulomb_force

"""
    FloatingConstraints{D,T}(; TranslationLocks, RotationLocks,
        PrescribedVelocity, PrescribedOmega, HingePoint, HingeAxis, HingeVelocity)

Constraints on a floating body in fixed world coordinates. Translation tuples
have `D` entries; rotation tuples have one entry in 2D and three in 3D. Locks
default to `false`. Prescribed entries default to `nothing` (force responsive);
a numeric entry imposes that constant velocity, including zero. A locked entry
cannot also have a prescribed velocity. Rotation locks constrain angular
velocity about world axes, rather than Euler angles.

Alternatively, `HingePoint` fixes a world-space pivot. In 3D, supply a nonzero
`HingeAxis` (normalized internally); in 2D the axis is perpendicular to the
plane. `HingeVelocity` optionally prescribes the scalar angular velocity about
that axis. Hinges cannot be combined with the component locks or prescriptions.
Pivot coordinates retain Float64 precision, including with Float32 physics.
"""
struct FloatingConstraints{D,T,R}
    active::Bool
    translation_free::SVector{D,Bool}
    velocity::SVector{D,T}
    rotation_free::SVector{R,Bool}
    omega::SVector{R,T}
    hinge::Bool
    point::SVector{D,Float64}
    axis::SVector{R,T}
    hinge_prescribed::Bool
    hinge_velocity::T
end

function FloatingConstraints{D,T}(;
        TranslationLocks = ntuple(_ -> false, D),
        RotationLocks = ntuple(_ -> false, D == 2 ? 1 : 3),
        PrescribedVelocity = ntuple(_ -> nothing, D),
        PrescribedOmega = ntuple(_ -> nothing, D == 2 ? 1 : 3),
        HingePoint = nothing, HingeAxis = nothing, HingeVelocity = nothing) where {D,T}
    D in (2, 3) || throw(ArgumentError("floating constraints require 2D or 3D"))
    R = D == 2 ? 1 : 3
    for (name, values, n) in ((:TranslationLocks, TranslationLocks, D), (:RotationLocks, RotationLocks, R),
                             (:PrescribedVelocity, PrescribedVelocity, D), (:PrescribedOmega, PrescribedOmega, R))
        length(values) == n || throw(ArgumentError("$name must have $n entries"))
    end
    all(x -> x isa Bool, TranslationLocks) && all(x -> x isa Bool, RotationLocks) ||
        throw(ArgumentError("locks must contain Bool values"))
    for (locks, values) in ((TranslationLocks, PrescribedVelocity), (RotationLocks, PrescribedOmega))
        for (locked, value) in zip(locks, values)
            locked && value !== nothing && throw(ArgumentError("a locked degree of freedom cannot also be prescribed"))
            value === nothing || (value isa Real && isfinite(T(value))) ||
                throw(ArgumentError("prescribed velocities must be finite real numbers or nothing"))
        end
    end
    tfree = SVector{D,Bool}(ntuple(d -> !TranslationLocks[d] && PrescribedVelocity[d] === nothing, D))
    rfree = SVector{R,Bool}(ntuple(d -> !RotationLocks[d] && PrescribedOmega[d] === nothing, R))
    velocity = SVector{D,T}(ntuple(d -> PrescribedVelocity[d] === nothing ? zero(T) : T(PrescribedVelocity[d]), D))
    omega = SVector{R,T}(ntuple(d -> PrescribedOmega[d] === nothing ? zero(T) : T(PrescribedOmega[d]), R))
    hinge = HingePoint !== nothing
    point = zero(SVector{D,Float64})
    axis = SVector{R,T}(ntuple(d -> d == R ? one(T) : zero(T), R))
    if hinge
        all(tfree) && all(rfree) || throw(ArgumentError("hinges cannot be combined with component constraints"))
        length(HingePoint) == D || throw(ArgumentError("HingePoint must have $D entries"))
        point = SVector{D,Float64}(HingePoint)
        all(isfinite, point) || throw(ArgumentError("HingePoint must be finite"))
        if D == 3
            HingeAxis === nothing && throw(ArgumentError("a 3D hinge requires HingeAxis"))
            length(HingeAxis) == 3 || throw(ArgumentError("HingeAxis must have three entries"))
            axis = SVector{3,T}(HingeAxis)
            scale = maximum(abs, axis)
            isfinite(scale) && scale > zero(T) || throw(ArgumentError("HingeAxis must be finite and nonzero"))
            axis = axis / scale
            axis = axis / sqrt(sum(abs2, axis))
        else
            HingeAxis === nothing || throw(ArgumentError("a 2D hinge uses the perpendicular axis; omit HingeAxis"))
        end
    else
        HingeAxis === nothing && HingeVelocity === nothing ||
            throw(ArgumentError("HingeAxis and HingeVelocity require HingePoint"))
    end
    speed = HingeVelocity === nothing ? zero(T) : T(HingeVelocity)
    isfinite(speed) || throw(ArgumentError("HingeVelocity must be finite"))
    return FloatingConstraints{D,T,R}(hinge || !all(tfree) || !all(rfree), tfree, velocity,
        rfree, omega, hinge, point, axis, HingeVelocity !== nothing, speed)
end

"""
    LinearSpring{D,T}(; Anchor, Attachment, Stiffness, Damping=0,
        RestLength=norm(Attachment-Anchor))

A damped axial spring from a fixed world `Anchor` to a body `Attachment` in
the initial geometry's world coordinates. Stiffness is N/m, damping is Ns/m,
and rest length is metres. It supports tension and compression. The attachment
turns with the body; its point velocity determines damping and its lever arm
determines torque. Put one or more springs in `FloatingDetails.Springs`.
"""
struct LinearSpring{D,T}
    Anchor::SVector{D,Float64}
    Attachment::SVector{D,Float64}
    Stiffness::T
    Damping::T
    RestLength::T
end

function LinearSpring{D,T}(;Anchor,Attachment,Stiffness,Damping=0,RestLength=nothing) where {D,T<:AbstractFloat}
    D in (2,3) || throw(ArgumentError("linear springs require 2D or 3D"))
    a,p = SVector{D,Float64}(Anchor),SVector{D,Float64}(Attachment)
    all(isfinite,a) && all(isfinite,p) || throw(ArgumentError("spring points must be finite"))
    k,c = T(Stiffness),T(Damping)
    length0 = RestLength === nothing ? T(sqrt(sum(abs2,p-a))) : T(RestLength)
    all(x -> isfinite(x) && x >= 0,(k,c,length0)) ||
        throw(ArgumentError("spring stiffness, damping and rest length must be finite and nonnegative"))
    return LinearSpring{D,T}(a,p,k,c,length0)
end

"""
    CoulombDamper{D,T}(; Anchor, Attachment, Force, RestLength=norm(Attachment-Anchor))

Fixed-anchor axial sliding damper with constant opposing force in newtons.
Put it in `FloatingDetails.Springs`. Zero axial velocity or a collapsed link
has zero force. The GPU caps the stage impulse at the evaluated stopping
impulse to avoid reversing motion through a velocity zero. This is sliding
Coulomb damping, not a static-friction constraint. `RestLength` is for output.
"""
struct CoulombDamper{D,T}
    Anchor::SVector{D,Float64}
    Attachment::SVector{D,Float64}
    Force::T
    RestLength::T
end

function CoulombDamper{D,T}(;Anchor,Attachment,Force,RestLength = nothing) where {D,T<:AbstractFloat}
    D in (2,3) || throw(ArgumentError("Coulomb dampers require 2D or 3D"))
    a,p = SVector{D,Float64}(Anchor),SVector{D,Float64}(Attachment)
    all(isfinite,a) && all(isfinite,p) || throw(ArgumentError("damper points must be finite"))
    force = T(Force)
    length0 = RestLength === nothing ? T(sqrt(sum(abs2,p-a))) : T(RestLength)
    isfinite(force) && force >= 0 && isfinite(length0) && length0 >= 0 ||
        throw(ArgumentError("damper force and rest length must be finite and nonnegative"))
    return CoulombDamper{D,T}(a,p,force,length0)
end

"Axial sliding resistance; `force_limit` optionally caps a finite-stage stopping impulse."
@inline function axial_coulomb_force(displacement,velocity,magnitude,force_limit = Inf)
    length = sqrt(dot(displacement,displacement))
    iszero(length) && return zero(velocity)
    direction = displacement/length
    speed = dot(velocity,direction)
    return typeof(velocity)(-min(magnitude,force_limit)*sign(speed)*direction)
end

"""
    axial_spring_force(displacement, point_velocity, stiffness, damping, rest_length)

Force on a spring attachment whose displacement is measured from a fixed
anchor. Damping acts along the spring axis. A collapsed link has zero force
because its axial direction is undefined. This allocation-free law supports
host and CUDA execution; the rigid-body integrator adds the attachment torque.
"""
@inline function axial_spring_force(displacement, point_velocity, stiffness, damping, rest_length)
    length = sqrt(dot(displacement, displacement))
    iszero(length) && return zero(point_velocity)
    direction = displacement / length
    axial_velocity = dot(point_velocity, direction)
    load = -stiffness * (length - rest_length) - damping * axial_velocity
    return typeof(point_velocity)(load * direction)
end

export VelocitySeries, RigidVelocity, BodySpring
include("RigidVelocityConfiguration.jl")
include("RigidBodyDynamicsConfiguration.jl")

end # module Constraints
