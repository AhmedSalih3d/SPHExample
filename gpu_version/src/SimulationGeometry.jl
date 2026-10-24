module SimulationGeometry

using StaticArrays
using StructArrays
using CSV
using Base: @kwdef

# Export relevant types and structs
export ParticleType, SPHGeometry, particle_struct_array, Fluid, Fixed, Moving, Floating, Frozen,
       MotionDetails, FloatingDetails, FloatingConstraints, GravityFactorValue, MotionLimiterValue, is_wall

# Use the existing @enum for ParticleType
@enum ParticleType::UInt8 begin
    Fluid    = UInt8(1)
    Fixed    = UInt8(2)
    Moving   = UInt8(3)
    Floating = UInt8(4)   # rigid body moved by the fluid forces (see `FloatingDetails`)
    Frozen   = UInt8(5)   # permanently inactive after crossing a locked GPU grid limit
end

@inline GravityFactorValue(::Type{T}, type::ParticleType) where {T} =
    type == Fluid ? -one(T) : (type == Moving ? one(T) : zero(T))

@inline MotionLimiterValue(::Type{T}, type::ParticleType) where {T} =
    type == Fluid ? one(T) : zero(T)

"""
    is_wall(type)

Whether a particle type denotes a prescribed boundary (`Fixed` or `Moving`),
as opposed to a fluid or force-responsive floating body.
"""
@inline is_wall(type::ParticleType) = (type == Fixed) | (type == Moving)

# Define a struct to store motion details, with parametric dimensions and floating point type
"""
    MotionDetails{D, T}(; Velocity, StartTime, Duration, Direction,
                        MoveParticles = true)

Prescribed velocity for a `Moving` particle group. By default the particles
translate with that velocity. Set `MoveParticles = false` for a stationary
wall with a prescribed boundary velocity, as in a lid-driven cavity.
"""
@kwdef struct MotionDetails{D, T}
    Velocity::T
    StartTime::T
    Duration::T
    Direction::SVector{D, T}  # Direction vector is now parametric based on dimensions D and FloatType T
    MoveParticles::Bool = true
end

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
    FloatingDetails{T}(; RelativeWeight, PauseTime = 0, Constraints = nothing)

A rigid body (the DualSPHysics "floating" object) whose particles move
together under gravity and the forces of the surrounding particles.
`RelativeWeight` is the density of the body relative to `ρ₀`, so each body
particle weighs `RelativeWeight * m₀`. The body is held still until
`PauseTime` (DualSPHysics `FtPause`), which lets the fluid settle first.
`Constraints = FloatingConstraints{D,T}(...)` restricts or prescribes selected
degrees of freedom. Pause and permanent grid freezing override all motion.
Supported in 2D and 3D with both symplectic and single-neighbor time stepping.
"""
@kwdef struct FloatingDetails{T}
    RelativeWeight::T
    PauseTime::T = zero(RelativeWeight)
    Constraints::Union{Nothing,FloatingConstraints} = nothing
end

FloatingDetails{T}(weight, pause) where {T} = FloatingDetails{T}(weight, pause, nothing)
FloatingDetails(weight::T, pause::T) where {T} = FloatingDetails{T}(weight, pause)

"""
    SPHGeometry{D, T}(; Particles, GroupMarker, Type, Motion = nothing,
                        Floating = nothing)
    SPHGeometry{D, T}(positions; Density, kwargs...)
    SPHGeometry{D, T}(; CSVFile, GroupMarker, Type, kwargs...)

An input particle group held in a `StructArray`. `Particles` requires `Position`
and `Density` fields; optional `ID` values are one based and must be unique
across groups. Without `ID`, allocation assigns IDs after the explicitly
numbered particles. Optional `Velocity`, `GhostPoints`, and `GhostNormals`
initialize those solver fields. Group `Type` and `GroupMarker` take precedence.

The CSV constructor loads particles once, mapping `(x, z)` in 2D and converting
zero based CSV IDs to one based IDs. Positions retain their input precision
until allocation, including for `GPUDoublePosition` runs. Allocation copies the
stored particles, so the geometry can be reused for independent simulations.
Coincident positions remain separate particles; CSV loading does not deduplicate
rows.
"""
struct SPHGeometry{D, T}
    Particles::StructArray
    CSVFile::String
    GroupMarker::Int
    Type::ParticleType
    Motion::Union{Nothing, MotionDetails}
    Floating::Union{Nothing, FloatingDetails}
end

"""
    particle_struct_array(positions, ρ::Real; fields...)

Build a particle `StructArray` with the given `Position` values and a
constant `Density` filled with `ρ`. Additional particle fields can be
supplied as keywords.
"""
function particle_struct_array(positions::AbstractVector, ρ::Real; fields...)
    columns = merge(
        (Position = positions, Density = fill(ρ, length(positions))),
        (; fields...),
    )
    return StructArray(columns)
end

function SPHGeometry{D, T}(; Particles = nothing, CSVFile = nothing,
        GroupMarker::Int, Type::ParticleType, Motion = nothing,
        Floating = nothing) where {D, T}
    D in (2, 3) || throw(ArgumentError("particle dimension must be 2 or 3"))
    (Particles === nothing) != (CSVFile === nothing) ||
        throw(ArgumentError("provide exactly one of Particles or CSVFile"))
    source = CSVFile === nothing ? "" : String(CSVFile)
    if Particles === nothing
        rows = isempty(source) ? () : CSV.File(source)
        positions = SVector{D, Float64}[]
        density = T[]
        ids = Int[]
        for row in rows
            xyz = (row[Symbol("Points:0")], row[Symbol("Points:1")],
                   row[Symbol("Points:2")])
            push!(positions, D == 2 ? SVector{D, Float64}(xyz[1], xyz[3]) :
                                     SVector{D, Float64}(xyz))
            push!(density, row.Rhop)
            push!(ids, row.Idp + 1)
        end
        Particles = StructArray((Position = positions, Density = density, ID = ids))
    end
    Particles isa StructArray || throw(ArgumentError("Particles must be a StructArray"))
    ndims(Particles) == 1 || throw(ArgumentError("Particles must be one dimensional"))
    for field in (:Position, :Density)
        hasproperty(Particles, field) || throw(ArgumentError("Particles needs $field"))
    end
    all(x -> length(x) == D, Particles.Position) ||
        throw(DimensionMismatch("particle positions must have $D components"))
    if hasproperty(Particles, :ID)
        ids = Particles.ID
        all(id -> id isa Integer && id > 0, ids) ||
            throw(ArgumentError("particle IDs must be positive integers"))
        length(unique(ids)) == length(ids) ||
            throw(ArgumentError("particle IDs must be unique"))
    end
    return SPHGeometry{D, T}(Particles, source, GroupMarker, Type, Motion, Floating)
end

function SPHGeometry{D, T}(positions::AbstractVector; Density, kwargs...) where {D, T}
    density = Density isa Number ? fill(T(Density), length(positions)) : T.(Density)
    length(density) == length(positions) ||
        throw(DimensionMismatch("one density is required per position"))
    particles = StructArray((Position = copy(positions), Density = density))
    return SPHGeometry{D, T}(; Particles = particles, kwargs...)
end

end # module SimulationGeometry
