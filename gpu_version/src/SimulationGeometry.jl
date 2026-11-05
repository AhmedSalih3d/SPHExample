module SimulationGeometry

using StaticArrays
using StructArrays
using CSV
using Base: @kwdef
using ..Constraints: FloatingConstraints, LinearSpring

# Export relevant types and structs
export ParticleType, SPHGeometry, particle_struct_array, Fluid, Fixed, Moving, Floating, Frozen,
       MotionDetails, FloatingDetails, FloatingConstraints, LinearSpring, GravityFactorValue, MotionLimiterValue, is_wall

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
    FloatingDetails{T}(; RelativeWeight, PauseTime = 0, Constraints = nothing,
          Springs=(), InitialVelocity=nothing, Center=nothing, Inertia=nothing)

A rigid body (the DualSPHysics "floating" object) whose particles move
together under gravity and the forces of the surrounding particles.
`RelativeWeight` is the density of the body relative to `ρ₀`, so each body
particle weighs `RelativeWeight * m₀`. The body is held still until
`PauseTime` (DualSPHysics `FtPause`), which lets the fluid settle first.
`Constraints = FloatingConstraints{D,T}(...)` restricts or prescribes selected
degrees of freedom. Pause and permanent grid freezing override all motion.
Supported in 2D and 3D with both symplectic and single-neighbor time stepping.
`Springs=(LinearSpring{D,T}(...), ...)` adds fixed-anchor spring forces on the
GPU at each evaluated body state, including torque and a spring timestep limit.
Optional `Center` overrides the sampled centre of mass in native coordinates.
`Inertia` overrides the body-frame inertia: a positive scalar in 2D, or a
positive diagonal tuple/vector or symmetric positive-definite 3×3 matrix in 3D.
`InitialVelocity=(vx,vy)` in 2D or `(vx,vy,vz)` in 3D sets a world-space initial
translation velocity. It does not prescribe later motion: forces and gravity
continue to change the velocity. Constraints project it onto allowed degrees
of freedom; pause and grid freezing override it. Values must be finite and
match the body's dimension, checked when constructing its device state.
"""
@kwdef struct FloatingDetails{T}
    RelativeWeight::T
    PauseTime::T = zero(RelativeWeight)
    Constraints::Union{Nothing,FloatingConstraints} = nothing
    Springs::Tuple = ()
    InitialVelocity::Union{Nothing,Tuple,AbstractVector} = nothing
    Center::Union{Nothing,Tuple,AbstractVector} = nothing
    Inertia::Union{Nothing,Real,Tuple,AbstractArray} = nothing
end

FloatingDetails{T}(weight,pause,constraints,springs,initial) where T = FloatingDetails{T}(weight,pause,constraints,springs,initial,nothing,nothing)

FloatingDetails{T}(weight, pause, constraints, springs) where {T} = FloatingDetails{T}(weight,pause,constraints,springs,nothing)

FloatingDetails{T}(weight, pause, constraints) where {T} = FloatingDetails{T}(weight, pause, constraints, ())
FloatingDetails{T}(weight, pause) where {T} = FloatingDetails{T}(weight, pause, nothing, ())
FloatingDetails(weight::T, pause::T) where {T} = FloatingDetails{T}(weight, pause)
FloatingDetails(weight::T, pause::T, constraints) where {T} = FloatingDetails{T}(weight, pause, constraints, ())

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
