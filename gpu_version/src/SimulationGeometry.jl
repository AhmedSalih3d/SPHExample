module SimulationGeometry

using StaticArrays
using Base: @kwdef

# Export relevant types and structs
export ParticleType, SPHGeometry, Fluid, Fixed, Moving, Floating, MotionDetails, FloatingDetails,
       GravityFactorValue, MotionLimiterValue

# Use the existing @enum for ParticleType
@enum ParticleType::UInt8 begin
    Fluid    = UInt8(1)
    Fixed    = UInt8(2)
    Moving   = UInt8(3)
    Floating = UInt8(4)   # rigid body moved by the fluid forces (see `FloatingDetails`)
end

@inline GravityFactorValue(::Type{T}, type::ParticleType) where {T} =
    type == Fluid ? -one(T) : (type == Moving ? one(T) : zero(T))

@inline MotionLimiterValue(::Type{T}, type::ParticleType) where {T} =
    type == Fluid ? one(T) : zero(T)

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
    FloatingDetails{T}(; RelativeWeight, PauseTime = 0)

A rigid body (the DualSPHysics "floating" object) whose particles move
together under gravity and the forces of the surrounding particles.
`RelativeWeight` is the density of the body relative to `ρ₀`, so each body
particle weighs `RelativeWeight * m₀`. The body is held still until
`PauseTime` (DualSPHysics `FtPause`), which lets the fluid settle first.
Supported in 2D with the symplectic time stepping.
"""
@kwdef struct FloatingDetails{T}
    RelativeWeight::T
    PauseTime::T = zero(RelativeWeight)
end

# Define the SPHGeometry struct to store the ParticleType enum and Motion details
@kwdef struct SPHGeometry{D, T}
    CSVFile::String
    GroupMarker::Int
    Type::ParticleType
    Motion::Union{Nothing, MotionDetails} = nothing  # Motion depends on dimension D and FloatType T
    Floating::Union{Nothing, FloatingDetails} = nothing  # required exactly when Type == Floating
end

end # module SimulationGeometry
