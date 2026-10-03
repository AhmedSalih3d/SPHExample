module SimulationGeometry

using StaticArrays
using Base: @kwdef

# Export relevant types and structs
export ParticleType, Geometry, Fluid, Fixed, Moving, MotionDetails

# Use the existing @enum for ParticleType
@enum ParticleType::UInt8 begin
    Fluid  = UInt8(1)
    Fixed  = UInt8(2)
    Moving = UInt8(3)
end

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

# Define the Geometry struct to store the ParticleType enum and Motion details
@kwdef struct Geometry{D, T}
    CSVFile::String
    GroupMarker::Int
    Type::ParticleType
    Motion::Union{Nothing, MotionDetails} = nothing  # Motion depends on dimension D and FloatType T
end

end # module SimulationGeometry
