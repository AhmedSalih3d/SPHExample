module SimulationGeometry

using StaticArrays
using Parameters

# Export relevant types and structs
export ParticleType, Geometry, Fluid, Fixed, Moving, FixedMoving, MotionDetails, GravityFactorValue, MotionLimiterValue, MotionPositionFactorValue

# Use the existing @enum for ParticleType
@enum ParticleType::UInt8 begin
    Fluid  = UInt8(1)
    Fixed  = UInt8(2)
    Moving = UInt8(3)
    FixedMoving = UInt8(4)
end

@inline function GravityFactorValue(::Type{T}, particle_type::ParticleType) where {T}
    if particle_type == Fluid
        return -one(T)
    elseif particle_type == Moving
        return one(T)
    end
    return zero(T)
end

@inline function MotionLimiterValue(::Type{T}, particle_type::ParticleType) where {T}
    return particle_type == Fluid ? one(T) : zero(T)
end

@inline function MotionPositionFactorValue(::Type{T}, particle_type::ParticleType) where {T}
    return particle_type == Moving ? one(T) : zero(T)
end

# Define a struct to store motion details, with parametric dimensions and floating point type
@with_kw struct MotionDetails{D, T}
    Velocity::T
    StartTime::T
    Duration::T
    Direction::SVector{D, T}  # Direction vector is now parametric based on dimensions D and FloatType T
    MovePosition::Bool = true
end

# Define the Geometry struct to store the ParticleType enum and Motion details
@with_kw struct Geometry{D, T}
    CSVFile::String
    GroupMarker::Int
    Type::ParticleType
    Motion::Union{Nothing, MotionDetails} = nothing  # Motion depends on dimension D and FloatType T
end

end # module SimulationGeometry
