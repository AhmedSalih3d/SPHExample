module SimulationGeometry

using StaticArrays
using Parameters

# Export relevant types and structs
export ParticleType, Geometry, Fluid, Fixed, Moving, MotionDetails, PeriodicityConditions

# Use the existing @enum for ParticleType
@enum ParticleType::UInt8 begin
    Fluid  = UInt8(1)
    Fixed  = UInt8(2)
    Moving = UInt8(3)
end

# Define a struct to store motion details, with parametric dimensions and floating point type
@with_kw struct MotionDetails{D, T}
    Velocity::T
    StartTime::T
    Duration::T
    Direction::SVector{D, T}  # Direction vector is now parametric based on dimensions D and FloatType T
end

# Define the Geometry struct to store the ParticleType enum and Motion details
@with_kw struct Geometry{D, T}
    CSVFile::String
    GroupMarker::Int
    Type::ParticleType
    Motion::Union{Nothing, MotionDetails{D, T}} = nothing  # Motion depends on dimension D and FloatType T
end

@with_kw struct PeriodicityConditions{D, T}
    IsPeriodic::SVector{D, Bool}    # A vector indicating periodicity in each dimension (true for periodic, false otherwise)
    MinBounds::SVector{D, T}        # Minimum boundary for each dimension (e.g., x_min, y_min, z_min)
    MaxBounds::SVector{D, T}        # Maximum boundary for each dimension (e.g., x_max, y_max, z_max)
    HeightIncrease::T               # Scalar value for increase in height (applies when reentering the domain)
end


end # module SimulationGeometry
