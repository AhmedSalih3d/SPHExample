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

# Define a struct to store motion details, with parametric dimensions and floating point type
@kwdef struct MotionDetails{D, T}
    Velocity::T
    StartTime::T
    Duration::T
    Direction::SVector{D, T}  # Direction vector is now parametric based on dimensions D and FloatType T
end

# Define the Geometry struct to store the ParticleType enum and Motion details
@kwdef struct Geometry{D, T}
    CSVFile::String
    GroupMarker::Int
    Type::ParticleType
    Motion::Union{Nothing, MotionDetails} = nothing  # Motion depends on dimension D and FloatType T
end

end # module SimulationGeometry
