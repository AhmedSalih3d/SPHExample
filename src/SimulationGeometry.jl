module SimulationGeometry

using StaticArrays
using Parameters

# Export relevant types and structs
export ParticleType, Geometry, Fluid, Fixed, Moving, Floating, MotionDetails

# Use the existing @enum for ParticleType
@enum ParticleType::UInt8 begin
    Fluid  = UInt8(1)
    Fixed  = UInt8(2)
    Moving = UInt8(3)
    Floating = UInt8(4)
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
    Motion::Union{Nothing, MotionDetails} = nothing  # Motion depends on dimension D and FloatType T
    Mass::Union{Nothing, T} = nothing
    COG::Union{Nothing, SVector{D, T}} = nothing
    Inertia::Union{Nothing, SMatrix{3, 3, T}} = nothing
end

end # module SimulationGeometry
