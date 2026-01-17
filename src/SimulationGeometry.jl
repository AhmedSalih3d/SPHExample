module SimulationGeometry

using StaticArrays
using Parameters

# Export relevant types and structs
export ParticleType, Geometry, Fluid, Fixed, Moving, MotionDetails, GravityFactor, MotionLimiter

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
    Motion::Union{Nothing, MotionDetails} = nothing  # Motion depends on dimension D and FloatType T
end

@inline GravityFactor(::Type{T}, ::Val{Fluid}) where {T} = -one(T)
@inline GravityFactor(::Type{T}, ::Val{Moving}) where {T} = one(T)
@inline GravityFactor(::Type{T}, ::Val{Fixed}) where {T} = zero(T)
@inline GravityFactor(::Type{T}, particle_type::ParticleType) where {T} = GravityFactor(T, Val(particle_type))

@inline MotionLimiter(::Type{T}, ::Val{Fluid}) where {T} = one(T)
@inline MotionLimiter(::Type{T}, ::Val{Moving}) where {T} = zero(T)
@inline MotionLimiter(::Type{T}, ::Val{Fixed}) where {T} = zero(T)
@inline MotionLimiter(::Type{T}, particle_type::ParticleType) where {T} = MotionLimiter(T, Val(particle_type))

end # module SimulationGeometry
