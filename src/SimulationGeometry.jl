module SimulationGeometry

using Parameters  # Import Parameters.jl for easier struct definition

# Export relevant types and structs
export ParticleType, Geometry, Fluid, Fixed, Moving

# Use the existing @enum for ParticleType
@enum ParticleType::UInt8 begin
    Fluid  = UInt8(1)
    Fixed  = UInt8(2)
    Moving = UInt8(3)
end

# Define the Geometry struct with named fields using @with_kw
@with_kw struct Geometry
    CSVFile::String
    GroupMarker::Int
    Type::ParticleType  # This uses the @enum ParticleType
    Motion::Union{Nothing, ParticleType} = nothing  # Motion can be represented using ParticleType (like Moving), default is nothing
end

end # module SimulationGeometry
