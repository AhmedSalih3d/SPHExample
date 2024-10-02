module SimulationGeometry

# Export relevant types and structs
export ParticleType, Geometry, Fluid, Fixed, Moving

# Use the existing @enum for ParticleType
@enum ParticleType::UInt8 begin
    Fluid  = UInt8(1)
    Fixed  = UInt8(2)
    Moving = UInt8(3)
end

# Define the Geometry struct to store the ParticleType enum
struct Geometry
    CSVFile::String
    GroupMarker::Int
    Type::ParticleType  # This uses the @enum ParticleType
    Motion::Union{Nothing, ParticleType}  # Motion can be represented using ParticleType (like Moving)
end

end # module SimulationGeometry
