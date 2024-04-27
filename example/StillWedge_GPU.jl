using SPHExample
using StructArrays
using CUDA


Dimensions = 2
FloatType  = Float64

SimConstantsWedge = SimulationConstants{FloatType}(dx=0.02,c₀=42.48576250492629, δᵩ = 0.1, CFL=0.2)

# Define the dictionary with specific types for keys and values to avoid any type ambiguity
SimGeometry = Dict{Symbol, Dict{String, Union{String, Int, ParticleType, Nothing}}}()

# Populate the dictionary
SimGeometry[:FixedBoundary] = Dict(
    "CSVFile"     => "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Bound.csv",
    "GroupMarker" => 1,
    "Type"        => Fixed,
    "Motion"      => nothing
)
SimGeometry[:Water] = Dict(
    "CSVFile"     => "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Fluid.csv",
    "GroupMarker" => 2,
    "Type"        => Fluid,
    "Motion"      => nothing
)
SimMetaDataWedge  = SimulationMetaData{Dimensions,FloatType}(
    SimulationName="StillWedge", 
    SaveLocation="E:/SecondApproach/TESTING_CPU_StillWedge",
    SimulationTime=4,
    OutputEach=0.01,
    FlagDensityDiffusion=true,
    FlagOutputKernelValues=false,
    FlagLog=true,
    FlagShifting=false,
)

SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation)

# Allocate data structures on the CPU
SimParticles, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺ = AllocateDataStructures(Dimensions, FloatType, SimGeometry)

# Allow scalar operations temporarily
CUDA.allowscalar(true)

# Replace storage for each variable to use GPU memory
SimParticles_GPU = replace_storage(CuVector, SimParticles)
dρdtI_GPU        = replace_storage(CuVector, dρdtI)
Velocityₙ⁺_GPU   = replace_storage(CuVector, Velocityₙ⁺)
Positionₙ⁺_GPU   = replace_storage(CuVector, Positionₙ⁺)
ρₙ⁺_GPU          = replace_storage(CuVector, ρₙ⁺)

# Disable scalar operations for performance optimization in CUDA operations
CUDA.allowscalar(false)