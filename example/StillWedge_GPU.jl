using SPHExample
using StructArrays
using CUDA


Dimensions = 2
FloatType  = Float64

SimConstants = SimulationConstants{FloatType}(dx=0.02,c₀=42.48576250492629, δᵩ = 0.1, CFL=0.2)

# Define the dictionary with specific types for keys and values to avoid any type ambiguity
SimGeometry = Dict{Symbol, Dict{String, Union{String, Int, ParticleType, Nothing}}}()

# Populate the dictionary
SimGeometry[:FixedBoundary] = Dict(
    "CSVFile"     => "./input/still_wedge/StillWedge_Dp$(SimConstants.dx)_Bound.csv",
    "GroupMarker" => 1,
    "Type"        => Fixed,
    "Motion"      => nothing
)
SimGeometry[:Water] = Dict(
    "CSVFile"     => "./input/still_wedge/StillWedge_Dp$(SimConstants.dx)_Fluid.csv",
    "GroupMarker" => 2,
    "Type"        => Fluid,
    "Motion"      => nothing
)
SimMetaData  = SimulationMetaData{Dimensions,FloatType}(
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

NumberOfPoints::Int = length(SimParticles)

# Produce sorting related variables
ParticleRanges         = CUDA.zeros(Int, NumberOfPoints + 1)
UniqueCells            = CUDA.zeros(CartesianIndex{Dimensions}, NumberOfPoints)
Stencil                = ConstructStencil(Val(Dimensions))

InverseCutOff = Val(1/(SimConstants.H))


# @inline function ExtractCells!(Particles, ::Val{InverseCutOff}) where InverseCutOff
#     # Replace unsafe_trunc with trunc if this ever errors
#     function map_floor(x)
#         unsafe_trunc(Int, muladd(x,InverseCutOff,2))
#     end

#     Cells  = @views Particles.Cells
#     Points = @views Particles.Position
#     @threads for i ∈ eachindex(Particles)
#         t = map(map_floor, Tuple(Points[i]))
#         Cells[i] = CartesianIndex(t)
#     end
#     return nothing
# end

# ###=== Function to update ordering
# function UpdateNeighbors!(Particles, CutOff, SortingScratchSpace, ParticleRanges, UniqueCells)
#     ExtractCells!(Particles, CutOff)

#     sort!(Particles, by = p -> p.Cells; scratch=SortingScratchSpace)

#     Cells = @views Particles.Cells
#     @. ParticleRanges             = zero(eltype(ParticleRanges))
#     IndexCounter                  = 1
#     ParticleRanges[IndexCounter]  = 1
#     UniqueCells[IndexCounter]     = Cells[1]

#     for i in 2:length(Cells)
#         if Cells[i] != Cells[i-1] # Equivalent to diff(Cells) != 0
#             IndexCounter                += 1
#             ParticleRanges[IndexCounter] = i
#             UniqueCells[IndexCounter]    = Cells[i]
#         end
#     end
#     ParticleRanges[IndexCounter + 1]  = length(ParticleRanges)

#     return IndexCounter 
# end