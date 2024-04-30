using SPHExample
using StructArrays
using CUDA
using Parameters
import LinearAlgebra: dot, norm


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

SimLogger = SimulationLogger(SimMetaData.SaveLocation)

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

CutOff = SimConstants.H
# InverseCutOff = Val(1/(SimConstants.H))

# CUDA kernel for extracting cells
function gpu_ExtractCells!(Particles, CutOff)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    Cells = Particles.Cells
    Points = Particles.Position

    i = index
    while i <= length(Cells)

        ci  = CartesianIndex((@. Int(fld(Points[i], CutOff)) + 2)... )
        Cells[i] = ci #+ 2 * one(ci) 

        i += stride
    end
    return
end

# Function to launch the CUDA kernel for extracting cells
function launch_ExtractCellsKernel!(Particles, CutOff)
    kernel = @cuda launch=false gpu_ExtractCells!(Particles, CutOff)
    config = launch_configuration(kernel.fun)
    
    threads = min(length(Particles.Cells), config.threads)
    blocks = cld(length(Particles.Cells), threads)

    # Launching the CUDA kernel with the calculated configuration
    CUDA.@sync kernel(Particles, CutOff; threads=threads, blocks=blocks)
end

launch_ExtractCellsKernel!(SimParticles_GPU, CutOff)


SortedIndices = CUDA.zeros(Int,NumberOfPoints)

sortperm!(SortedIndices, SimParticles_GPU.Cells)

for prop in propertynames(SimParticles_GPU)
    getproperty(SimParticles_GPU,prop) .= getproperty(SimParticles_GPU, prop)[SortedIndices]
end

ParticleRanges .= 0

ParticleRangesIndices = findall(diff(SimParticles_GPU.Cells) .!= CartesianIndex(0,0))

ParticleRanges[1:1] = 1
ParticleRanges[2:(length(ParticleRangesIndices)+1)] .= ParticleRangesIndices .+ 1
ParticleRanges[end:end] = length(ParticleRanges)

function zero_last_comparator(x, y)
    # If both are zeros, they are considered equal
    if x == 0 && y == 0
        return false
    # If only x is zero, y should come first
    elseif x == 0
        return false
    # If only y is zero, x should come first
    elseif y == 0
        return true
    else
        # Otherwise, compare them numerically
        return x < y
    end
end

sort!(ParticleRanges, lt=zero_last_comparator)


IndexCounter = findfirst(isequal(0), ParticleRanges) - 2

Position       = SimParticles_GPU.Position
Cells          = SimParticles_GPU.Cells
Density        = SimParticles_GPU.Density
Pressure       = SimParticles_GPU.Pressure
Velocity       = SimParticles_GPU.Velocity
Acceleration   = SimParticles_GPU.Acceleration
GravityFactor  = SimParticles_GPU.GravityFactor
MotionLimiter  = SimParticles_GPU.MotionLimiter
# ParticleType   = SimParticles_GPU.Type
ParticleMarker = SimParticles_GPU.GroupMarker
Kernel         = SimParticles_GPU.Kernel
KernelGradient = SimParticles_GPU.KernelGradient

    # A few time stepping controls implemented to allow for an adaptive time step
    function SPHExample.Δt(Position, Velocity, Acceleration, SimulationConstants)
        @unpack c₀, h, CFL, η² = SimulationConstants
        
        visc = maximum(@. abs(h * dot(Velocity,Position) / (dot(Position,Position) + η²)))
        dt1  = minimum(@. sqrt(h / norm(Acceleration)))

        dt2   = h / (c₀+visc)

        dt    = CFL*min(dt1,dt2)

        return dt
    end

dt  = Δt(Position, Velocity, Acceleration, SimConstants)
dt₂ = dt * 0.5


    function SPHExample.Pressure!(Press, Density, SimulationConstants)
        @unpack c₀,γ,ρ₀ = SimulationConstants

        Press .= @. EquationOfStateGamma7(Density,c₀,ρ₀)
    end


Pressure!(SimParticles_GPU.Pressure,SimParticles_GPU.Density,SimConstants)

UniqueCells = Cells[collect(ParticleRanges[1:IndexCounter])]

function gpu_NeighborLoop!(Particles, UniqueCells, ParticleRanges, Stencil, IndexCounter)
    index  = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x


    i = index
    while i <= IndexCounter
        CellIndex  = UniqueCells[i]

        StartIndex = ParticleRanges[i] 
        EndIndex   = ParticleRanges[i+1] - 1

        @inbounds for i = StartIndex:EndIndex, j = (i+1):EndIndex

        end

        for S ∈ Stencil
            SCellIndex = CellIndex + S
            # Returns a range, x:x for exact match and x:(x-1) for no match
            # utilizes that it is a sorted array and requires no isequal constructor,
            # so I prefer this for now
            NeighborCellIndex = searchsorted(UniqueCells, SCellIndex)

            if length(NeighborCellIndex) != 0
                StartIndex_       = ParticleRanges[NeighborCellIndex[1]] 
                EndIndex_         = ParticleRanges[NeighborCellIndex[1]+1] - 1

                @inbounds for i = StartIndex:EndIndex, j = StartIndex_:EndIndex_
                    
                end
            end
        end

        i += stride
    end
    return
end

# Function to launch the CUDA kernel for extracting cells
function launch_NeighborLoopKernel!(Particles, UniqueCells, ParticleRanges, Stencil, IndexCounter)
    kernel = @cuda launch=false gpu_NeighborLoop!(Particles, UniqueCells, ParticleRanges, Stencil, IndexCounter)
    config = launch_configuration(kernel.fun)
    
    threads = min(length(Particles.Cells), config.threads)
    blocks = cld(length(Particles.Cells), threads)

    # Launching the CUDA kernel with the calculated configuration
    CUDA.@sync kernel(Particles, UniqueCells, ParticleRanges, Stencil, IndexCounter; threads=threads, blocks=blocks)
end

launch_NeighborLoopKernel!(SimParticles_GPU, UniqueCells, ParticleRanges, CuVector(Stencil), IndexCounter)
