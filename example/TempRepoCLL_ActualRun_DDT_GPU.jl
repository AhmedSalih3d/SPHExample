using SPHExample
using BenchmarkTools
using StaticArrays
using Parameters
using Plots; using Measures
using StructArrays
import LinearAlgebra: dot, norm, diagm, diag, cond, det
using LoopVectorization
using Polyester
using JET
using Formatting
using ProgressMeter
using TimerOutputs
using FastPow
using ChunkSplitters
import Cthulhu as Deep
import CellListMap: InPlaceNeighborList, update!, neighborlist!
using Bumper
using CUDA

import Base.Threads: nthreads, @threads
include("../src/ProduceVTP.jl")


function ConstructGravitySVector(_::SVector{N, T}, value) where {N, T}
    return SVector{N, T}(ntuple(i -> i == N ? value : 0, N))
end

function ConstructStencil(v::Val{d}) where d
    n_ = CartesianIndices(ntuple(_->-1:1,v))
    half_length = length(n_) ÷ 2
    n  = n_[1:half_length]

    return n
end

###=== Extract Cells
function ExtractCells!(Cells, Points, CutOff, Nmax=length(Cells))
    index  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    @inbounds for i = index:stride:Nmax
        Cells[i] =  CartesianIndex(@. Int(fld(Points[i], CutOff)) ...) + CartesianIndex(1,1) + CartesianIndex(1,1) #+ ZeroOffset + HalfPad
    end
    return nothing
end

# Actual Function to Call for Kernel Gradient Calc
function KernelExtractCells!(Cells, Points, CutOff, Nmax=length(Cells))
    kernel  = @cuda launch=false ExtractCells!(Cells, Points, CutOff, Nmax)
    config  = launch_configuration(kernel.fun)
    threads = min(Nmax, config.threads)
    blocks  = cld(Nmax, threads)

    return (kernel,threads,blocks)
end

###===

###=== SimStep
function SimStep(SimConstants, i,j, Position, Kernel)
    @unpack ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η², H² = SimConstants

    xᵢⱼ  = Position[i] - Position[j]
    xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)

    if  xᵢⱼ² <= H²
        dᵢⱼ  = sqrt(xᵢⱼ²)
        q    = clamp(dᵢⱼ * h⁻¹,0.0,2.0)
        Wᵢⱼ  = αD*(1-q/2)^4*(2*q + 1)

        Kernel[i] += Wᵢⱼ
        Kernel[j] += Wᵢⱼ
    end

    
end
###===

###=== Function to update ordering
#https://cuda.juliagpu.org/stable/tutorials/performance/
function UpdateNeighbors!(Cells, SortedIndices, ParticleRanges, Position, Density, Acceleration, Velocity)
    FunctionExtractCells!(Cells,Position)

    sortperm!(SortedIndices,Cells)

    Cells         .= Cells[SortedIndices]
    Position      .= Position[SortedIndices]
    Density       .= Density[SortedIndices]     
    Acceleration  .= Acceleration[SortedIndices]
    Velocity      .= Velocity[SortedIndices]  

    ParticleRanges = [1 ; findall(.!iszero.(diff(cuCells))) .+ 1]
    CUDA.@allowscalar push!(ParticleRanges, length(Cells) + 1)

    return ParticleRanges #Optimize out in shaa Allah!
end

function KernelNeighborLoop!(SimConstants, UniqueCells, ParticleRanges, Stencil, Position, Kernel)
    kernel  = @cuda launch=false NeighborLoop!(SimConstants, UniqueCells, ParticleRanges, Stencil, Position, Kernel)
    config  = launch_configuration(kernel.fun)
    threads = min(length(UniqueCells), config.threads)
    blocks  = cld(length(UniqueCells), threads)

    return (kernel,threads,blocks)
end
###===

###=== Function to process each cell and its neighbors
#https://cuda.juliagpu.org/stable/tutorials/performance/
function NeighborLoop!(SimConstants, Cells, ParticleRanges, Stencil, Position, Kernel)
    index  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    @views UniqueCells = Cells[ParticleRanges[1:end-1]]

    Nmax   = length(UniqueCells)

    iter = index
    @inbounds while iter <= Nmax
        CellIndex = UniqueCells[iter]
        # @cuprint "CellIndex: " CellIndex[1] "," CellIndex[2]

        StartIndex = ParticleRanges[iter] 
        EndIndex   = ParticleRanges[iter+1] - 1

        # @cuprint " |> StartIndex: " StartIndex " EndIndex: " EndIndex "\n"
        for i = StartIndex:EndIndex
            for j = (i+1):EndIndex
                SimStep(SimConstants, i,j, Position, Kernel)
            end
        end
        
        for S ∈ Stencil
            SCellIndex = CellIndex + S
    
            # @cuprint "SCellIndex: " SCellIndex[1] "," SCellIndex[2] " " @cuprintln ""
    
            if SCellIndex ∈ UniqueCells
                NeighborCellIndex = findfirst(isequal(SCellIndex), UniqueCells)

                if isnothing(NeighborCellIndex)
                    continue
                end

                StartIndex_       = ParticleRanges[NeighborCellIndex] 
                EndIndex_         = ParticleRanges[NeighborCellIndex+1] - 1 

                # @cuprintln "    StartIndex_: " StartIndex_ " EndIndex_: " EndIndex_

                for i = StartIndex:EndIndex
                    for j = StartIndex_:EndIndex_
                        SimStep(SimConstants, i, j, Position, Kernel)
                    end
                end
            end
        end
        # @cuprintln ""

        iter += stride
    end
end

function KernelNeighborLoop!(SimConstants, Cells, ParticleRanges, Stencil, Position, Kernel)
    kernel  = @cuda launch=false NeighborLoop!(SimConstants, Cells, ParticleRanges, Stencil, Position, Kernel)
    config  = launch_configuration(kernel.fun)
    threads = min(length(UniqueCells), config.threads)
    blocks  = cld(length(UniqueCells), threads)

    return (kernel,threads,blocks)
end
###===

Dimensions = 2
FloatType  = Float64
FluidCSV   = "./input/still_wedge_mdbc/StillWedge_Dp0.02_Fluid.csv"
BoundCSV   = "./input/still_wedge_mdbc/StillWedge_Dp0.02_Bound.csv"

# Unpack the relevant simulation meta data
# @unpack HourGlass, SaveLocation, SimulationName, SilentOutput, ThreadsCPU = SimMetaData;

# Unpack simulation constants
SimConstantsWedge = SimulationConstants{FloatType}(c₀=42.48576250492629)
@unpack ρ₀, dx, h, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η², H = SimConstantsWedge

# Load in the fluid and boundary particles. Return these points and both data frames
# @inline is a hack here to remove all allocations further down due to uncertainty of the points type at compile time
@inline points, density_fluid, density_bound  = LoadParticlesFromCSV(Dimensions,FloatType, FluidCSV,BoundCSV)
Position = convert(Vector{SVector{Dimensions,FloatType}},points.V)
Density  = deepcopy([density_bound; density_fluid])


cuPosition      = cu(Position)
cuDensity       = cu(Density)
cuAcceleration  = similar(cuPosition)
cuVelocity      = similar(cuPosition)
cuKernel        = similar(cuDensity)

cuCells         = similar(cuPosition, CartesianIndex{Dimensions})
# ParticleRanges  = CUDA.zeros(Int,length(cuCells)+1) #+1 last cell to include as well, first cell is included in directly due to use of diff which reduces number of elements by 1!
# CUDA.@allowscalar ParticleRanges[1]   = 1
# CUDA.@allowscalar ParticleRanges[end] = length(cuCells) + 1 #Have to add 1 even though it is wrong due to -1 at EndIndex
SortedIndices   = similar(cuCells, Int)

Stencil         = cu(ConstructStencil(Val(Dimensions)))

# Ensure zero, similar does not!
ResetArrays!(cuAcceleration, cuVelocity, cuKernel, cuCells, cuRanges, SortedIndices)

###= Preallocate functions and sizes for GPU exec
FuncExtractCells!, ThreadsExtractCells!, BlocksExtractCells! = KernelExtractCells!(cuCells,cuPosition,H)
FunctionExtractCells!(cuCells,cuPosition) = @cuda threads=ThreadsExtractCells! blocks=2  ExtractCells!(cuCells,cuPosition,H)
###=

FuncNeighborLoop!, ThreadsNeighborLoop!, BlocksNeighborLoop! = KernelNeighborLoop!(SimConstantsWedge, cuCells, ParticleRanges, Stencil,  cuPosition, cuKernel)
# More than two blocks leads to a race-condition
FunctionNeighborLoop!(SimConstants, Cells, ParticleRanges, Stencil, Position, Kernel) = @cuda threads=ThreadsNeighborLoop! blocks=2  NeighborLoop!(SimConstants, Cells, ParticleRanges, Stencil, Position, Kernel)

ParticleRanges  = UpdateNeighbors!(cuCells, SortedIndices, ParticleRanges, cuPosition, cuDensity, cuAcceleration, cuVelocity)
FunctionNeighborLoop!(SimConstantsWedge, cuCells, ParticleRanges, Stencil, cuPosition, cuKernel)

to_3d(vec_2d) = [SVector(v..., 0.0) for v in vec_2d]
PolyDataTemplate("E:/GPU_SPH/TESTING/Test" * "_" * lpad(0,6,"0") * ".vtp", to_3d(Array(cuPosition)), ["Kernel"], Array(cuKernel))