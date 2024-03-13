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
function SimStep(SimConstants, i,j, CutOffSquared, Position, Kernel)
    @unpack ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η² = SimConstants

    xᵢⱼ  = Position[i] - Position[j]
    xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)

    if  xᵢⱼ² <= CutOffSquared
        dᵢⱼ  = sqrt(xᵢⱼ²)
        q    = clamp(dᵢⱼ * h⁻¹,0.0,2.0)
        Wᵢⱼ  = αD*(1-q/2)^4*(2*q + 1)

        Kernel[i] += Wᵢⱼ
        Kernel[j] += Wᵢⱼ
    end

    
end
###===

###=== Function to process each cell and its neighbors
#https://cuda.juliagpu.org/stable/tutorials/performance/
function NeighborLoop!(SimConstants, UniqueCells, ParticleRanges, Stencil, CutOffSquared, Position, Kernel)
    index  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    Nmax   = length(UniqueCells) - 1
    
    iter = index
    @inbounds while iter <= Nmax
        CellIndex = UniqueCells[iter]
        @cuprint "CellIndex: " CellIndex[1] "," CellIndex[2]

        StartIndex = ParticleRanges[iter] 
        EndIndex   = ParticleRanges[iter+1] - 1

        @cuprint " |> StartIndex: " StartIndex " EndIndex: " EndIndex "\n"
        for i = StartIndex:EndIndex
            for j = StartIndex:EndIndex
                if i != j
                    SimStep(SimConstants, i,j, CutOffSquared, Position, Kernel)
                end
            end
        end
        
        for S ∈ Stencil
            SCellIndex = CellIndex + S
    
            @cuprint "SCellIndex: " SCellIndex[1] "," SCellIndex[2] " "
            @cuprintln ""
    
            if SCellIndex ∈ UniqueCells
                NeighborCellIndex = findfirst(isequal(SCellIndex), UniqueCells)

                if isnothing(NeighborCellIndex)
                    continue
                end

                StartIndex_       = ParticleRanges[NeighborCellIndex] 
                EndIndex_         = ParticleRanges[NeighborCellIndex+1] - 1

                @cuprintln "    StartIndex_: " StartIndex_ " EndIndex_: " EndIndex_
                for i = StartIndex:EndIndex
                    for j = StartIndex_:EndIndex_
                        if i != j
                            SimStep(SimConstants, i, j, CutOffSquared, Position, Kernel)
                        end
                    end
                end
            end
        end
        @cuprintln ""

        iter += stride
    end
end

function KernelNeighborLoop!(SimConstants, UniqueCells, ParticleRanges, Stencil, CutOffSquared, Position, Kernel)
    kernel  = @cuda launch=false NeighborLoop!(SimConstants, UniqueCells, ParticleRanges, Stencil, CutOffSquared, Position, Kernel)
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
@unpack ρ₀, dx, h, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η² = SimConstantsWedge
CutOff = 2h
CutOffSquared = CutOff^2

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
cuRanges        = similar(cuCells, Int)
p               = similar(cuCells, Int)

Stencil         = cu(ConstructStencil(Val(Dimensions)))


# Ensure zero, similar does not!
ResetArrays!(cuAcceleration, cuVelocity, cuKernel, cuCells, cuRanges)

###= Preallocate functions and sizes for GPU exec
FuncExtractCells!, ThreadsExtractCells!, BlocksExtractCells! = KernelExtractCells!(cuCells,cuPosition,CutOff)
FunctionExtractCells!(cuCells,cuPosition) = @cuda threads=ThreadsExtractCells! blocks=BlocksExtractCells!  ExtractCells!(cuCells,cuPosition,CutOff)
###=

FunctionExtractCells!(cuCells,cuPosition)

sortperm!(p,cuCells)

cuCells         .= cuCells[p]
cuPosition      .= cuPosition[p]
cuDensity       .= cuDensity[p]     
cuAcceleration  .= cuAcceleration[p]
cuVelocity      .= cuVelocity[p]  

cuRanges[1:1]   .= 1
cuRanges[2:end] .= .!iszero.(diff(cuCells))

ParticleRanges = [1;findall(.!iszero.(diff(cuCells))) .+ 1] #This works but not findall on cuRanges
UniqueCells    = cuCells[ParticleRanges]


FuncNeighborLoop!, ThreadsNeighborLoop!, BlocksNeighborLoop! = KernelNeighborLoop!(SimConstantsWedge, UniqueCells, ParticleRanges, Stencil, CutOffSquared, cuPosition, cuKernel)
FunctionNeighborLoop!(SimConstants, UniqueCells, ParticleRanges, Stencil, CutOffSquared, Position, Kernel) = @cuda threads=ThreadsNeighborLoop!  blocks=BlocksNeighborLoop!  NeighborLoop!(SimConstants, UniqueCells, ParticleRanges, Stencil, CutOffSquared, Position, Kernel)
FunctionNeighborLoop!(SimConstantsWedge, UniqueCells, ParticleRanges, Stencil, CutOffSquared, cuPosition, cuKernel)

to_3d(vec_2d) = [SVector(v..., 0.0) for v in vec_2d]
PolyDataTemplate("E:/GPU_SPH/TESTING/Test" * "_" * lpad(0,6,"0") * ".vtp", to_3d(Array(cuPosition)), ["Kernel"], Array(cuKernel))

# CUDA.@allowscalar for iter = 5#1:length(UniqueCells) - 1
#     CellIndex = UniqueCells[iter]
#     @cuprintln "CellIndex: " CellIndex[1] "," CellIndex[2]

#     PR        = ParticleRanges[iter:iter+1]
#     PR[2]    -= 1 #Non inclusive range
#     # Particles in sorted Cells
#     @cuprintln "ParticleRanges: " PR[1] ":" PR[2]
 
#     for S ∈ Stencil
#         SCellIndex = CellIndex + S

#         @cuprintln "SCellIndex: " SCellIndex[1] "," SCellIndex[2]

#         if SCellIndex ∈ UniqueCells
#             NeighborCellIndex = findfirst(isequal(SCellIndex), UniqueCells)
#             PR_     = ParticleRanges[NeighborCellIndex:iter+1]
#             PR_[2] -= 1 #Non inclusive range
#             @cuprintln "    ParticleRanges: " PR_[1] ":" PR_[2]
#         end
#     end
# end