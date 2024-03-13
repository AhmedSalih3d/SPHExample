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


###=== Extract Cells

function ExtractCells!(cells, p,R,::Val{d}) where d

    @inbounds @batch for i ∈ eachindex(p)
        vs = @. Int(fld(p[i],R))
        cells[i] = tuple(vs...)
    end

    return cells
end

function ExtractCells!(Cells, Points, Nmax=length(Cells))
    index  = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    @inbounds for i = index:stride:Nmax-1
        Cells[i+1] =  Cells[i]
    end
    return nothing
end

# Actual Function to Call for Kernel Gradient Calc
function KernelExtractCells!(Cells, Points, Nmax=length(Cells))
    kernel  = @cuda launch=false ExtractCells!(Cells, Points, Nmax)
    config  = launch_configuration(kernel.fun)
    threads = min(Nmax, config.threads)
    blocks  = cld(Nmax, threads)

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
# @unpack ρ₀, dx, h, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η² = SimConstants

# Load in the fluid and boundary particles. Return these points and both data frames
# @inline is a hack here to remove all allocations further down due to uncertainty of the points type at compile time
@inline points, density_fluid, density_bound  = LoadParticlesFromCSV(Dimensions,FloatType, FluidCSV,BoundCSV)
Position = convert(Vector{SVector{Dimensions,FloatType}},points.V)
Density  = deepcopy([density_bound; density_fluid])


cuPosition     = cu(Position)
cuDensity      = cu(Density)
cuAcceleration = similar(cuPosition)
cuVelocity     = similar(cuPosition)

cuCells        = similar(cuPosition, CartesianIndex{Dimensions})

###= Preallocate functions and sizes for GPU exec
FuncExtractCells!, ThreadsExtractCells!, BlocksExtractCells! = KernelExtractCells!(cuCells,cuPosition)
FunctionExtractCells!(cuCells,cuPosition) = @cuda threads=ThreadsExtractCells! blocks=BlocksExtractCells!  ExtractCells!(cuCells,cuPosition)
###=

FunctionExtractCells!(cuCells,cuPosition)
println(cuCells)