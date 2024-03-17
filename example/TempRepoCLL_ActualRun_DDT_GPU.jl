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
CUDA.allowscalar(false)

import Base.Threads: nthreads, @threads
include("../src/ProduceVTP.jl")

const MaxThreads = attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)


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

    CUDA.@sync kernel(Cells, Points, CutOff, Nmax; threads, blocks)
end

###===

###=== SimStep
@inline function dρᵢdt_dρⱼdt_(ρᵢ,ρⱼ,m₀,vᵢⱼ,∇ᵢWᵢⱼ)
    #original: - ρᵢ * dot((m₀/ρⱼ) *  -vᵢⱼ ,  ∇ᵢWᵢⱼ)
    symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ) # = dot(vᵢⱼ , -∇ᵢWᵢⱼ)
    dρdt⁺          = - ρᵢ * (m₀/ρⱼ) *  symmetric_term
    dρdt⁻          = - ρⱼ * (m₀/ρᵢ) *  symmetric_term

    return dρdt⁺, dρdt⁻
end

@fastpow function EquationOfStateGamma7(ρ,c₀,ρ₀)
    return ((c₀^2*ρ₀)/7) * ((ρ/ρ₀)^7 - 1)
end


function SimStep(SimConstants, i,j, Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, SharedMemory, SharedIndex)
    @unpack ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η², H² = SimConstants

    xᵢⱼ  = Position[i] - Position[j]
    xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)

    if  xᵢⱼ² <= H²
        dᵢⱼ  = sqrt(xᵢⱼ²) #Using sqrt is what takes a lot of time?
        q    = clamp(dᵢⱼ * h⁻¹,0.0,2.0)
        Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)

        invd²η² = inv(dᵢⱼ*dᵢⱼ+η²) 

        Kernel[i] += Wᵢⱼ
        Kernel[j] += Wᵢⱼ

        ∇ᵢWᵢⱼ = @fastpow (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 

        KernelGradient[i] +=  ∇ᵢWᵢⱼ
        KernelGradient[j] += -∇ᵢWᵢⱼ

        # ρᵢ        = Density[i]
        # ρⱼ        = Density[j]
      
        # vᵢ        = Velocity[i]
        # vⱼ        = Velocity[j]
        # vᵢⱼ       = vᵢ - vⱼ

        # dρdt⁺, dρdt⁻ = dρᵢdt_dρⱼdt_(ρᵢ,ρⱼ,m₀,vᵢⱼ,∇ᵢWᵢⱼ)

        # Pᵢ        =  EquationOfStateGamma7(ρᵢ,c₀,ρ₀)
        # Pⱼ        =  EquationOfStateGamma7(ρⱼ,c₀,ρ₀)
        # Pfac      = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)

        # ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
        # cond      = dot(vᵢⱼ, xᵢⱼ)
        # cond_bool = cond < 0.0
        # μᵢⱼ       = h*cond * invd²η²
        # Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
        # Πⱼ        = - Πᵢ

        # dvdt⁺ = - m₀ * (Pfac) *  ∇ᵢWᵢⱼ + Πᵢ
        # dvdt⁻ = - dvdt⁺ + Πⱼ

        # dvdtI[i] += dvdt⁺
        # dvdtI[j] += dvdt⁻

        # dρdtI[i] += dρdt⁺
        # dρdtI[j] += dρdt⁻
    end

    return nothing
end
###===

###=== Function to update ordering
#https://cuda.juliagpu.org/stable/tutorials/performance/
function UpdateNeighbors!(Cells, CutOff, SortedIndices, Position, Density, Acceleration, Velocity, DiffCells)
    KernelExtractCells!(Cells,Position,CutOff)

    sortperm!(SortedIndices,Cells)

    @. Cells           =  Cells[SortedIndices]
    @. Position        =  Position[SortedIndices]
    @. Density         =  Density[SortedIndices]
    @. Acceleration    =  Acceleration[SortedIndices]
    @. Velocity        =  Velocity[SortedIndices]

    DiffCells         .= diff(Cells)
    ParticleRanges     = [cu([1]) ; findall(.!iszero.(DiffCells)) .+ 1; cu([length(Cells) + 1])]

    # Passing the view is fine, since it is not needed to actualize the vector
    @views UniqueCells        = Cells[ParticleRanges[1:end-1]]

    return ParticleRanges, UniqueCells #Optimize out in shaa Allah!
end
###===

###=== Function to process each cell and its neighbors
#https://cuda.juliagpu.org/stable/tutorials/performance/
# 192 bytes and 4 allocs from launch config
# INLINE IS SO IMPORTANT 10X SPEED
function NeighborLoop!(SimConstants, UniqueCells, ParticleRanges, Stencil, Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI)
    index  = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    iter = index

    @inbounds while iter <= length(UniqueCells)
        CellIndex = UniqueCells[iter]
        # @cuprint "CellIndex: " CellIndex[1] "," CellIndex[2]

        StartIndex = ParticleRanges[iter] 
        EndIndex   = ParticleRanges[iter+1] - 1


        n = EndIndex - StartIndex + 1
        to_alloc = Int(ceil((n * (n - 1)) / 2))

        # SharedMemory = CuDynamicSharedArray(Float64, to_alloc)
        SharedMemory   = @cuDynamicSharedMem(eltype(eltype(Position)), 1) #to_alloc)

        # @cuprint " |> StartIndex: " StartIndex " EndIndex: " EndIndex "\n"
        @inbounds for i = StartIndex:EndIndex, j = (i+1):EndIndex
            SimStep(SimConstants, i,j, Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, SharedMemory, 0)
        end
        
        @inbounds for S ∈ Stencil
            SCellIndex = CellIndex + S
    
            # @cuprint "SCellIndex: " SCellIndex[1] "," SCellIndex[2] " " @cuprintln ""
    
            # This is weirdly enough 5-10% faster than isequal approach bu still allocates?
            # Needle = isequal(SCellIndex) #This allocates 8 bytes :(
            # NeighborCellIndex = findfirst(Needle, UniqueCells)
            c = 0
            NeighborCellIndex = 0
            @inbounds for i ∈ eachindex(UniqueCells)
                c += 1
                if LinearIndices(Tuple(UniqueCells[end]))[SCellIndex] == LinearIndices(Tuple(UniqueCells[end]))[UniqueCells[i]]
                # if SCellIndex == UniqueCells[i]
                    NeighborCellIndex = c
                    # break - in reality I should break, but letting it run fully is faster, incredible
                end
            end

            # NeighborCellIndex = argmax(UniqueCells .== SCellIndex)

            # if !isnothing(NeighborCellIndex)
            if !iszero(NeighborCellIndex)
                StartIndex_       = ParticleRanges[NeighborCellIndex] 
                EndIndex_         = ParticleRanges[NeighborCellIndex+1] - 1

                # @cuprintln "    StartIndex_: " StartIndex_ " EndIndex_: " EndIndex_

                n_        = EndIndex_ - StartIndex_ + 1
                to_alloc_ = Int(ceil((n_ * (n_ - 1)) / 2))
        
                # As explained by ChatGPT this should be intra particle interactions and then outside * inside to account for it all
                SharedMemory_ = CuDynamicSharedArray(Float64, n * n_ + to_alloc_)

                @inbounds for i = StartIndex:EndIndex, j = StartIndex_:EndIndex_
                    SimStep(SimConstants, i, j, Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, SharedMemory_, 0)
                end

            end
        end
        # @cuprintln ""

        iter += stride
    end

    return nothing
end

function ThreadsAndBlocksNeighborLoop(SimConstants, UniqueCells, ParticleRanges, Stencil, Position, Kernel, KernelGradient, Density, Velocity, cudρdtI, cudvdtI)
    kernel  = @cuda launch=false NeighborLoop!(SimConstants, UniqueCells, ParticleRanges, Stencil, Position, Kernel, KernelGradient, Density, Velocity, cudρdtI, cudvdtI)
    config  = launch_configuration(kernel.fun)
    threads = min(length(UniqueCells), config.threads)
    blocks  = cld(length(UniqueCells), threads)

    return threads, blocks
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



cuPosition       = cu(Position)
cuDensity        = cu(Density)
cuAcceleration   = similar(cuPosition)
cuVelocity       = similar(cuPosition)
cuKernel         = similar(cuDensity)
cuKernelGradient = similar(cuPosition)

cudρdtI          = similar(cuDensity)
cudvdtI          = similar(cuPosition)

cuCells          = similar(cuPosition, CartesianIndex{Dimensions})
cuDiffCells      = cu(zeros(CartesianIndex{Dimensions},length(cuCells)-1))
cuParticleRanges = cu(zeros(Int, length(cuCells) + 1))
# ParticleRanges  = CUDA.zeros(Int,length(cuCells)+1) #+1 last cell to include as well, first cell is included in directly due to use of diff which reduces number of elements by 1!
# CUDA.@allowscalar ParticleRanges[1]   = 1
# CUDA.@allowscalar ParticleRanges[end] = length(cuCells) + 1 #Have to add 1 even though it is wrong due to -1 at EndIndex
SortedIndices   = similar(cuCells, Int)

Stencil         = cu(ConstructStencil(Val(Dimensions)))

# Ensure zero, similar does not!
ResetArrays!(cuAcceleration, cuVelocity, cuKernel, cuKernelGradient, cuCells, SortedIndices, cudρdtI, cudvdtI)


# Normal run and save data
ParticleRanges, UniqueCells = UpdateNeighbors!(cuCells, H, SortedIndices, cuPosition, cuDensity, cuAcceleration, cuVelocity, cuDiffCells)
threads1,blocks1 = ThreadsAndBlocksNeighborLoop(SimConstantsWedge, UniqueCells, ParticleRanges, Stencil, cuPosition, cuKernel, cuKernelGradient, cuDensity, cuVelocity, cudρdtI, cudvdtI)
@cuda always_inline=true fastmath=true threads=threads1 blocks=blocks1 shmem = 100*sizeof(FloatType) NeighborLoop!(SimConstantsWedge, UniqueCells, ParticleRanges, Stencil, cuPosition, cuKernel, cuKernelGradient, cuDensity, cuVelocity, cudρdtI, cudvdtI)
SimMetaData  = SimulationMetaData{Dimensions,FloatType}(
    SimulationName="Test", 
    SaveLocation="E:/GPU_SPH/TESTING/",
)
KERNEL = Array(cuKernel)
KERNEL_GRADIENT = Array(cuKernelGradient)
to_3d(vec_2d) = [SVector(v..., 0.0) for v in vec_2d]
create_vtp_file(SimMetaData, SimConstantsWedge, to_3d(Array(cuPosition)); KERNEL, KERNEL_GRADIENT)
#

println(CUDA.@profile trace=true ParticleRanges, UniqueCells  = UpdateNeighbors!(cuCells, H, SortedIndices, cuPosition, cuDensity, cuAcceleration, cuVelocity,  cuDiffCells))
println(CUDA.@profile trace=true @cuda always_inline=true fastmath=true threads=threads1 blocks=blocks1 NeighborLoop!(SimConstantsWedge, UniqueCells, ParticleRanges, Stencil, cuPosition, cuKernel, cuKernelGradient, cuDensity, cuVelocity, cudρdtI, cudvdtI))

display(@benchmark CUDA.@sync ParticleRanges, UniqueCells  = UpdateNeighbors!($cuCells, $H, $SortedIndices, $cuPosition, $cuDensity, $cuAcceleration, $cuVelocity,  $cuDiffCells))
display(@benchmark CUDA.@sync @cuda always_inline=true fastmath=true threads=$threads1 blocks=$blocks1 NeighborLoop!($SimConstantsWedge, $UniqueCells, $ParticleRanges, $Stencil, $cuPosition, $cuKernel, $cuKernelGradient, $cuDensity, $cuVelocity, $cudρdtI, $cudvdtI))
println("CUDA allocations: ", CUDA.@allocated ParticleRanges, UniqueCells  = UpdateNeighbors!(cuCells, H, SortedIndices, cuPosition, cuDensity, cuAcceleration, cuVelocity,  cuDiffCells))
println("CUDA allocations: ", CUDA.@allocated @cuda always_inline=true fastmath=true threads=threads1 blocks=blocks1 NeighborLoop!(SimConstantsWedge, UniqueCells, ParticleRanges, Stencil, cuPosition, cuKernel, cuKernelGradient, cuDensity, cuVelocity, cudρdtI, cudvdtI))

