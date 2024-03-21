using SPHExample
using BenchmarkTools
using StaticArrays
using Parameters
using StructArrays
import LinearAlgebra: dot, norm, diagm, diag, cond, det
using LoopVectorization
using FastPow
import CellListMap: InPlaceNeighborList, update!, neighborlist!
using Bumper

import Base.Threads: nthreads, @threads
include("../src/ProduceVTP.jl")

function update_arr1_bumper!(arr1,indices)
    @no_escape begin
        temp  = @alloc(eltype(arr1),length(arr1))

        temp .= @view arr1[indices]
        arr1 .= temp
        
    end
end


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
function ExtractCells!(Cells, Points, CutOff)
    for i ∈ eachindex(Cells)
        Cells[i] =  CartesianIndex(@. Int(fld(Points[i], CutOff)) ...) + CartesianIndex(1,1) + CartesianIndex(1,1) #+ ZeroOffset + HalfPad
    end
    return nothing
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


function SimStepLocalCell(SimConstants, Position, Kernel, KernelGradientX, KernelGradientY, Density, Velocity, dρdtI, dvdtIX, dvdtIY, StartIndex, EndIndex)
    @unpack ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η², H² = SimConstants

    @inbounds for i = StartIndex:EndIndex, j = (i+1):EndIndex

        xᵢⱼ  = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)

        if  xᵢⱼ² <= H²
            dᵢⱼ  = sqrt(xᵢⱼ²) #Using sqrt is what takes a lot of time?
            q    = clamp(dᵢⱼ * h⁻¹,0.0,2.0)
            Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)

            invd²η² = inv(dᵢⱼ*dᵢⱼ+η²)

            ∇ᵢWᵢⱼ = @fastpow (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 

            ρᵢ        = Density[i]
            ρⱼ        = Density[j]
        
            vᵢ        = Velocity[i]
            vⱼ        = Velocity[j]
            vᵢⱼ       = vᵢ - vⱼ

            dρdt⁺, dρdt⁻ = dρᵢdt_dρⱼdt_(ρᵢ,ρⱼ,m₀,vᵢⱼ,∇ᵢWᵢⱼ)

            Pᵢ        =  EquationOfStateGamma7(ρᵢ,c₀,ρ₀)
            Pⱼ        =  EquationOfStateGamma7(ρⱼ,c₀,ρ₀)
            Pfac      = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)

            ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
            cond      = dot(vᵢⱼ, xᵢⱼ)
            cond_bool = cond < 0.0
            μᵢⱼ       = h*cond * invd²η²
            Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
            Πⱼ        = - Πᵢ

            dvdt⁺ = - m₀ * (Pfac) *  ∇ᵢWᵢⱼ + Πᵢ
            dvdt⁻ = - dvdt⁺ + Πⱼ

            dρdtI[i] += dρdt⁺
            dρdtI[j] += dρdt⁻

            dvdtIX[i] +=  dvdt⁺[1]
            dvdtIX[j] +=  dvdt⁻[1]
            dvdtIY[i] +=  dvdt⁺[2]
            dvdtIY[j] +=  dvdt⁻[2]

            Kernel[i] += Wᵢⱼ
            Kernel[j] += Wᵢⱼ

            KernelGradientX[i] +=  ∇ᵢWᵢⱼ[1]
            KernelGradientX[j] += -∇ᵢWᵢⱼ[1]
            KernelGradientY[i] +=  ∇ᵢWᵢⱼ[2]
            KernelGradientY[j] += -∇ᵢWᵢⱼ[2]
        end
    end

    return nothing
end


function SimStepNeighborCell(SimConstants, Position, Kernel, KernelGradientX, KernelGradientY, Density, Velocity, dρdtI, dvdtIX, dvdtIY, StartIndex, EndIndex, StartIndex_, EndIndex_)
    @unpack ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η², H² = SimConstants

    @inbounds for i = StartIndex:EndIndex, j = StartIndex_:EndIndex_

        xᵢⱼ  = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)

        if  xᵢⱼ² <= H²
            dᵢⱼ  = sqrt(xᵢⱼ²) #Using sqrt is what takes a lot of time?
            q    = clamp(dᵢⱼ * h⁻¹,0.0,2.0)
            Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)

            invd²η² = inv(dᵢⱼ*dᵢⱼ+η²)

            ∇ᵢWᵢⱼ = @fastpow (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 

            ρᵢ        = Density[i]
            ρⱼ        = Density[j]
        
            vᵢ        = Velocity[i]
            vⱼ        = Velocity[j]
            vᵢⱼ       = vᵢ - vⱼ

            dρdt⁺, dρdt⁻ = dρᵢdt_dρⱼdt_(ρᵢ,ρⱼ,m₀,vᵢⱼ,∇ᵢWᵢⱼ)

            Pᵢ        =  EquationOfStateGamma7(ρᵢ,c₀,ρ₀)
            Pⱼ        =  EquationOfStateGamma7(ρⱼ,c₀,ρ₀)
            Pfac      = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)

            ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
            cond      = dot(vᵢⱼ, xᵢⱼ)
            cond_bool = cond < 0.0
            μᵢⱼ       = h*cond * invd²η²
            Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
            Πⱼ        = - Πᵢ

            dvdt⁺ = - m₀ * (Pfac) *  ∇ᵢWᵢⱼ + Πᵢ
            dvdt⁻ = - dvdt⁺ + Πⱼ

            dρdtI[i] += dρdt⁺
            dρdtI[j] += dρdt⁻

            dvdtIX[i] +=  dvdt⁺[1]
            dvdtIX[j] +=  dvdt⁻[1]
            dvdtIY[i] +=  dvdt⁺[2]
            dvdtIY[j] +=  dvdt⁻[2]

            Kernel[i] += Wᵢⱼ
            Kernel[j] += Wᵢⱼ

            KernelGradientX[i] +=  ∇ᵢWᵢⱼ[1]
            KernelGradientX[j] += -∇ᵢWᵢⱼ[1]
            KernelGradientY[i] +=  ∇ᵢWᵢⱼ[2]
            KernelGradientY[j] += -∇ᵢWᵢⱼ[2]
        end
    end

    return nothing
end

###===

###=== Function to update ordering
#https://cuda.juliagpu.org/stable/tutorials/performance/
function UpdateNeighbors!(Cells, CutOff, SortedIndices, Position, Density, Acceleration, Velocity, ParticleSplitter, ParticleSplitterLinearIndices)
    ExtractCells!(Cells,Position,CutOff)

    sortperm!(SortedIndices,Cells)

    # @. Cells           =  Cells[SortedIndices]
    # @. Position        =  Position[SortedIndices]
    # @. Density         =  Density[SortedIndices]
    # @. Acceleration    =  Acceleration[SortedIndices]
    # @. Velocity        =  Velocity[SortedIndices]

    update_arr1_bumper!(Cells, SortedIndices)
    update_arr1_bumper!(Position, SortedIndices)
    update_arr1_bumper!(Density, SortedIndices)
    update_arr1_bumper!(Acceleration, SortedIndices)
    update_arr1_bumper!(Velocity, SortedIndices)    

    ParticleSplitter[findall(.!iszero.(diff(Cells))) .+ 1] .= true

    # # Passing the view is fine, since it is not needed to actualize the vector
    @views ParticleRanges     = ParticleSplitterLinearIndices[ParticleSplitter]
    @views UniqueCells        = Cells[ParticleRanges[1:end-1]]

    return ParticleRanges, UniqueCells #Optimize out in shaa Allah!
end
###===

###=== Function to process each cell and its neighbors
#https://cuda.juliagpu.org/stable/tutorials/performance/
# 192 bytes and 4 allocs from launch config
# INLINE IS SO IMPORTANT 10X SPEED
function NeighborLoop!(SimConstants, UniqueCells, ParticleRanges, Stencil, Position, Kernel, KernelGradientX, KernelGradientY, Density, Velocity, dρdtI, dvdtIX, dvdtIY)
    for iter ∈ eachindex(UniqueCells)
        CellIndex = UniqueCells[iter]

        StartIndex = ParticleRanges[iter] 
        EndIndex   = ParticleRanges[iter+1] - 1

        SimStepLocalCell(SimConstants, Position, Kernel, KernelGradientX, KernelGradientY, Density, Velocity, dρdtI, dvdtIX, dvdtIY, StartIndex, EndIndex)

        @inbounds for S ∈ Stencil
            SCellIndex = CellIndex + S
    
            c = 0
            NeighborCellIndex = 0
            @inbounds for i ∈ eachindex(UniqueCells)
                c += 1
                cond = LinearIndices(Tuple(UniqueCells[end]))[SCellIndex] == LinearIndices(Tuple(UniqueCells[end]))[UniqueCells[i]]
                val  = ifelse(cond, c, 0)
                NeighborCellIndex += val
            end

            if !iszero(NeighborCellIndex)
                StartIndex_       = ParticleRanges[NeighborCellIndex] 
                EndIndex_         = ParticleRanges[NeighborCellIndex+1] - 1

                SimStepNeighborCell(SimConstants, Position, Kernel, KernelGradientX, KernelGradientY, Density, Velocity, dρdtI, dvdtIX, dvdtIY, StartIndex, EndIndex, StartIndex_, EndIndex_)
            end
        end
    end

    return nothing
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

Acceleration    = similar(Position)
Velocity        = similar(Position)
Kernel          = similar(Density)
KernelGradientX = zeros(length(Density))
KernelGradientY = zeros(length(Density))

dρdtI           = similar(Density)

dvdtIX = zeros(length(Density))
dvdtIY = zeros(length(Density))

Cells                                   = similar(Position, CartesianIndex{Dimensions})
DiffCells                               = zeros(CartesianIndex{Dimensions},length(Cells)-1)
ParticleSplitter                        = zeros(Bool, length(Cells) + 1)
# ParticleRanges  = CUDA.zeros(Int,length(cuCells)+1) #+1 last cell to include as well, first cell is included in directly due to use of diff which reduces number of elements by 1!
ParticleSplitter[1]   = true
ParticleSplitter[end] = true #Have to add 1 even though it is wrong due to -1 at EndIndex, length + 1

ParticleSplitterLinearIndices = LinearIndices(ParticleSplitter)

SortedIndices   = similar(Cells, Int)

Stencil         = ConstructStencil(Val(Dimensions))

# Ensure zero, similar does not!
ResetArrays!(Acceleration, Velocity, Kernel, KernelGradientX, KernelGradientY, Cells, SortedIndices, dρdtI, dvdtIX, dvdtIY)



# Normal run and save data
ParticleRanges, UniqueCells = UpdateNeighbors!(Cells, H, SortedIndices, Position, Density, Acceleration, Velocity, ParticleSplitter, ParticleSplitterLinearIndices)  
NeighborLoop!(SimConstantsWedge, UniqueCells, ParticleRanges, Stencil, Position, Kernel, KernelGradientX, KernelGradientY, Density, Velocity, dρdtI, dvdtIX, dvdtIY)
SimMetaData  = SimulationMetaData{Dimensions,FloatType}(
    SimulationName="Test", 
    SaveLocation="E:/SecondApproach/TESTING_CPU",
)
KernelGradient = [KernelGradientX'; KernelGradientY']
to_3d(vec_2d) = [SVector(v..., 0.0) for v in vec_2d]
create_vtp_file(SimMetaData, SimConstantsWedge, to_3d(Position); Kernel, KernelGradient)

# println(CUDA.@profile trace=true @cuda always_inline=true fastmath=true threads=threads1 blocks=blocks1 shmem=shmem NeighborLoop!(SimConstantsWedge, UniqueCells, ParticleRanges, Stencil, cuPosition, cuKernel, cuKernelGradientX, cuKernelGradientY, cuDensity, cuVelocity, cudρdtI, cudvdtIX, cudvdtIY))

display(@benchmark ParticleRanges,UniqueCells     = UpdateNeighbors!($Cells, $H, $SortedIndices, $Position, $Density, $Acceleration, $Velocity, $ParticleSplitter, $ParticleSplitterLinearIndices))
display(@benchmark NeighborLoop!($SimConstantsWedge, $UniqueCells, $ParticleRanges, $Stencil, $Position, $Kernel, $KernelGradientX, $KernelGradientY, $Density, $Velocity, $dρdtI, $dvdtIX, $dvdtIY))
# println("CUDA allocations: ", CUDA.@allocated ParticleRanges, UniqueCells  = UpdateNeighbors!(cuCells, H, SortedIndices, cuPosition, cuDensity, cuAcceleration, cuVelocity, cuParticleSplitter, cuParticleSplitterLinearIndices))
# println("CUDA allocations: ", CUDA.@allocated @cuda always_inline=true fastmath=true threads=threads1 blocks=blocks1 shmem = shmem NeighborLoop!(SimConstantsWedge, UniqueCells, ParticleRanges, Stencil, cuPosition, cuKernel, cuKernelGradientX, cuKernelGradientY, cuDensity, cuVelocity, cudρdtI, cudvdtIX, cudvdtIY))

