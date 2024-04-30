using SPHExample
using StructArrays
using CUDA
using Parameters
using FastPow
import LinearAlgebra: dot, norm
using HDF5
using TimerOutputs
using StaticArrays


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
    SimulationName="StillWedge3", 
    SaveLocation="E:/SecondApproach/StillWedge_GPU",
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


SortedIndices = CUDA.zeros(Int,NumberOfPoints)

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

function UpdateNeighbours!(Particles, SortedIndices, CutOff)
    launch_ExtractCellsKernel!(Particles, CutOff)

    sortperm!(SortedIndices, Particles.Cells)

    for prop in propertynames(Particles)
        getproperty(Particles,prop) .= getproperty(Particles, prop)[SortedIndices]
    end


    ParticleRanges .= 0

    ParticleRangesIndices = findall(diff(Particles.Cells) .!= CartesianIndex(0,0))
    
    ParticleRanges[1:1] = 1
    ParticleRanges[2:(length(ParticleRangesIndices)+1)] .= ParticleRangesIndices .+ 1
    ParticleRanges[end:end] = length(ParticleRanges)
    sort!(ParticleRanges, lt=zero_last_comparator)

    IndexCounter = findfirst(isequal(0), ParticleRanges) - 2

    return IndexCounter
end


IndexCounter = UpdateNeighbours!(SimParticles_GPU, SortedIndices, CutOff)

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

# ComputeInteractions
    function ComputeInteractionsGPU!(Particles, SimConstants, dρdtI, i, j)
        # @unpack FlagViscosityTreatment, FlagDensityDiffusion, FlagOutputKernelValues = SimMetaData
        @unpack ρ₀, h, h⁻¹, m₀, αD, α, g, c₀, δᵩ, η², H², Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants

        xᵢⱼ  = Particles.Position[i] - Particles.Position[j]
        xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)              
        if  xᵢⱼ² <= H²
            #https://discourse.julialang.org/t/sqrt-abs-x-is-even-faster-than-sqrt/58154/2
            dᵢⱼ  = sqrt(abs(xᵢⱼ²))

            q    = min(dᵢⱼ * h⁻¹, 2.0)
            invd²η²   =  1.0 / (dᵢⱼ*dᵢⱼ+η²)
            ∇ᵢWᵢⱼ     = @fastpow (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 
            ρᵢ        = Particles.Density[i]
            ρⱼ        = Particles.Density[j]
        
            vᵢ        = Particles.Velocity[i]
            vⱼ        = Particles.Velocity[j]
            vᵢⱼ       = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺          = - ρᵢ * (m₀/ρⱼ) *  density_symmetric_term
            dρdt⁻          = - ρⱼ * (m₀/ρᵢ) *  density_symmetric_term

            # Density diffusion
            if true #FlagDensityDiffusion
                if SimConstants.g == 0
                    ρᵢⱼᴴ  = 0.0
                    ρⱼᵢᴴ  = 0.0
                else
                    Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
                    ρᵢⱼᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pᵢⱼᴴ, Cb⁻¹)
                    Pⱼᵢᴴ  = -Pᵢⱼᴴ
                    ρⱼᵢᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pⱼᵢᴴ, Cb⁻¹)
                end

                ρⱼᵢ   = ρⱼ - ρᵢ

                Ψᵢⱼ   = 2( ρⱼᵢ  - ρᵢⱼᴴ) * (-xᵢⱼ) * invd²η²
                Ψⱼᵢ   = 2(-ρⱼᵢ  - ρⱼᵢᴴ) * ( xᵢⱼ) * invd²η²

                MLcond = Particles.MotionLimiter[i] * Particles.MotionLimiter[j]
                Dᵢ    =  δᵩ * h * c₀ * (m₀/ρⱼ) * dot(Ψᵢⱼ ,  ∇ᵢWᵢⱼ) * MLcond
                Dⱼ    =  δᵩ * h * c₀ * (m₀/ρᵢ) * dot(Ψⱼᵢ , -∇ᵢWᵢⱼ) * MLcond
            else
                Dᵢ  = 0.0
                Dⱼ  = 0.0
            end
            CUDA.@atomic dρdtI[i] += dρdt⁺ + Dᵢ
            CUDA.@atomic dρdtI[j] += dρdt⁻ + Dⱼ


            Pᵢ      =  Particles.Pressure[i]
            Pⱼ      =  Particles.Pressure[j]
            Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
            dvdt⁺   = - m₀ * Pfac *  ∇ᵢWᵢⱼ
            dvdt⁻   = - dvdt⁺

            # if FlagViscosityTreatment == :ArtificialViscosity
                ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
                cond      = dot(vᵢⱼ, xᵢⱼ)
                cond_bool = cond < 0.0
                μᵢⱼ       = h*cond * invd²η²
                Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
                Πⱼ        = - Πᵢ
            # else
            #     Πᵢ        = zero(xᵢⱼ)
            #     Πⱼ        = Πᵢ
            # end
        
        #     # if FlagViscosityTreatment == :Laminar || FlagViscosityTreatment == :LaminarSPS
        #     #     # 4 comes from 2 divided by 0.5 from average density
        #     #     # should divide by ρᵢ eq 6 DPC
        #     #     # ν₀∇²uᵢ = (1/ρᵢ) * ( (4 * m₀ * (ρᵢ * ν₀) * dot( xᵢⱼ, ∇ᵢWᵢⱼ)  ) / ( (ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²) ) ) *  vᵢⱼ
        #     #     # ν₀∇²uⱼ = (1/ρⱼ) * ( (4 * m₀ * (ρⱼ * ν₀) * dot(-xᵢⱼ,-∇ᵢWᵢⱼ)  ) / ( (ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²) ) ) * -vᵢⱼ
        #     #     visc_symmetric_term = (4 * m₀ * ν₀ * dot( xᵢⱼ, ∇ᵢWᵢⱼ)) / ((ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²))
        #     #     # ν₀∇²uᵢ = (1/ρᵢ) * visc_symmetric_term *  vᵢⱼ * ρᵢ
        #     #     # ν₀∇²uⱼ = (1/ρⱼ) * visc_symmetric_term * -vᵢⱼ * ρⱼ
        #     #     ν₀∇²uᵢ =  visc_symmetric_term *  vᵢⱼ
        #     #     ν₀∇²uⱼ = -ν₀∇²uᵢ #visc_symmetric_term * -vᵢⱼ
        #     # else
        #     #     ν₀∇²uᵢ = zero(xᵢⱼ)
        #     #     ν₀∇²uⱼ = ν₀∇²uᵢ
        #     # end
        
        #     # if FlagViscosityTreatment == :LaminarSPS 
        #     #     Iᴹ       = diagm(one.(xᵢⱼ))
        #     #     #julia> a .- a'
        #     #     # 3×3 SMatrix{3, 3, Float64, 9} with indices SOneTo(3)×SOneTo(3):
        #     #     # 0.0  0.0  0.0
        #     #     # 0.0  0.0  0.0
        #     #     # 0.0  0.0  0.0
        #     #     # Strain *rate* tensor is the gradient of velocity
        #     #     Sᵢ = ∇vᵢ =  (m₀/ρⱼ) * (vⱼ - vᵢ) * ∇ᵢWᵢⱼ'
        #     #     norm_Sᵢ  = sqrt(2 * sum(Sᵢ .^ 2))
        #     #     νtᵢ      = (SmagorinskyConstant * dx)^2 * norm_Sᵢ
        #     #     trace_Sᵢ = sum(diag(Sᵢ))
        #     #     τᶿᵢ      = 2*νtᵢ*ρᵢ * (Sᵢ - (1/3) * trace_Sᵢ * Iᴹ) - (2/3) * ρᵢ * BlinConstant * dx^2 * norm_Sᵢ^2 * Iᴹ
        #     #     Sⱼ = ∇vⱼ =  (m₀/ρᵢ) * (vᵢ - vⱼ) * -∇ᵢWᵢⱼ'
        #     #     norm_Sⱼ  = sqrt(2 * sum(Sⱼ .^ 2))
        #     #     νtⱼ      = (SmagorinskyConstant * dx)^2 * norm_Sⱼ
        #     #     trace_Sⱼ = sum(diag(Sⱼ))
        #     #     τᶿⱼ      = 2*νtⱼ*ρⱼ * (Sⱼ - (1/3) * trace_Sⱼ * Iᴹ) - (2/3) * ρⱼ * BlinConstant * dx^2 * norm_Sⱼ^2 * Iᴹ
        
        #     #     # MATHEMATICALLY THIS IS DOT PRODUCT TO GO FROM TENSOR TO VECTOR, BUT USE * IN JULIA TO REPRESENT IT
        #     #     dτdtᵢ = (m₀/(ρⱼ * ρᵢ)) * (τᶿᵢ + τᶿⱼ) *  ∇ᵢWᵢⱼ 
        #     #     dτdtⱼ = (m₀/(ρᵢ * ρⱼ)) * (τᶿᵢ + τᶿⱼ) * -∇ᵢWᵢⱼ 
        #     # else
        #     #     dτdtᵢ  = zero(xᵢⱼ)
        #     #     dτdtⱼ  = dτdtᵢ
        #     # end
        
        # Cannot use CUDA.@atomic here, be careful! You will get some weird spikes at places
        Particles.Acceleration[i] += dvdt⁺ + Πᵢ #+ ν₀∇²uᵢ + dτdtᵢ
        Particles.Acceleration[j] += dvdt⁻ + Πⱼ #+ ν₀∇²uⱼ + dτdtⱼ

            
        #     # if FlagOutputKernelValues
                Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)

                CUDA.@atomic Particles.Kernel[i] += Wᵢⱼ
                CUDA.@atomic Particles.Kernel[j] += Wᵢⱼ
        #     #     KernelThreaded[ichunk][i]         += Wᵢⱼ
        #     #     KernelThreaded[ichunk][j]         += Wᵢⱼ
        #     #     KernelGradientThreaded[ichunk][i] +=  ∇ᵢWᵢⱼ
        #     #     KernelGradientThreaded[ichunk][j] += -∇ᵢWᵢⱼ
        #     # end


        #     # if SimMetaData.FlagShifting
                # Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)
        
        #     #     MLcond = MotionLimiter[i] * MotionLimiter[j]

        #     #     ∇CᵢThreaded[ichunk][i]   += (m₀/ρᵢ) *  ∇ᵢWᵢⱼ
        #     #     ∇CᵢThreaded[ichunk][j]   += (m₀/ρⱼ) * -∇ᵢWᵢⱼ
        
        #     #     # Switch signs compared to DSPH, else free surface detection does not make sense
        #     #     # Agrees, https://arxiv.org/abs/2110.10076, it should have been r_ji
        #     #     ∇◌rᵢThreaded[ichunk][i]  += (m₀/ρⱼ) * dot(-xᵢⱼ , ∇ᵢWᵢⱼ)  * MLcond
        #     #     ∇◌rᵢThreaded[ichunk][j]  += (m₀/ρᵢ) * dot( xᵢⱼ ,-∇ᵢWᵢⱼ)  * MLcond
        #     # end
        end

        return nothing
    end


UniqueCells = Cells[collect(ParticleRanges[1:IndexCounter])]

function gpu_NeighborLoop!(Particles, SimConstants, UniqueCells, ParticleRanges, Stencil, dρdtI, IndexCounter)
    index  = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x


    i = index
    while i <= IndexCounter
        CellIndex  = UniqueCells[i]

        StartIndex = ParticleRanges[i] 
        EndIndex   = ParticleRanges[i+1] - 1

        @inbounds for i = StartIndex:EndIndex, j = (i+1):EndIndex
            ComputeInteractionsGPU!(Particles, SimConstants, dρdtI, i, j)
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
                    ComputeInteractionsGPU!(Particles, SimConstants, dρdtI, i, j)
                end
            end
        end

        i += stride
    end
    return
end

# Function to launch the CUDA kernel for extracting cells
function launch_NeighborLoopKernel!(Particles, SimConstants, UniqueCells, ParticleRanges, Stencil, dρdtI, IndexCounter)
    kernel = @cuda launch=false gpu_NeighborLoop!(Particles, SimConstants, UniqueCells, ParticleRanges, Stencil, dρdtI, IndexCounter)
    config = launch_configuration(kernel.fun)
    
    threads = min(length(Particles.Cells), config.threads)
    blocks = cld(length(Particles.Cells), threads)

    # Launching the CUDA kernel with the calculated configuration
    CUDA.@sync kernel(Particles, SimConstants, UniqueCells, ParticleRanges, Stencil, dρdtI, IndexCounter; threads=threads, blocks=blocks)
end

launch_NeighborLoopKernel!(SimParticles_GPU, SimConstants, UniqueCells, ParticleRanges, CuVector(Stencil), dρdtI_GPU, IndexCounter)

copyto!(SimParticles,SimParticles_GPU)

# Produce data saving functions
SaveLocation_ = SimMetaData.SaveLocation * "/" * SimMetaData.SimulationName
SaveLocation  = (Iteration) -> SaveLocation_ * "_" * lpad(Iteration,6,"0") * ".vtkhdf"
fid_vector    = Vector{HDF5.File}(undef, Int(SimMetaData.SimulationTime/SimMetaData.OutputEach + 1))
SaveFile   = (Index) -> SaveVTKHDF(fid_vector, Index, SaveLocation(Index),to_3d(SimParticles.Position),["Kernel", "KernelGradient", "Density", "Pressure","Velocity", "Acceleration", "BoundaryBool" , "ID", "Type", "GroupMarker"], SimParticles.Kernel, SimParticles.KernelGradient, SimParticles.Density, SimParticles.Pressure, SimParticles.Velocity, SimParticles.Acceleration, SimParticles.BoundaryBool, SimParticles.ID, UInt8.(SimParticles.Type), SimParticles.GroupMarker)
fid_vector[1] = SaveFile(1)
close.(fid_vector[map( x-> isassigned(fid_vector, x), 1:length(fid_vector))])

        # Construct Motion Definition
        MotionDefinition = Dict{Int, Dict{String, Union{FloatType, SVector{Dimensions, FloatType}}}}()

        # Loop through SimulationGeometry to populate MotionDefinition
        for (_, details) in pairs(SimGeometry)
            motion = get(details, "Motion", nothing)
            if isa(motion, Dict)
                group_marker = details["GroupMarker"]
                MotionDefinition[group_marker] = motion
            end
        end
    

function SimulationLoop(ComputeInteractions!, SimMetaData, SimConstants, SimParticles, Stencil,  ParticleRanges, UniqueCells)
    Position       = SimParticles.Position
    Density        = SimParticles.Density
    Pressure       = SimParticles.Pressure
    Velocity       = SimParticles.Velocity
    Acceleration   = SimParticles.Acceleration
    GravityFactor  = SimParticles.GravityFactor
    MotionLimiter  = SimParticles.MotionLimiter
    ParticleType   = SimParticles.Type
    ParticleMarker = SimParticles.GroupMarker
    Kernel         = SimParticles.Kernel
    KernelGradient = SimParticles.KernelGradient

    ### Some functions to simplify code inside of this function
    # function ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)
    #     @inbounds for i in eachindex(Position)
    #         if ParticleType[i] == Moving
    #             ShouldMove      = MotionDefinition[ParticleMarker[i]]["StartTime"] <= SimMetaData.TotalTime <= (MotionDefinition[ParticleMarker[i]]["StartTime"] + MotionDefinition[ParticleMarker[i]]["Duration"])
    #             MotionVel       = MotionDefinition[ParticleMarker[i]]["Velocity"]  
    #             MotionDir       = MotionDefinition[ParticleMarker[i]]["Direction"]
    #             Velocity[i]     = MotionVel   * MotionDir * ShouldMove
    #             Position[i]    += Velocity[i] * dt₂
    #         end
    #     end
    # end

    ###

    @timeit SimMetaData.HourGlass "01 Update TimeStep"  dt  = Δt(Position, Velocity, Acceleration, SimConstants)
    dt₂ = dt * 0.5

    # In theory, the maximal speed is the speed of sound, this should give a safe guard
    # any ensure it is always updated in a reasonable manner. This only works well, assuming that
    # c₀ >= maximum(norm.(Velocity))
    # Remove if statement logic if you want to update each iteration
    if mod(SimMetaData.Iteration, ceil(Int, 1 / (SimConstants.c₀ * dt * (1/SimConstants.CFL)) )) == 0 || SimMetaData.Iteration == 1
        @timeit SimMetaData.HourGlass "02 Calculate IndexCounter" IndexCounter = UpdateNeighbours!(SimParticles, SortedIndices, CutOff)
    else
        IndexCounter    = findfirst(isequal(0), ParticleRanges) - 2
    end

    # @timeit SimMetaData.HourGlass "Motion" ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)

    # ###=== First step of resetting arrays
    # @timeit SimMetaData.HourGlass "ResetArrays" ResetArrays!(dρdtI, Acceleration, ∇Cᵢ, ∇◌rᵢ)
    # @timeit SimMetaData.HourGlass "ResetArrays" @. ResetArrays!(dρdtIThreaded, AccelerationThreaded)

    # if SimMetaData.FlagOutputKernelValues
    #     @timeit SimMetaData.HourGlass "ResetArrays" ResetArrays!(Kernel, KernelGradient)
    #     @timeit SimMetaData.HourGlass "ResetArrays" @. ResetArrays!(KernelThreaded, KernelGradientThreaded)
    # end

    # if SimMetaData.FlagShifting
    #     @timeit SimMetaData.HourGlass "ResetArrays" ResetArrays!(∇Cᵢ, ∇◌rᵢ)
    #     @timeit SimMetaData.HourGlass "ResetArrays" @. ResetArrays!(∇CᵢThreaded, ∇◌rᵢThreaded)
    # end
    # ###===


    @timeit SimMetaData.HourGlass "03 Pressure"                          Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
    @timeit SimMetaData.HourGlass "04 First NeighborLoop"                launch_NeighborLoopKernel!(SimParticles_GPU, SimConstants, UniqueCells, ParticleRanges, CuVector(Stencil), dρdtI_GPU, IndexCounter)
    # @timeit SimMetaData.HourGlass "Reduction"                            reduce_sum!(dρdtI, dρdtIThreaded)
    # @timeit SimMetaData.HourGlass "Reduction"                            reduce_sum!(Acceleration, AccelerationThreaded)

    # if SimMetaData.FlagShifting
    #     @timeit SimMetaData.HourGlass "Reduction"                        reduce_sum!(∇Cᵢ, ∇CᵢThreaded)
    #     @timeit SimMetaData.HourGlass "Reduction"                        reduce_sum!(∇◌rᵢ, ∇◌rᵢThreaded)
    # end


    # @timeit SimMetaData.HourGlass "05 Update To Half TimeStep" @inbounds for i in eachindex(Position)
    #     Acceleration[i]  +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
    #     Positionₙ⁺[i]     =  Position[i]   + Velocity[i]   * dt₂  * MotionLimiter[i]
    #     Velocityₙ⁺[i]     =  Velocity[i]   + Acceleration[i]  *  dt₂ * MotionLimiter[i]
    #     ρₙ⁺[i]            =  Density[i]    + dρdtI[i]       *  dt₂
    # end

    # @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary"  LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, MotionLimiter)

    # ###=== Second step of resetting arrays
    # @timeit SimMetaData.HourGlass "ResetArrays" ResetArrays!(dρdtI, Acceleration, ∇Cᵢ, ∇◌rᵢ)
    # @timeit SimMetaData.HourGlass "ResetArrays" @. ResetArrays!(dρdtIThreaded, AccelerationThreaded)

    # if SimMetaData.FlagOutputKernelValues
    #     @timeit SimMetaData.HourGlass "ResetArrays" ResetArrays!(Kernel, KernelGradient)
    #     @timeit SimMetaData.HourGlass "ResetArrays" @. ResetArrays!(KernelThreaded, KernelGradientThreaded)
    # end

    # if SimMetaData.FlagShifting
    #     @timeit SimMetaData.HourGlass "ResetArrays" ResetArrays!(∇Cᵢ, ∇◌rᵢ)
    #     @timeit SimMetaData.HourGlass "ResetArrays" @. ResetArrays!(∇CᵢThreaded, ∇◌rᵢThreaded)
    # end
    # ###===

    # @timeit SimMetaData.HourGlass "Motion" ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)

    # @timeit SimMetaData.HourGlass "03 Pressure"                 Pressure!(SimParticles.Pressure, ρₙ⁺,SimConstants)
    # @timeit SimMetaData.HourGlass "08 Second NeighborLoop"      NeighborLoop!(ComputeInteractions!, SimMetaData, SimConstants, ParticleRanges, Stencil, Positionₙ⁺, KernelThreaded, KernelGradientThreaded, ρₙ⁺, Pressure, Velocityₙ⁺, dρdtIThreaded, AccelerationThreaded, ∇CᵢThreaded, ∇◌rᵢThreaded, MotionLimiter, UniqueCells, EnumeratedIndices)
    # @timeit SimMetaData.HourGlass "Reduction"                   reduce_sum!(dρdtI, dρdtIThreaded)
    # @timeit SimMetaData.HourGlass "Reduction"                   reduce_sum!(Acceleration, AccelerationThreaded)

        
    # if SimMetaData.FlagOutputKernelValues
    #     @timeit SimMetaData.HourGlass "Reduction"               reduce_sum!(Kernel, KernelThreaded)
    #     @timeit SimMetaData.HourGlass "Reduction"               reduce_sum!(KernelGradient, KernelGradientThreaded)
    # end

    # if SimMetaData.FlagShifting
    #     @timeit SimMetaData.HourGlass "Reduction"               reduce_sum!(∇Cᵢ, ∇CᵢThreaded)
    #     @timeit SimMetaData.HourGlass "Reduction"               reduce_sum!(∇◌rᵢ, ∇◌rᵢThreaded)
    # end


    # @timeit SimMetaData.HourGlass "09 Final LimitDensityAtBoundary" LimitDensityAtBoundary!(Density, SimConstants.ρ₀, MotionLimiter)

    # @timeit SimMetaData.HourGlass "10 Final Density"                DensityEpsi!(Density, dρdtI, ρₙ⁺, dt)


    # if !SimMetaData.FlagShifting
    #     @timeit SimMetaData.HourGlass "11 Update To Final TimeStep"  @inbounds for i in eachindex(Position)
    #         Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
    #         Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
    #         Position[i]       +=  (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt) * MotionLimiter[i]
    #     end
    # else
    #     A     = 2# Value between 1 to 6 advised
    #     A_FST = 0; # zero for internal flows
    #     A_FSM = length(first(Position)); #2d, 3d val different
    #     @timeit SimMetaData.HourGlass "11 Update To Final TimeStep"  @inbounds for i in eachindex(Position)
    #         Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
    #         Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
    
    #         A_FSC                  = (∇◌rᵢ[i] - A_FST)/(A_FSM - A_FST)
    #         if A_FSC < 0
    #             δxᵢ = zero(eltype(Position))
    #         else
    #             δxᵢ = -A_FSC * A * SimConstants.h * norm(Velocity[i]) * dt * ∇Cᵢ[i]
    #         end
    
    #         Position[i]           += (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt + δxᵢ) * MotionLimiter[i]
    #     end
    # end

    SimMetaData.Iteration      += 1
    SimMetaData.CurrentTimeStep = dt
    SimMetaData.TotalTime      += dt

    
    return nothing
end

SimulationLoop(ComputeInteractionsGPU!, SimMetaData, SimConstants, SimParticles_GPU, CuVector(Stencil),  ParticleRanges, UniqueCells)