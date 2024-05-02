using SPHExample
using StructArrays
using CUDA
using Parameters
using FastPow
import LinearAlgebra: dot, norm
using HDF5
using TimerOutputs
using StaticArrays
using Base.Threads
using ProgressMeter
using Format

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
        kernel = @cuda always_inline=true fastmath=true launch=false gpu_ExtractCells!(Particles, CutOff)
        config = launch_configuration(kernel.fun)
        
        threads = min(length(Particles.Cells), config.threads)
        blocks = cld(length(Particles.Cells), threads)

        # Launching the CUDA kernel with the calculated configuration
        CUDA.@sync kernel(Particles, CutOff; threads=threads, blocks=blocks)
    end


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

    function UpdateNeighbours!(Particles, SortedIndices, ParticleRanges, CutOff)
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


    # A few time stepping controls implemented to allow for an adaptive time step
    function SPHExample.Δt(Position, Velocity, Acceleration, SimulationConstants)
        @unpack c₀, h, CFL, η² = SimulationConstants
        
        visc = maximum(@. abs(h * dot(Velocity,Position) / (dot(Position,Position) + η²)))
        dt1  = minimum(@. sqrt(h / norm(Acceleration)))

        dt2   = h / (c₀+visc)

        dt    = CFL*min(dt1,dt2)

        return dt
    end


    function SPHExample.Pressure!(Press, Density, SimulationConstants)
        @unpack c₀,γ,ρ₀ = SimulationConstants

        Press .= @. EquationOfStateGamma7(Density,c₀,ρ₀)
    end

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

            dρdtI[i] += dρdt⁺ + Dᵢ
            dρdtI[j] += dρdt⁻ + Dⱼ


            Pᵢ      =  Particles.Pressure[i]
            Pⱼ      =  Particles.Pressure[j]
            Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
            dvdt⁺   = - m₀ * Pfac *  ∇ᵢWᵢⱼ
            dvdt⁻   = - dvdt⁺

            ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
            cond      = dot(vᵢⱼ, xᵢⱼ)
            cond_bool = cond < 0.0
            μᵢⱼ       = h*cond * invd²η²
            Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
            Πⱼ        = - Πᵢ
   
            Particles.Acceleration[i] += dvdt⁺ + Πᵢ
            Particles.Acceleration[j] += dvdt⁻ + Πⱼ

            Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)

            Particles.Kernel[i] += Wᵢⱼ
            Particles.Kernel[j] += Wᵢⱼ
            Particles.KernelGradient[i]   +=  ∇ᵢWᵢⱼ
            Particles.KernelGradient[j]   += -∇ᵢWᵢⱼ
        end

        return nothing
    end

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
    kernel = @cuda always_inline=true fastmath=true launch=false gpu_NeighborLoop!(Particles, SimConstants, UniqueCells, ParticleRanges, Stencil, dρdtI, IndexCounter)
    config = launch_configuration(kernel.fun)
    
    threads = min(length(Particles.Cells), config.threads)
    blocks = cld(length(Particles.Cells), threads)

    # Launching the CUDA kernel with the calculated configuration
    CUDA.@sync kernel(Particles, SimConstants, UniqueCells, ParticleRanges, Stencil, dρdtI, IndexCounter; threads=threads, blocks=blocks)
end

function LimitDensityAtBoundaryGPU!(Density, ρ₀, MotionLimiter)
    mask = (Density .< ρ₀) .& (.!Bool.(MotionLimiter))
    Density[mask] .= ρ₀
end

function DensityEpsiGPU!(Density, dρdtIₙ⁺, ρₙ⁺, Δt)
    epsi = - (dρdtIₙ⁺ ./ ρₙ⁺) .* Δt
    Density .*= ((2 .- epsi) ./ (2 .+ epsi))
end



function SimulationLoop(SimMetaData, SimConstants, SimParticles, Stencil,  ParticleRanges, SortedIndices, dρdtI, ρₙ⁺, Positionₙ⁺, Velocityₙ⁺)
    Position       = SimParticles.Position
    Cells          = SimParticles.Cells
    Velocity       = SimParticles.Velocity
    Acceleration   = SimParticles.Acceleration
    Kernel         = SimParticles.Kernel
    KernelGradient = SimParticles.KernelGradient

    function update_half_timestep!(Position, Velocity, Positionₙ⁺, Velocityₙ⁺, Acceleration, ρₙ⁺, Density, dρdtI, GravityFactor, MotionLimiter, SimConstants, dt₂)
        # Update Acceleration vector by adding a gravity component
        Acceleration .= Acceleration .+ CUDA.fill(SVector{2,Float64}(0,-SimConstants.g), length(Acceleration))
    
        # Update Positionₙ⁺ vector
        Positionₙ⁺ .= Position .+ Velocity .* dt₂ .* MotionLimiter
    
        # Update Velocityₙ⁺ vector
        Velocityₙ⁺ .= Velocity .+ Acceleration .* dt₂ .* MotionLimiter
    
        # Update ρₙ⁺ vector
        ρₙ⁺ .= Density .+ dρdtI .* dt₂
    end

    function update_final_timestep!(Position, Velocity, Acceleration, GravityFactor, MotionLimiter, SimConstants, dt)
        # Update Acceleration by adding a gravity component
        Acceleration .= Acceleration .+ CUDA.fill(SVector{2,Float64}(0,-SimConstants.g), length(Acceleration))
    
        # Update Velocity
        Velocity .= Velocity .+ Acceleration .* dt .* MotionLimiter
    
        # Update Position
        delta_pos = ((Velocity .- Acceleration .* dt .* MotionLimiter) .+ Velocity) ./ 2 .* dt
        Position .= Position .+ delta_pos .* MotionLimiter
    end

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
        @timeit SimMetaData.HourGlass "02 Calculate IndexCounter" IndexCounter = UpdateNeighbours!(SimParticles, SortedIndices, ParticleRanges, SimConstants.H) # CutOff = SimConstants.H
    else
        IndexCounter    = findfirst(isequal(0), ParticleRanges) - 2
    end

    UniqueCells = Cells[collect(ParticleRanges[1:IndexCounter])]


    # @timeit SimMetaData.HourGlass "Motion" ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)

    # ###=== First step of resetting arrays
    @timeit SimMetaData.HourGlass "ResetArrays" ResetArrays!(dρdtI, Acceleration)
    # @timeit SimMetaData.HourGlass "ResetArrays" @. ResetArrays!(dρdtIThreaded, AccelerationThreaded)

    # if SimMetaData.FlagOutputKernelValues
        @timeit SimMetaData.HourGlass "ResetArrays" ResetArrays!(Kernel, KernelGradient)
    #     @timeit SimMetaData.HourGlass "ResetArrays" @. ResetArrays!(KernelThreaded, KernelGradientThreaded)
    # end

    # if SimMetaData.FlagShifting
    #     @timeit SimMetaData.HourGlass "ResetArrays" ResetArrays!(∇Cᵢ, ∇◌rᵢ)
    #     @timeit SimMetaData.HourGlass "ResetArrays" @. ResetArrays!(∇CᵢThreaded, ∇◌rᵢThreaded)
    # end
    # ###===


    @timeit SimMetaData.HourGlass "03 Pressure"                          Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
    @timeit SimMetaData.HourGlass "04 First NeighborLoop"                launch_NeighborLoopKernel!(SimParticles, SimConstants, UniqueCells, ParticleRanges, Stencil, dρdtI, IndexCounter)
    # @timeit SimMetaData.HourGlass "Reduction"                            reduce_sum!(dρdtI, dρdtIThreaded)
    # @timeit SimMetaData.HourGlass "Reduction"                            reduce_sum!(Acceleration, AccelerationThreaded)

    # if SimMetaData.FlagShifting
    #     @timeit SimMetaData.HourGlass "Reduction"                        reduce_sum!(∇Cᵢ, ∇CᵢThreaded)
    #     @timeit SimMetaData.HourGlass "Reduction"                        reduce_sum!(∇◌rᵢ, ∇◌rᵢThreaded)
    # end


    @timeit SimMetaData.HourGlass "05 Update To Half TimeStep" update_half_timestep!(SimParticles.Position, SimParticles.Velocity, Positionₙ⁺, Velocityₙ⁺, Acceleration, ρₙ⁺, SimParticles.Density, dρdtI, SimParticles.GravityFactor,SimParticles. MotionLimiter, SimConstants, dt₂)

    @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary"  LimitDensityAtBoundaryGPU!(ρₙ⁺, SimConstants.ρ₀, SimParticles.MotionLimiter)

    # ###=== Second step of resetting arrays
    @timeit SimMetaData.HourGlass "ResetArrays" ResetArrays!(dρdtI, Acceleration)
    # @timeit SimMetaData.HourGlass "ResetArrays" @. ResetArrays!(dρdtIThreaded, AccelerationThreaded)

    # if SimMetaData.FlagOutputKernelValues
        @timeit SimMetaData.HourGlass "ResetArrays" ResetArrays!(Kernel, KernelGradient)
    #     @timeit SimMetaData.HourGlass "ResetArrays" @. ResetArrays!(KernelThreaded, KernelGradientThreaded)
    # end

    # if SimMetaData.FlagShifting
    #     @timeit SimMetaData.HourGlass "ResetArrays" ResetArrays!(∇Cᵢ, ∇◌rᵢ)
    #     @timeit SimMetaData.HourGlass "ResetArrays" @. ResetArrays!(∇CᵢThreaded, ∇◌rᵢThreaded)
    # end
    # ###===

    # @timeit SimMetaData.HourGlass "Motion" ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)

    @timeit SimMetaData.HourGlass "03 Pressure"                 Pressure!(SimParticles.Pressure, ρₙ⁺,SimConstants)
    @timeit SimMetaData.HourGlass "08 Second NeighborLoop"      launch_NeighborLoopKernel!(SimParticles, SimConstants, UniqueCells, ParticleRanges, Stencil, dρdtI, IndexCounter)
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


    @timeit SimMetaData.HourGlass "09 Final LimitDensityAtBoundary" LimitDensityAtBoundaryGPU!(SimParticles.Density, SimConstants.ρ₀, SimParticles.MotionLimiter)

    @timeit SimMetaData.HourGlass "10 Final Density"                DensityEpsiGPU!(SimParticles.Density, dρdtI, ρₙ⁺, dt)


    # if !SimMetaData.FlagShifting
    #     @timeit SimMetaData.HourGlass "11 Update To Final TimeStep"  @inbounds for i in eachindex(Position)
    #         Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
    #         Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
    #         Position[i]       +=  (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt) * MotionLimiter[i]
    #     end
    @timeit SimMetaData.HourGlass "11 Update To Final TimeStep" update_final_timestep!(Position, Velocity, Acceleration, SimParticles.GravityFactor, SimParticles.MotionLimiter, SimConstants, dt)
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

function RunSimulationGPU(;SimGeometry::Dict, #Don't further specify type for now
    SimMetaData::SimulationMetaData{Dimensions, FloatType},
    SimConstants::SimulationConstants,
    SimLogger::SimulationLogger
    ) where {Dimensions,FloatType}

    
    # Delete previous result files
    foreach(rm, filter(endswith(".vtp"), readdir(SimMetaData.SaveLocation,join=true)))
    # https://discourse.julialang.org/t/find-what-has-locked-held-a-file/23278
    GC.gc()
    try
        foreach(rm, filter(endswith(".vtkhdf"), readdir(SimMetaData.SaveLocation,join=true)))
    catch err
        @warn("File could not be deleted, manually delete else program cannot conclude.")
        display(err)
    end

    # Unpack the relevant simulation meta data
    @unpack HourGlass = SimMetaData;

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

    NumberOfPoints = length(SimParticles)::Int #Have to type declare, else error?

    if SimMetaData.FlagLog
        InitializeLogger(SimLogger,SimConstants,SimMetaData, SimGeometry, SimParticles)
    end
    

    Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)

    # # Shifting correction
    # ∇Cᵢ               = zeros(SVector{Dimensions,FloatType},NumberOfPoints)            
    # ∇◌rᵢ              = zeros(FloatType,NumberOfPoints)    

    # @inline begin
    #     n_copy = Base.Threads.nthreads()
    #     KernelThreaded         = [copy(SimParticles.Kernel)         for _ in 1:n_copy]
    #     KernelGradientThreaded = [copy(SimParticles.KernelGradient) for _ in 1:n_copy]
    #     dρdtIThreaded          = [copy(dρdtI)                       for _ in 1:n_copy]
    #     AccelerationThreaded   = [copy(SimParticles.KernelGradient) for _ in 1:n_copy]
    #     ∇CᵢThreaded            = [copy(∇Cᵢ )                        for _ in 1:n_copy]
    #     ∇◌rᵢThreaded           = [copy(∇◌rᵢ)                        for _ in 1:n_copy]   
    # end

    # Produce sorting related variables
    ParticleRanges         = CUDA.zeros(Int, NumberOfPoints + 1)
    UniqueCells            = CUDA.zeros(CartesianIndex{Dimensions}, NumberOfPoints)
    Stencil                = CuVector(ConstructStencil(Val(Dimensions)))
    SortedIndices          = CUDA.zeros(Int, NumberOfPoints)

    # Produce data saving functions
    SaveLocation_ = SimMetaData.SaveLocation * "/" * SimMetaData.SimulationName
    SaveLocation  = (Iteration) -> SaveLocation_ * "_" * lpad(Iteration,6,"0") * ".vtkhdf"

    fid_vector    = Vector{HDF5.File}(undef, Int(SimMetaData.SimulationTime/SimMetaData.OutputEach + 1))

    SaveFile   = (Index) -> SaveVTKHDF(fid_vector, Index, SaveLocation(Index),to_3d(SimParticles.Position),["Kernel", "KernelGradient", "Density", "Pressure","Velocity", "Acceleration", "BoundaryBool" , "ID", "Type", "GroupMarker"], SimParticles.Kernel, SimParticles.KernelGradient, SimParticles.Density, SimParticles.Pressure, SimParticles.Velocity, SimParticles.Acceleration, SimParticles.BoundaryBool, SimParticles.ID, UInt8.(SimParticles.Type), SimParticles.GroupMarker)
    SimMetaData.OutputIterationCounter += 1 #Since a file has been saved
    SaveFile(SimMetaData.OutputIterationCounter)
    

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

    # Normal run and save data
    generate_showvalues(Iteration, TotalTime, TimeLeftInSeconds) = () -> [(:(Iteration),format(FormatExpr("{1:d}"),  Iteration)), (:(TotalTime),format(FormatExpr("{1:3.3f}"), TotalTime)), (:(TimeLeftInSeconds),format(FormatExpr("{1:3.1f} [s]"), TimeLeftInSeconds))]

    @inbounds while true

        SimulationLoop(SimMetaData, SimConstants, SimParticles_GPU, Stencil,  ParticleRanges, SortedIndices, dρdtI_GPU, ρₙ⁺_GPU, Positionₙ⁺_GPU, Velocityₙ⁺_GPU)

        if SimMetaData.TotalTime >= SimMetaData.OutputEach * SimMetaData.OutputIterationCounter


        try 
            copyto!(SimParticles,SimParticles_GPU)
            @timeit HourGlass "12A Output Data" SaveFile(SimMetaData.OutputIterationCounter + 1)
        catch err
            @warn("File write failed.")
            display(err)
        end

            if SimMetaData.FlagLog
                LogStep(SimLogger, SimMetaData, HourGlass)
                SimMetaData.StepsTakenForLastOutput = SimMetaData.Iteration
            end

            SimMetaData.OutputIterationCounter += 1
        end

        TimeLeftInSeconds = (SimMetaData.SimulationTime - SimMetaData.TotalTime) * (TimerOutputs.tottime(HourGlass)/1e9 / SimMetaData.TotalTime)
        @timeit HourGlass "13 Next TimeStep" next!(SimMetaData.ProgressSpecification; showvalues = generate_showvalues(SimMetaData.Iteration , SimMetaData.TotalTime, TimeLeftInSeconds))

        if SimMetaData.TotalTime > SimMetaData.SimulationTime
            
            if SimMetaData.FlagLog
                LogFinal(SimLogger, HourGlass)
                close(SimLogger.LoggerIo)
            end

            
            # This should not be counted in actual run 
            @timeit HourGlass "12B Close hdfvtk output files"  @threads for i in eachindex(fid_vector)
                if isassigned(fid_vector, i)
                    close(fid_vector[i])
                end
            end

            finish!(SimMetaData.ProgressSpecification)
            show(HourGlass,sortby=:name)
            show(HourGlass)

            break
        end
    end
end

let        
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
        SimulationName="StillWedge2", 
        SaveLocation="E:/SecondApproach/StillWedge_GPU",
        SimulationTime=0.01,
        OutputEach=0.01,
        FlagDensityDiffusion=true,
        FlagOutputKernelValues=false,
        FlagLog=true,
        FlagShifting=false,
    )

    SimLogger = SimulationLogger(SimMetaData.SaveLocation)
    
    CUDA.@profile RunSimulationGPU(
            SimGeometry        = SimGeometry,
            SimMetaData        = SimMetaData,
            SimConstants       = SimConstants,
            SimLogger          = SimLogger
        )
end