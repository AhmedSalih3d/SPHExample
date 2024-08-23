module SPHCellList

export ConstructStencil, ExtractCells!, UpdateNeighbors!, NeighborLoop!, ComputeInteractions!, RunSimulation

using Parameters, FastPow, StaticArrays, Base.Threads, ChunkSplitters
import LinearAlgebra: dot

using ..SimulationEquations
using ..AuxillaryFunctions
using ..SimulationMetaDataConfiguration
using ..SimulationConstantsConfiguration
using ..SimulationLoggerConfiguration
using ..PreProcess
using ..ProduceHDFVTK
using ..TimeStepping
using ..OpenExternalPrograms

using StaticArrays
import StructArrays: StructArray
import LinearAlgebra: dot, norm, diagm, diag, cond, det
import Parameters: @unpack
import FastPow: @fastpow
import ProgressMeter: next!, finish!
using Format
using TimerOutputs
using Logging, LoggingExtras
using HDF5
using Base.Threads

    function ConstructStencil(v::Val{d}) where d
        n_ = CartesianIndices(ntuple(_->-1:1,v))
        half_length = length(n_) ÷ 2
        n  = n_[1:half_length]

        return n
    end

    @inline function ExtractCells!(Particles, ::Val{InverseCutOff}) where InverseCutOff
        # Replace unsafe_trunc with trunc if this ever errors
        function map_floor(x)
            unsafe_trunc(Int, muladd(x,InverseCutOff,2))
        end

        Cells  = @views Particles.Cells
        Points = @views Particles.Position
        @threads for i ∈ eachindex(Particles)
            t = map(map_floor, Tuple(Points[i]))
            Cells[i] = CartesianIndex(t)
        end
        return nothing
    end

    ###=== Function to update ordering
    function UpdateNeighbors!(Particles, CutOff, SortingScratchSpace, ParticleRanges, UniqueCells)
        ExtractCells!(Particles, CutOff)

        sort!(Particles, by = p -> p.Cells; scratch=SortingScratchSpace)

        Cells = @views Particles.Cells
        @. ParticleRanges             = zero(eltype(ParticleRanges))
        IndexCounter                  = 1
        ParticleRanges[IndexCounter]  = 1
        UniqueCells[IndexCounter]     = Cells[1]

        for i in eachindex(Cells)[2:end]
            if Cells[i] != Cells[i-1] # Equivalent to diff(Cells) != 0
                IndexCounter                 += 1
                ParticleRanges[IndexCounter]  = i
                UniqueCells[IndexCounter]     = Cells[i]
            end
        end
        ParticleRanges[IndexCounter + 1]  = length(ParticleRanges)

        return IndexCounter 
    end

# Neither Polyester.@batch per core or thread is faster
###=== Function to process each cell and its neighbors
    function NeighborLoop!(SimMetaData, SimConstants, ParticleRanges, Stencil, Position, Kernel, KernelGradient, Density, Pressure, Velocity, dρdtI, dvdtI,  ∇CᵢThreaded, ∇◌rᵢThreaded, MotionLimiter, UniqueCells, EnumeratedIndices)
        @threads for (ichunk, inds) in @views EnumeratedIndices
            for iter in inds
                CellIndex = UniqueCells[iter]

                StartIndex = ParticleRanges[iter] 
                EndIndex   = ParticleRanges[iter+1] - 1

                @inbounds for i = StartIndex:EndIndex, j = (i+1):EndIndex
                    @inline ComputeInteractions!(SimMetaData, SimConstants, Position, Kernel, KernelGradient, Density, Pressure, Velocity, dρdtI, dvdtI, ∇CᵢThreaded, ∇◌rᵢThreaded, i, j, MotionLimiter, ichunk)
                end

                @inbounds for S ∈ Stencil
                    SCellIndex = CellIndex + S

                    # Returns a range, x:x for exact match and x:(x-1) for no match
                    # utilizes that it is a sorted array and requires no isequal constructor,
                    # so I prefer this for now
                    NeighborCellIndex = searchsorted(UniqueCells, SCellIndex)

                    if length(NeighborCellIndex) != 0
                        StartIndex_       = ParticleRanges[NeighborCellIndex[1]] 
                        EndIndex_         = ParticleRanges[NeighborCellIndex[1]+1] - 1

                        @inbounds for i = StartIndex:EndIndex, j = StartIndex_:EndIndex_
                            @inline ComputeInteractions!(SimMetaData, SimConstants, Position, Kernel, KernelGradient, Density, Pressure, Velocity, dρdtI, dvdtI, ∇CᵢThreaded, ∇◌rᵢThreaded, i, j, MotionLimiter, ichunk)
                        end
                    end
                end
            end
        end

        return nothing
    end

    # Really important to overload default function, gives 10x speed up?
    # Overload the default function to do what you pleas
    function ComputeInteractions!(SimMetaData, SimConstants, Position, KernelThreaded, KernelGradientThreaded, Density, Pressure, Velocity, dρdtI, dvdtI, ∇CᵢThreaded, ∇◌rᵢThreaded, i, j, MotionLimiter, ichunk)
        @unpack FlagViscosityTreatment, FlagDensityDiffusion, FlagOutputKernelValues = SimMetaData
        @unpack ρ₀, h, h⁻¹, m₀, αD, α, g, c₀, δᵩ, η², H², Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants

        xᵢⱼ  = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)              
        if  xᵢⱼ² <= H²
            #https://discourse.julialang.org/t/sqrt-abs-x-is-even-faster-than-sqrt/58154/2
            dᵢⱼ  = sqrt(abs(xᵢⱼ²))

            # clamp seems faster than min, no util
            q         = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0) #min(dᵢⱼ * h⁻¹, 2.0) - 8% util no DDT
            invd²η²   =  1.0 / (dᵢⱼ*dᵢⱼ+η²)
            ∇ᵢWᵢⱼ     = @fastpow (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 
            ρᵢ        = Density[i]
            ρⱼ        = Density[j]
        
            vᵢ        = Velocity[i]
            vⱼ        = Velocity[j]
            vᵢⱼ       = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺          = - ρᵢ * (m₀/ρⱼ) *  density_symmetric_term
            dρdt⁻          = - ρⱼ * (m₀/ρᵢ) *  density_symmetric_term

            # Density diffusion
            if FlagDensityDiffusion
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

                MLcond = MotionLimiter[i] * MotionLimiter[j]
                Dᵢ    =  δᵩ * h * c₀ * (m₀/ρⱼ) * dot(Ψᵢⱼ ,  ∇ᵢWᵢⱼ) * MLcond
                Dⱼ    =  δᵩ * h * c₀ * (m₀/ρᵢ) * dot(Ψⱼᵢ , -∇ᵢWᵢⱼ) * MLcond
            else
                Dᵢ  = 0.0
                Dⱼ  = 0.0
            end
            dρdtI[ichunk][i] += dρdt⁺ + Dᵢ
            dρdtI[ichunk][j] += dρdt⁻ + Dⱼ


            Pᵢ      =  Pressure[i]
            Pⱼ      =  Pressure[j]
            Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
            dvdt⁺   = - m₀ * Pfac *  ∇ᵢWᵢⱼ
            dvdt⁻   = - dvdt⁺

            if FlagViscosityTreatment == :ArtificialViscosity
                ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
                cond      = dot(vᵢⱼ, xᵢⱼ)
                cond_bool = cond < 0.0
                μᵢⱼ       = h*cond * invd²η²
                Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
                Πⱼ        = - Πᵢ
            else
                Πᵢ        = zero(xᵢⱼ)
                Πⱼ        = Πᵢ
            end
        
            if FlagViscosityTreatment == :Laminar || FlagViscosityTreatment == :LaminarSPS
                # 4 comes from 2 divided by 0.5 from average density
                # should divide by ρᵢ eq 6 DPC
                # ν₀∇²uᵢ = (1/ρᵢ) * ( (4 * m₀ * (ρᵢ * ν₀) * dot( xᵢⱼ, ∇ᵢWᵢⱼ)  ) / ( (ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²) ) ) *  vᵢⱼ
                # ν₀∇²uⱼ = (1/ρⱼ) * ( (4 * m₀ * (ρⱼ * ν₀) * dot(-xᵢⱼ,-∇ᵢWᵢⱼ)  ) / ( (ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²) ) ) * -vᵢⱼ
                visc_symmetric_term = (4 * m₀ * ν₀ * dot( xᵢⱼ, ∇ᵢWᵢⱼ)) / ((ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²))
                # ν₀∇²uᵢ = (1/ρᵢ) * visc_symmetric_term *  vᵢⱼ * ρᵢ
                # ν₀∇²uⱼ = (1/ρⱼ) * visc_symmetric_term * -vᵢⱼ * ρⱼ
                ν₀∇²uᵢ =  visc_symmetric_term *  vᵢⱼ
                ν₀∇²uⱼ = -ν₀∇²uᵢ #visc_symmetric_term * -vᵢⱼ
            else
                ν₀∇²uᵢ = zero(xᵢⱼ)
                ν₀∇²uⱼ = ν₀∇²uᵢ
            end
        
            if FlagViscosityTreatment == :LaminarSPS 
                Iᴹ       = diagm(one.(xᵢⱼ))
                #julia> a .- a'
                # 3×3 SMatrix{3, 3, Float64, 9} with indices SOneTo(3)×SOneTo(3):
                # 0.0  0.0  0.0
                # 0.0  0.0  0.0
                # 0.0  0.0  0.0
                # Strain *rate* tensor is the gradient of velocity
                Sᵢ = ∇vᵢ =  (m₀/ρⱼ) * (vⱼ - vᵢ) * ∇ᵢWᵢⱼ'
                norm_Sᵢ  = sqrt(2 * sum(Sᵢ .^ 2))
                νtᵢ      = (SmagorinskyConstant * dx)^2 * norm_Sᵢ
                trace_Sᵢ = sum(diag(Sᵢ))
                τᶿᵢ      = 2*νtᵢ*ρᵢ * (Sᵢ - (1/3) * trace_Sᵢ * Iᴹ) - (2/3) * ρᵢ * BlinConstant * dx^2 * norm_Sᵢ^2 * Iᴹ
                Sⱼ = ∇vⱼ =  (m₀/ρᵢ) * (vᵢ - vⱼ) * -∇ᵢWᵢⱼ'
                norm_Sⱼ  = sqrt(2 * sum(Sⱼ .^ 2))
                νtⱼ      = (SmagorinskyConstant * dx)^2 * norm_Sⱼ
                trace_Sⱼ = sum(diag(Sⱼ))
                τᶿⱼ      = 2*νtⱼ*ρⱼ * (Sⱼ - (1/3) * trace_Sⱼ * Iᴹ) - (2/3) * ρⱼ * BlinConstant * dx^2 * norm_Sⱼ^2 * Iᴹ
        
                # MATHEMATICALLY THIS IS DOT PRODUCT TO GO FROM TENSOR TO VECTOR, BUT USE * IN JULIA TO REPRESENT IT
                dτdtᵢ = (m₀/(ρⱼ * ρᵢ)) * (τᶿᵢ + τᶿⱼ) *  ∇ᵢWᵢⱼ 
                dτdtⱼ = (m₀/(ρᵢ * ρⱼ)) * (τᶿᵢ + τᶿⱼ) * -∇ᵢWᵢⱼ 
            else
                dτdtᵢ  = zero(xᵢⱼ)
                dτdtⱼ  = dτdtᵢ
            end
        
            dvdtI[ichunk][i] += dvdt⁺ + Πᵢ + ν₀∇²uᵢ + dτdtᵢ
            dvdtI[ichunk][j] += dvdt⁻ + Πⱼ + ν₀∇²uⱼ + dτdtⱼ

            
            if FlagOutputKernelValues
                Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)
                KernelThreaded[ichunk][i]         += Wᵢⱼ
                KernelThreaded[ichunk][j]         += Wᵢⱼ
                KernelGradientThreaded[ichunk][i] +=  ∇ᵢWᵢⱼ
                KernelGradientThreaded[ichunk][j] += -∇ᵢWᵢⱼ
            end


            if SimMetaData.FlagShifting
                Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)
        
                MLcond = MotionLimiter[i] * MotionLimiter[j]

                ∇CᵢThreaded[ichunk][i]   += (m₀/ρᵢ) *  ∇ᵢWᵢⱼ
                ∇CᵢThreaded[ichunk][j]   += (m₀/ρⱼ) * -∇ᵢWᵢⱼ
        
                # Switch signs compared to DSPH, else free surface detection does not make sense
                # Agrees, https://arxiv.org/abs/2110.10076, it should have been r_ji
                ∇◌rᵢThreaded[ichunk][i]  += (m₀/ρⱼ) * dot(-xᵢⱼ , ∇ᵢWᵢⱼ)  * MLcond
                ∇◌rᵢThreaded[ichunk][j]  += (m₀/ρᵢ) * dot( xᵢⱼ ,-∇ᵢWᵢⱼ)  * MLcond
            end
        end

        return nothing
    end

    # Neither Polyester.@batch per core or thread is faster
###=== Function to process each cell and its neighbors
function NeighborLoop_MDBC!(SimMetaData, SimConstants, ParticleRanges, Stencil, Position, Types, GhostPoints, GhostNormals)

    return nothing
end


    function reduce_sum!(target_array, arrays)
        n = length(target_array)
        num_threads = nthreads()
        chunk_size = ceil(Int, n / num_threads)
        @inbounds @threads for t in 1:num_threads
            local start_idx = 1 + (t-1) * chunk_size
            local end_idx = min(t * chunk_size, n)
            for j in eachindex(arrays)
                local array = arrays[j]  # Access array only once per thread
                @simd for i in start_idx:end_idx
                    @inbounds target_array[i] += array[i]
                end
            end
        end
    end

    function ResetStep!(SimMetaData, dρdtI, Acceleration, dρdtIThreaded, AccelerationThreaded, Kernel, KernelGradient, KernelThreaded, KernelGradientThreaded, ∇Cᵢ, ∇◌rᵢ, ∇CᵢThreaded, ∇◌rᵢThreaded)
        ResetArrays!(dρdtI, Acceleration)
        @. ResetArrays!(dρdtIThreaded, AccelerationThreaded)

        if SimMetaData.FlagOutputKernelValues
            ResetArrays!(Kernel, KernelGradient)
            @. ResetArrays!(KernelThreaded, KernelGradientThreaded)
        end

        if SimMetaData.FlagShifting
            ResetArrays!(∇Cᵢ, ∇◌rᵢ)
            @. ResetArrays!(∇CᵢThreaded, ∇◌rᵢThreaded)
        end

        return nothing
    end

    function ReductionStep!(SimMetaData, dρdtI, dρdtIThreaded, Acceleration, AccelerationThreaded, Kernel, KernelThreaded, KernelGradient, KernelGradientThreaded, ∇Cᵢ, ∇CᵢThreaded, ∇◌rᵢ, ∇◌rᵢThreaded)
        reduce_sum!(dρdtI, dρdtIThreaded)
        reduce_sum!(Acceleration, AccelerationThreaded)
  
        if SimMetaData.FlagOutputKernelValues
            reduce_sum!(Kernel, KernelThreaded)
            reduce_sum!(KernelGradient, KernelGradientThreaded)
        end

        if SimMetaData.FlagShifting
            reduce_sum!(∇Cᵢ, ∇CᵢThreaded)
            reduce_sum!(∇◌rᵢ, ∇◌rᵢThreaded)
        end
    
        return nothing
    end
    
    @inbounds function SimulationLoop(SimMetaData, SimConstants, SimParticles, GhostPoints, GhostNormals, Stencil,  ParticleRanges, UniqueCells, SortingScratchSpace, KernelThreaded, KernelGradientThreaded, dρdtI, dρdtIThreaded, AccelerationThreaded, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇CᵢThreaded, ∇◌rᵢ, ∇◌rᵢThreaded, MotionDefinition, InverseCutOff)
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
        function ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)
            @inbounds for i in eachindex(Position)
                if ParticleType[i] == Moving
                    ShouldMove      = MotionDefinition[ParticleMarker[i]]["StartTime"] <= SimMetaData.TotalTime <= (MotionDefinition[ParticleMarker[i]]["StartTime"] + MotionDefinition[ParticleMarker[i]]["Duration"])
                    MotionVel       = MotionDefinition[ParticleMarker[i]]["Velocity"]  
                    MotionDir       = MotionDefinition[ParticleMarker[i]]["Direction"]
                    Velocity[i]     = MotionVel   * MotionDir * ShouldMove
                    Position[i]    += Velocity[i] * dt₂
                end
            end
        end
    
        ###
    
        @timeit SimMetaData.HourGlass "01 Update TimeStep"  dt  = Δt(Position, Velocity, Acceleration, SimConstants)
        dt₂ = dt * 0.5

        # In theory, the maximal speed is the speed of sound, this should give a safe guard
        # any ensure it is always updated in a reasonable manner. This only works well, assuming that
        # c₀ >= maximum(norm.(Velocity))
        # Remove if statement logic if you want to update each iteration
        if mod(SimMetaData.Iteration, ceil(Int, 1 / (SimConstants.c₀ * dt * (1/SimConstants.CFL)) )) == 0 || SimMetaData.Iteration == 1
            @timeit SimMetaData.HourGlass "02 Calculate IndexCounter" IndexCounter = UpdateNeighbors!(SimParticles, InverseCutOff, SortingScratchSpace,  ParticleRanges, UniqueCells)
        else
            findfirst_int(predicate, collection) = (idx = findfirst(predicate, collection); idx === nothing ? -1 : idx)
            IndexCounter    = findfirst_int(isequal(0), ParticleRanges) - 2
        end

        UniqueCellsView   = view(UniqueCells, 1:IndexCounter)
        EnumeratedIndices = enumerate(chunks(UniqueCellsView; n=nthreads()))


        @timeit SimMetaData.HourGlass "Motion" ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)
    
        ###=== First step of resetting arrays
        @timeit SimMetaData.HourGlass "ResetArrays" ResetStep!(SimMetaData, dρdtI, Acceleration, dρdtIThreaded, AccelerationThreaded, Kernel, KernelGradient, KernelThreaded, KernelGradientThreaded, ∇Cᵢ, ∇◌rᵢ, ∇CᵢThreaded, ∇◌rᵢThreaded)
        ###===
    
        @timeit SimMetaData.HourGlass "03 Pressure"                          Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
        @timeit SimMetaData.HourGlass "04 First NeighborLoop mDBC"           NeighborLoop_MDBC!(SimMetaData, SimConstants, ParticleRanges, Stencil, Position, ParticleType, GhostPoints, GhostNormals)
        @timeit SimMetaData.HourGlass "04 First NeighborLoop"                NeighborLoop!(SimMetaData, SimConstants, ParticleRanges, Stencil, Position, KernelThreaded, KernelGradientThreaded, Density, Pressure, Velocity, dρdtIThreaded, AccelerationThreaded,  ∇CᵢThreaded, ∇◌rᵢThreaded, MotionLimiter, UniqueCellsView, EnumeratedIndices)
        @timeit SimMetaData.HourGlass "Reduction"                            ReductionStep!(SimMetaData, dρdtI, dρdtIThreaded, Acceleration, AccelerationThreaded, Kernel, KernelThreaded, KernelGradient, KernelGradientThreaded, ∇Cᵢ, ∇CᵢThreaded, ∇◌rᵢ, ∇◌rᵢThreaded)
    
        @timeit SimMetaData.HourGlass "05 Update To Half TimeStep" @inbounds for i in eachindex(Position)
            Acceleration[i]  +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
            Positionₙ⁺[i]     =  Position[i]   + Velocity[i]   * dt₂  * MotionLimiter[i]
            Velocityₙ⁺[i]     =  Velocity[i]   + Acceleration[i]  *  dt₂ * MotionLimiter[i]
            ρₙ⁺[i]            =  Density[i]    + dρdtI[i]       *  dt₂
        end
    
        @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary"  LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, MotionLimiter)
    
        ###=== Second step of resetting arrays
        ResetStep!(SimMetaData, dρdtI, Acceleration, dρdtIThreaded, AccelerationThreaded, Kernel, KernelGradient, KernelThreaded, KernelGradientThreaded, ∇Cᵢ, ∇◌rᵢ, ∇CᵢThreaded, ∇◌rᵢThreaded)
        ###===

        @timeit SimMetaData.HourGlass "Motion" ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)
    
        @timeit SimMetaData.HourGlass "03 Pressure"                 Pressure!(SimParticles.Pressure, ρₙ⁺,SimConstants)
        @timeit SimMetaData.HourGlass "08 Second NeighborLoop"      NeighborLoop!(SimMetaData, SimConstants, ParticleRanges, Stencil, Positionₙ⁺, KernelThreaded, KernelGradientThreaded, ρₙ⁺, Pressure, Velocityₙ⁺, dρdtIThreaded, AccelerationThreaded, ∇CᵢThreaded, ∇◌rᵢThreaded, MotionLimiter, UniqueCellsView, EnumeratedIndices)
        @timeit SimMetaData.HourGlass "Reduction"                   ReductionStep!(SimMetaData, dρdtI, dρdtIThreaded, Acceleration, AccelerationThreaded, Kernel, KernelThreaded, KernelGradient, KernelGradientThreaded, ∇Cᵢ, ∇CᵢThreaded, ∇◌rᵢ, ∇◌rᵢThreaded)

    
        @timeit SimMetaData.HourGlass "09 Final LimitDensityAtBoundary" LimitDensityAtBoundary!(Density, SimConstants.ρ₀, MotionLimiter)
    
        @timeit SimMetaData.HourGlass "10 Final Density"                DensityEpsi!(Density, dρdtI, ρₙ⁺, dt)
    
    
        if !SimMetaData.FlagShifting
            @timeit SimMetaData.HourGlass "11 Update To Final TimeStep"  @inbounds for i in eachindex(Position)
                Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
                Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
                Position[i]       +=  (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt) * MotionLimiter[i]
            end
        else
            A     = 2# Value between 1 to 6 advised
            A_FST = 0; # zero for internal flows
            A_FSM = length(first(Position)); #2d, 3d val different
            @timeit SimMetaData.HourGlass "11 Update To Final TimeStep"  @inbounds for i in eachindex(Position)
                Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
                Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
        
                A_FSC                  = (∇◌rᵢ[i] - A_FST)/(A_FSM - A_FST)
                if A_FSC < 0
                    δxᵢ = zero(eltype(Position))
                else
                    δxᵢ = -A_FSC * A * SimConstants.h * norm(Velocity[i]) * dt * ∇Cᵢ[i]
                end
        
                Position[i]           += (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt + δxᵢ) * MotionLimiter[i]
            end
        end
    
        SimMetaData.Iteration      += 1
        SimMetaData.CurrentTimeStep = dt
        SimMetaData.TotalTime      += dt

        
        return nothing
    end
    
    ###===
    function RunSimulation(;SimGeometry::Dict, #Don't further specify type for now
        SimMetaData::SimulationMetaData{Dimensions, FloatType},
        SimConstants::SimulationConstants,
        SimLogger::SimulationLogger
        ) where {Dimensions,FloatType}
    
        # Unpack the relevant simulation meta data
        @unpack HourGlass = SimMetaData;
    
        # Load in particles
        SimParticles, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺ = AllocateDataStructures(Dimensions,FloatType, SimGeometry)
        
        # Hardcode loading of mdbc particles for now
        _, GhostPoints, GhostNormals = LoadBoundaryNormals(Dimensions, FloatType, SimGeometry[:FixedBoundary]["GhostNodes"])

        if SimMetaData.FlagLog
            InitializeLogger(SimLogger,SimConstants,SimMetaData, SimGeometry, SimParticles)
        end
        
        NumberOfPoints = length(SimParticles)::Int #Have to type declare, else error?
        @inline Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
    
        # Shifting correction
        ∇Cᵢ               = zeros(SVector{Dimensions,FloatType},NumberOfPoints)            
        ∇◌rᵢ              = zeros(FloatType,NumberOfPoints)    
    
        @inline begin
            n_copy = Base.Threads.nthreads()
            KernelThreaded         = [copy(SimParticles.Kernel)         for _ in 1:n_copy]
            KernelGradientThreaded = [copy(SimParticles.KernelGradient) for _ in 1:n_copy]
            dρdtIThreaded          = [copy(dρdtI)                       for _ in 1:n_copy]
            AccelerationThreaded   = [copy(SimParticles.KernelGradient) for _ in 1:n_copy]
            ∇CᵢThreaded            = [copy(∇Cᵢ )                        for _ in 1:n_copy]
            ∇◌rᵢThreaded           = [copy(∇◌rᵢ)                        for _ in 1:n_copy]   
        end
    
        # Produce sorting related variables
        ParticleRanges         = zeros(Int, NumberOfPoints + 1)
        UniqueCells            = zeros(CartesianIndex{Dimensions}, NumberOfPoints)
        Stencil                = ConstructStencil(Val(Dimensions))
        _, SortingScratchSpace = Base.Sort.make_scratch(nothing, eltype(SimParticles), NumberOfPoints)
    
        # Produce data saving functions
        SaveLocation_   = SimMetaData.SaveLocation * "/" * SimMetaData.SimulationName
        SaveLocation    = (Iteration) -> SaveLocation_ * "_" * lpad(Iteration,6,"0") * ".vtkhdf"
        SaveLocation_g  = (Iteration) -> SaveLocation_ * "_GhostNodes_" * lpad(Iteration,6,"0") * ".vtkhdf"
    
        fid_vector    = Vector{HDF5.File}(undef, Int(SimMetaData.SimulationTime/SimMetaData.OutputEach + 1))
        fid_vector_g  = Vector{HDF5.File}(undef, 1)
    
        OutputVariableNames = ["Kernel", "KernelGradient", "Density", "Pressure","Velocity", "Acceleration", "BoundaryBool" , "ID", "Type", "GroupMarker"]
        if Dimensions == 2
            SaveFile   = (Index) -> SaveVTKHDF(fid_vector, Index, SaveLocation(Index),to_3d(SimParticles.Position), OutputVariableNames, SimParticles.Kernel, SimParticles.KernelGradient, SimParticles.Density, SimParticles.Pressure, SimParticles.Velocity, SimParticles.Acceleration, SimParticles.BoundaryBool, SimParticles.ID, UInt8.(SimParticles.Type), SimParticles.GroupMarker)
        else
            SaveFile   = (Index) -> SaveVTKHDF(fid_vector, Index, SaveLocation(Index),SimParticles.Position, OutputVariableNames, SimParticles.Kernel, SimParticles.KernelGradient, SimParticles.Density, SimParticles.Pressure, SimParticles.Velocity, SimParticles.Acceleration, SimParticles.BoundaryBool, SimParticles.ID, UInt8.(SimParticles.Type), SimParticles.GroupMarker)
        end

        SaveVTKHDF(fid_vector_g, 1, SaveLocation_g(1), to_3d(GhostPoints), ["Normals"], GhostNormals)
        @. close(fid_vector_g)

        SimMetaData.OutputIterationCounter += 1 #Since a file has been saved
        @inline SaveFile(SimMetaData.OutputIterationCounter)
        
        InverseCutOff = Val(1/(SimConstants.H))

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
    
            SimulationLoop(SimMetaData, SimConstants, SimParticles, GhostPoints, GhostNormals, Stencil, ParticleRanges, UniqueCells, SortingScratchSpace, KernelThreaded, KernelGradientThreaded, dρdtI, dρdtIThreaded, AccelerationThreaded, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇CᵢThreaded, ∇◌rᵢ, ∇◌rᵢThreaded, MotionDefinition, InverseCutOff)
    

            if SimMetaData.TotalTime >= SimMetaData.OutputEach * SimMetaData.OutputIterationCounter
    
                try 
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
             
                    AutoOpenLogFile(SimLogger, SimMetaData)
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

                AutoOpenParaview(SaveLocation_, SimMetaData, OutputVariableNames)
                
                break
            end
        end
    end
    

end
