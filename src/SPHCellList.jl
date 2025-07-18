module SPHCellList

export ConstructStencil, ExtractCells!, UpdateNeighbors!, NeighborLoop!, ComputeInteractions!, RunSimulation

using Parameters, FastPow, StaticArrays, Base.Threads, ChunkSplitters
import LinearAlgebra: dot

using ..SimulationEquations
using ..SimulationGeometry
using ..AuxiliaryFunctions
using ..SimulationMetaDataConfiguration
using ..SimulationConstantsConfiguration
using ..SimulationLoggerConfiguration
using ..PreProcess
using ..ProduceHDFVTK
using ..TimeStepping
using ..OpenExternalPrograms
using ..SPHKernels
using ..SPHViscosityModels
using ..SPHDensityDiffusionModels

using StaticArrays
import StructArrays: StructArray, foreachfield
import LinearAlgebra: dot, norm, diagm, diag, cond, det
import Parameters: @unpack
import FastPow: @fastpow
import ProgressMeter: next!, finish!
using Format
using TimerOutputs
using Logging, LoggingExtras
using HDF5
using Base.Threads
using UnicodePlots
using LinearAlgebra
using Bumper

    function ConstructStencil(v::Val{d}) where d
        n_ = CartesianIndices(ntuple(_->-1:1,v))
        half_length = length(n_) ÷ 2
        n  = n_[1:half_length]

        return n
    end

    """
    Extracts the cells for each particle based on their positions and the inverse cutoff value.

    # Arguments
    - `Particles`: The particles whose cells are to be extracted.
    - `::Val{InverseCutOff}`: The inverse cutoff value used for cell extraction.

    # Returns
    - `nothing`: This function modifies the `Particles` in place.
    """
    # Replace unsafe_trunc with trunc if this ever errors
    @inline function map_floor(x, InverseCutOff)
        # This is different than just doing muladd(x,InverseCutOff,0.5) because it rounds towards zero.
        # Consider -1.7 + 0.5, this would give -1.2 and then trunced 1, but we want -2, therefore absolute addition before hand
        # We add 0.5 instead of 1, to ensure proper rounding behavior when restoring the sign for negative numbers.
        Int(sign(x)) * unsafe_trunc(Int, muladd(abs(x),InverseCutOff,0.5))
    end
   
    @inline function ExtractCells!(Particles, InverseCutOff)
        @inbounds @simd ivdep for i ∈ eachindex(Particles.Cells)
            # t = map(map_floor, Tuple(Particles.Position[i]))
            t = CartesianIndex(map(x -> map_floor(x, InverseCutOff), Tuple(Particles.Position[i])))
            Particles.Cells[i] = CartesianIndex(t)
        end

        return nothing
    end

    """
    Updates the neighbor list and sorts particles by their cell indices.

    # Arguments
    - `Particles`: The particles whose neighbors are to be updated.
    - `CutOff`: The cutoff value used for cell extraction.
    - `SortingScratchSpace`: Scratch space for sorting.
    - `ParticleRanges`: Array to store the ranges of particles in each cell.
    - `UniqueCells`: Array to store the unique cells.

    # Returns
    - `IndexCounter`: The number of unique cells identified.
    """
    function UpdateNeighbors!(Particles, InverseCutOff, SortingScratchSpace,
                              ParticleRanges, UniqueCells, CellDict)
        ExtractCells!(Particles, InverseCutOff)

        sort!(Particles, by = p -> p.Cells; scratch=SortingScratchSpace)
        Cells = @views Particles.Cells
        @. ParticleRanges             = zero(eltype(ParticleRanges))
        ParticleRanges[1] = 1
        IndexCounter                  = 2
        ParticleRanges[IndexCounter]  = 1
        UniqueCells[IndexCounter]     = Cells[1]
        empty!(CellDict)
        CellDict[Cells[1]] = IndexCounter

        @inbounds @simd ivdep for i in eachindex(Cells)[2:end]
            if Cells[i] != Cells[i-1] # Equivalent to diff(Cells) != 0
                IndexCounter                 += 1
                ParticleRanges[IndexCounter]  = i
                UniqueCells[IndexCounter]     = Cells[i]
                CellDict[Cells[i]]           = IndexCounter
            end
        end
        ParticleRanges[IndexCounter + 1]  = length(ParticleRanges)

        return IndexCounter 
    end

    
# Neither Polyester.@batch per core or thread is faster
###=== Function to process each cell and its neighbors
    function NeighborLoop!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                           SimMetaData, SimConstants, SimParticles,
                           SimThreadedArrays, ParticleRanges, CellDict, Stencil,
                           Position, Density, Pressure, Velocity, MotionLimiter,
                           UniqueCells, EnumeratedIndices) where {SDD<:SPHDensityDiffusion, SV<:SPHViscosity}
        @sync begin
            for (ichunk, inds) ∈ EnumeratedIndices 
                @spawn for iter ∈ inds

                    CellIndex = UniqueCells[iter]
                    SimParticles.ChunkID[iter] = ichunk

                    StartIndex = ParticleRanges[iter]
                    EndIndex   = ParticleRanges[iter+1] - 1

                    @inbounds for i = StartIndex:EndIndex, j = (i+1):EndIndex
                        @inline ComputeInteractions!(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants, SimParticles, SimThreadedArrays, Position, Density, Pressure, Velocity, i, j, MotionLimiter, ichunk)
                    end

                    @inbounds for S ∈ Stencil
                        SCellIndex = CellIndex + S

                        # Returns a range, x>:x for exact match and x=:x for no match
                        # utilizes that it is a sorted array and requires no isequal constructor,
                        # so I prefer this for now
                        NeighborIdx = get(CellDict, SCellIndex, 1)

                        StartIndex_       = ParticleRanges[NeighborIdx]
                        EndIndex_         = ParticleRanges[NeighborIdx + 1] - 1

                        @inbounds for i = StartIndex:EndIndex, j = StartIndex_:EndIndex_
                            @inline ComputeInteractions!(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants, SimParticles, SimThreadedArrays, Position, Density, Pressure, Velocity, i, j, MotionLimiter, ichunk)
                        end
                    end
                end
            end
        end
        
        return nothing
    end

    f(SimKernel, GhostPoint) = CartesianIndex(map(x->map_floor(x,SimKernel.H⁻¹), Tuple(GhostPoint)))
    function NeighborLoopMDBC!(SimKernel,
                               SimMetaData::SimulationMetaData{Dimensions, _},
                               SimConstants, ParticleRanges, CellDict, Position,
                               Density, GhostPoints, GhostNormals, ParticleType,
                               bᵧ, Aᵧ) where {Dimensions, _}
        
        FullStencil = CartesianIndices(ntuple(_->-1:1, Dimensions))

        @inbounds @threads for iter in eachindex(GhostPoints)
            GhostPoint = GhostPoints[iter]

            if !iszero(GhostPoint)
                # zero‐initialize per‐ghost accumulators
                b_acc = zero(bᵧ[iter])            # an SVector{D+1,FloatType}
                A_acc = zero(Aᵧ[iter])            # an SMatrix{D+1,D+1,FloatType}
            
                # compute and accumulate into the locals
                GhostCellIndex = f(SimKernel, GhostPoints[iter])
                @inbounds for S ∈ FullStencil
                    SCellIndex = GhostCellIndex + S

                    # Returns a range, x>:x for exact match and x=:x for no match
                    # utilizes that it is a sorted array and requires no isequal constructor,
                    # so I prefer this for now
                    NeighborIdx = get(CellDict, SCellIndex, 1)

                    StartIndex_       = ParticleRanges[NeighborIdx] 
                    EndIndex_         = ParticleRanges[NeighborIdx + 1] - 1

                    for j in StartIndex_:EndIndex_
                        # change ComputeInteractions to take & return contributions, e.g.:
                        bΔ, AΔ = ComputeInteractionsMDBC!(SimKernel, SimMetaData, SimConstants,
                                                        Position, Density, ParticleType,
                                                        GhostPoints, iter, j)
                        b_acc += bΔ
                        A_acc += AΔ
                    end
                end
            
                # write out once
                bᵧ[iter] = b_acc
                Aᵧ[iter] = A_acc
            end    
        end

        return nothing
    end

    Base.@propagate_inbounds function ComputeInteractions!(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel, SimMetaData, SimConstants, SimParticles, SimThreadedArrays, Position, Density, Pressure, Velocity, i, j, MotionLimiter, ichunk) where {SDD<:SPHDensityDiffusion, SV<:SPHViscosity}
        @unpack FlagOutputKernelValues = SimMetaData
        @unpack ρ₀, m₀, α, γ, g, c₀, δᵩ, Cb, Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants

        @unpack h⁻¹, h, η², H², αD = SimKernel 

        xᵢⱼ  = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)              
        if  xᵢⱼ² <= H²
            #https://discourse.julialang.org/t/sqrt-abs-x-is-even-faster-than-sqrt/58154/2
            dᵢⱼ  = sqrt(abs(xᵢⱼ²))

            # clamp seems faster than min, no util
            q         = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0) #min(dᵢⱼ * h⁻¹, 2.0) - 8% util no DDT
            ∇ᵢWᵢⱼ     = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)
            
            ρᵢ        = Density[i]
            ρⱼ        = Density[j]
        
            vᵢ        = Velocity[i]
            vⱼ        = Velocity[j]
            vᵢⱼ       = vᵢ - vⱼ
            density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
            dρdt⁺          = - ρᵢ * (m₀/ρⱼ) *  density_symmetric_term
            dρdt⁻          = - ρⱼ * (m₀/ρᵢ) *  density_symmetric_term

            Dᵢ, Dⱼ = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, xᵢⱼ², i, j, MotionLimiter)

            SimThreadedArrays.dρdtIThreaded[ichunk][i] += dρdt⁺ + Dᵢ
            SimThreadedArrays.dρdtIThreaded[ichunk][j] += dρdt⁻ + Dⱼ


            Pᵢ      =  Pressure[i]
            Pⱼ      =  Pressure[j]
            Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
            f_ab    = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
            dvdt⁺   = - m₀ * (Pfac + f_ab) *  ∇ᵢWᵢⱼ

            visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, dᵢⱼ, i, j)

            uₘ = dvdt⁺ + visc_term
            SimThreadedArrays.AccelerationThreaded[ichunk][i] += uₘ
            SimThreadedArrays.AccelerationThreaded[ichunk][j] -= uₘ 

            
            if FlagOutputKernelValues
                Wᵢⱼ  = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)
                SimThreadedArrays.KernelThreaded[ichunk][i]         += Wᵢⱼ
                SimThreadedArrays.KernelThreaded[ichunk][j]         += Wᵢⱼ
                SimThreadedArrays.KernelGradientThreaded[ichunk][i] +=  ∇ᵢWᵢⱼ
                SimThreadedArrays.KernelGradientThreaded[ichunk][j] += -∇ᵢWᵢⱼ
            end

            if SimMetaData.FlagShifting
                
                MLcond = MotionLimiter[i] * MotionLimiter[j]

                SimThreadedArrays.∇CᵢThreaded[ichunk][i]   += (m₀/ρᵢ) *  ∇ᵢWᵢⱼ
                SimThreadedArrays.∇CᵢThreaded[ichunk][j]   += (m₀/ρⱼ) * -∇ᵢWᵢⱼ
        
                # Switch signs compared to DSPH, else free surface detection does not make sense
                # Agrees, https://arxiv.org/abs/2110.10076, it should have been r_ji
                SimThreadedArrays.∇◌rᵢThreaded[ichunk][i]  += (m₀/ρⱼ) * dot(-xᵢⱼ , ∇ᵢWᵢⱼ)  * MLcond
                SimThreadedArrays.∇◌rᵢThreaded[ichunk][j]  += (m₀/ρᵢ) * dot( xᵢⱼ ,-∇ᵢWᵢⱼ)  * MLcond
            end
        end

        return nothing
    end

    Base.@propagate_inbounds function ComputeInteractionsMDBC!(SimKernel, SimMetaData::SimulationMetaData{Dimensions, FloatType}, SimConstants, Position, Density, ParticleType, GhostPoints, i, j) where {Dimensions, FloatType}
        @unpack ρ₀, m₀, α, γ, g, c₀, δᵩ, Cb, Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants
        
        @unpack h⁻¹, h, η², H², αD = SimKernel 

        DimensionsPlus = Dimensions + 1
        # always zero‐initialize
        bΔ = zero(SVector{DimensionsPlus,FloatType})
        AΔ = zero(SMatrix{DimensionsPlus, DimensionsPlus,FloatType})

        # ᵢ is ghost node! ⱼ is fluid node

        if ParticleType[j] == Fluid

            xᵢⱼ  = GhostPoints[i] - Position[j]

            xᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)
            if xᵢⱼ² <= H²
                dᵢⱼ = sqrt(abs(xᵢⱼ²))
                q = clamp(dᵢⱼ * h⁻¹, 0.0, 2.0)
        
                ρⱼ = Density[j]

        
                Wᵢⱼ = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)

                ∇ᵢWᵢⱼ = @fastpow ∇Wᵢⱼ(SimKernel, q, xᵢⱼ)

                Vⱼ = m₀ / ρⱼ
        
                VⱼWᵢⱼ = Vⱼ * Wᵢⱼ
        
                bΔ  = SVector{DimensionsPlus, FloatType}(m₀ * Wᵢⱼ, (m₀ * ∇ᵢWᵢⱼ)...)

                # Filling the Aᵧ matrix is done in column-major order
                xⱼᵢ = -xᵢⱼ
                first_column = [VⱼWᵢⱼ; Vⱼ * ∇ᵢWᵢⱼ]
                AΔ = SMatrix{DimensionsPlus, DimensionsPlus, FloatType, DimensionsPlus*DimensionsPlus}(
                    first_column...,
                    ((xⱼᵢ * first_column')')...
                )
            end
        end
        
    
        return bΔ, AΔ
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
                @simd ivdep for i in start_idx:end_idx
                    @inbounds target_array[i] += array[i]
                end
            end
        end
    end

    function ResetStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ)
        # Threaded zeroing for main arrays
        @threads for arr in (dρdtI, Acceleration)
            fill!(arr, zero(eltype(arr)))
        end

        if SimMetaData.FlagOutputKernelValues
            @threads for arr in (Kernel, KernelGradient)
                fill!(arr, zero(eltype(arr)))
            end
        end

        if SimMetaData.FlagShifting
            @threads for arr in (∇Cᵢ, ∇◌rᵢ)
                fill!(arr, zero(eltype(arr)))
            end
        end

        # Threaded zeroing for fields in SimThreadedArrays
        foreachfield(f -> begin
            Threads.@threads for v in f
                fill!(v, zero(eltype(v)))
            end
        end, SimThreadedArrays)

        return nothing
    end

    function ReductionStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ)
        reduce_sum!(dρdtI, SimThreadedArrays.dρdtIThreaded)
        reduce_sum!(Acceleration, SimThreadedArrays.AccelerationThreaded)
  
        if SimMetaData.FlagOutputKernelValues
            reduce_sum!(Kernel, SimThreadedArrays.KernelThreaded)
            reduce_sum!(KernelGradient, SimThreadedArrays.KernelGradientThreaded)
        end

        if SimMetaData.FlagShifting
            reduce_sum!(∇Cᵢ, SimThreadedArrays.∇CᵢThreaded)
            reduce_sum!(∇◌rᵢ, SimThreadedArrays.∇◌rᵢThreaded)
        end
    
        return nothing
    end
    
    ### Some functions to simplify code inside of this function
    function ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionsDefinition, SimMetaData)
        @inbounds @simd ivdep for i in eachindex(Position)
            if ParticleType[i] == Moving
                motion = MotionsDefinition[ParticleMarker[i]]
    
                if motion !== nothing
                    ShouldMove = (motion.StartTime <= SimMetaData.TotalTime) &&
                                 (SimMetaData.TotalTime <= (motion.StartTime + motion.Duration))
    
                    # Retrieve motion parameters
                    MotionVel = motion.Velocity
                    MotionDir = motion.Direction
    
                    # Update Velocity and Position
                    Velocity[i] = MotionVel * MotionDir * ShouldMove
                    Position[i] += Velocity[i] * dt₂
                end
            end
        end

        return nothing
    end

    function ApplyMDBCCorrection(SimConstants, SimParticles, bᵧ, Aᵧ)

        Position    = SimParticles.Position
        Density     = SimParticles.Density
        GhostPoints = SimParticles.GhostPoints

        ρ₀ = SimConstants.ρ₀
        #https://github.com/DualSPHysics/DualSPHysics/blob/f4fa76ad5083873fa1c6dd3b26cdce89c55a9aeb/src/source/JSphCpu_mdbc.cpp#L347
        @inbounds @simd ivdep for i in eachindex(Position)
            A = Aᵧ[i]

            # Since Aᵧ is not reset anymore, we need to check if it is zero
            if !iszero(GhostPoints[i])
                if abs(det(A)) >= 1e-3
                        GhostPointDensity = A \ bᵧ[i]
                        diff = Position[i] - GhostPoints[i]
                        v1   = first(GhostPointDensity) + sum(GhostPointDensity[j+1] * diff[j] for j in eachindex(diff))
                        Density[i] = isnan(v1) ? ρ₀ : v1
                elseif first(A) > 0.0
                        v = first(bᵧ[i]) / first(A)
                        Density[i] = isnan(v) ? ρ₀ : v
                end
            end
        end
    end
    
    function HalfTimeStep(::SimulationMetaData{Dimensions, FloatType}, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂) where {Dimensions, FloatType}
        @unpack Position, Density, Velocity, Acceleration, GravityFactor, MotionLimiter = SimParticles

        @inbounds @simd ivdep for i in eachindex(Position)
            Acceleration[i]  +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
            Positionₙ⁺[i]     =  Position[i]   + Velocity[i]   * dt₂  * MotionLimiter[i]
            Velocityₙ⁺[i]     =  Velocity[i]   + Acceleration[i]  *  dt₂ * MotionLimiter[i]
            ρₙ⁺[i]            =  Density[i]    + dρdtI[i]       *  dt₂
        end


        return nothing
    end

    function FullTimeStep(SimMetaData, SimKernel, SimConstants, SimParticles, ∇Cᵢ, ∇◌rᵢ, dt)
        @unpack Position, Velocity, Acceleration, GravityFactor, MotionLimiter = SimParticles
  
        if !SimMetaData.FlagShifting
            @inbounds @simd ivdep for i in eachindex(Position)
                Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
                Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
                Position[i]       +=  (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt) * MotionLimiter[i]
            end
        else
            A     = 2# Value between 1 to 6 advised
            A_FST = 0; # zero for internal flows
            A_FSM = length(first(Position)); #2d, 3d val different
            @inbounds @simd ivdep for i in eachindex(Position)
                Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
                Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
        
                A_FSC                  = (∇◌rᵢ[i] - A_FST)/(A_FSM - A_FST)
                if A_FSC < 0
                    δxᵢ = zero(eltype(Position))
                else
                    δxᵢ = -A_FSC * A * SimKernel.h * norm(Velocity[i]) * dt * ∇Cᵢ[i]
                end
        
                Position[i]           += (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt + δxᵢ) * MotionLimiter[i]
            end
        end

        return nothing
    end

    function UpdateMetaData!(SimMetaData, dt)
        SimMetaData.Iteration      += 1
        SimMetaData.CurrentTimeStep = dt
        SimMetaData.TotalTime      += dt

        return nothing
    end

    @inline next_output_time(SimMetaData) = next_output_time(SimMetaData.OutputTimes, SimMetaData)

    @inline next_output_time(interval::Real, SimMetaData) = interval * SimMetaData.OutputIterationCounter

    @inline function next_output_time(times::AbstractVector, SimMetaData)
        idx = SimMetaData.OutputIterationCounter
        if idx < length(times)
            return times[idx]
        else
            return SimMetaData.SimulationTime
        end
    end

    """
        update_delta_x!(Δx, posₙ⁺, pos)

    Increment Δx by twice the maximum ‖posₙ⁺[i] – pos[i]‖, without ever allocating.
    Returns the new Δx.
    """
    @inline function update_delta_x!(Δx::T,
                                    posₙ⁺::AbstractVector{SVector{D, T}},
                                    pos   ::AbstractVector{SVector{D, T}}) where {D, T<:Real}
        maxd = zero(T)
        @inbounds for i in eachindex(posₙ⁺, pos)
            # compute squared norm manually
            sumsq = zero(T)
            @inbounds for j in 1:D
                d = posₙ⁺[i][j] - pos[i][j]
                sumsq += d*d
            end
            # sqrt/T is allocation-free on scalars
            nrm = sqrt(sumsq)
            if nrm > maxd
                maxd = nrm
            end
        end
        return Δx + 4*maxd
    end

    
    @inbounds function SimulationLoop(SimDensityDiffusion::SDD, SimViscosity::SV, SimKernel,
                                      SimMetaData::SimulationMetaData{Dimensions, FloatType},
                                      SimConstants, SimParticles, Stencil,
                                      ParticleRanges, UniqueCells, CellDict,
                                      SortingScratchSpace, SimThreadedArrays,
                                      dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺,
                                      ∇Cᵢ, ∇◌rᵢ, MotionDefinition) where {Dimensions, FloatType, SDD<:SPHDensityDiffusion, SV<:SPHViscosity}
        Position       = SimParticles.Position
        Density        = SimParticles.Density
        Pressure       = SimParticles.Pressure
        Velocity       = SimParticles.Velocity
        Acceleration   = SimParticles.Acceleration
        MotionLimiter  = SimParticles.MotionLimiter
        ParticleType   = SimParticles.Type
        ParticleMarker = SimParticles.GroupMarker
        Kernel         = SimParticles.Kernel
        KernelGradient = SimParticles.KernelGradient
        GhostPoints    = SimParticles.GhostPoints
        GhostNormals   = SimParticles.GhostNormals

        ###
        DimensionsPlus = Dimensions + 1
        Δx = one(eltype(Density)) + SimKernel.h
        UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
        EnumeratedIndices = enumerate(index_chunks(UniqueCellsView; n=nthreads()))
        @no_escape begin
            while SimMetaData.TotalTime <= next_output_time(SimMetaData)

                Δx = update_delta_x!(Δx, Positionₙ⁺, SimParticles.Position)

                # println("Δx: ", Δx, "h: ", SimKernel.h," dt: ", SimMetaData.CurrentTimeStep, " Iteration: ", SimMetaData.Iteration, " TotalTime: ", SimMetaData.TotalTime, " OutputIterationCounter: ", SimMetaData.OutputIterationCounter)

                @timeit SimMetaData.HourGlass "01 Update TimeStep"  dt  = Δt(Position, Velocity, Acceleration, SimConstants, SimKernel)
                dt₂ = dt * 0.5

                @timeit SimMetaData.HourGlass "02 Calculate IndexCounter"  begin
                    # Note: If particles are not inside of the neighbor list visualiation, try setting this if statement to always true, since UniqueCells will be updated always then
                    # In theory, the maximal speed is the speed of sound, this should give a safe guard
                    # and ensure it is always updated in a reasonable manner. This only works well, assuming that
                    # c₀ >= maximum(norm.(Velocity))
                    # Remove if statement logic if you want to update each iteration
                    # if mod(SimMetaData.Iteration, ceil(Int, SimKernel.H / (SimConstants.c₀ * dt * (1/SimConstants.CFL)) )) == 0 || SimMetaData.Iteration == 1
                    if Δx >= SimKernel.h
                        @timeit SimMetaData.HourGlass "02a Actual Calculate IndexCounter" SimMetaData.IndexCounter = UpdateNeighbors!(SimParticles, SimKernel.H⁻¹, SortingScratchSpace,  ParticleRanges, UniqueCells, CellDict)
                        Δx = zero(eltype(Density))
                        UniqueCellsView   = view(UniqueCells, 1:SimMetaData.IndexCounter)
                        EnumeratedIndices = enumerate(index_chunks(UniqueCellsView; n=nthreads()))
                    end
                end

                @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)
            
                if !SimMetaData.FlagSingleStepTimeStepping
                    ###=== First step of resetting arrays
                    @timeit SimMetaData.HourGlass "ResetArrays"                          ResetStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ)
                    ###===
                
                    @timeit SimMetaData.HourGlass "03 Pressure"                          Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
                    if SimMetaData.FlagMDBCSimple
                        bᵧ = @alloc(SVector{DimensionsPlus, FloatType}, length(Position))
                        Aᵧ = @alloc(SMatrix{DimensionsPlus, DimensionsPlus, FloatType, DimensionsPlus * DimensionsPlus}, length(Position))
                        @timeit SimMetaData.HourGlass "04a First NeighborLoopMDBC"           NeighborLoopMDBC!(SimKernel, SimMetaData, SimConstants, ParticleRanges, CellDict, Position, Density, GhostPoints, GhostNormals, ParticleType, bᵧ, Aᵧ)
                        @timeit SimMetaData.HourGlass "04b Apply MDBC before Half TimeStep"  ApplyMDBCCorrection(SimConstants, SimParticles, bᵧ, Aᵧ)
                    end

                    @timeit SimMetaData.HourGlass "04 First NeighborLoop"                NeighborLoop!(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants, SimParticles, SimThreadedArrays, ParticleRanges, CellDict, Stencil, Position, Density, Pressure, Velocity, MotionLimiter, UniqueCellsView, EnumeratedIndices)
                    @timeit SimMetaData.HourGlass "Reduction"                            ReductionStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ)
                end


                @timeit SimMetaData.HourGlass "05b Update To Half TimeStep"              HalfTimeStep(SimMetaData, SimConstants, SimParticles, Positionₙ⁺, Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂)


                @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary"           LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, MotionLimiter)
            
                ###=== Second step of resetting arrays
                @timeit SimMetaData.HourGlass "ResetArrays"                              ResetStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ)
                ###===

                @timeit SimMetaData.HourGlass "Motion"                                   ProgressMotion(Position, Velocity, ParticleType, ParticleMarker, dt₂, MotionDefinition, SimMetaData)
            
                @timeit SimMetaData.HourGlass "03 Pressure"                              Pressure!(SimParticles.Pressure, ρₙ⁺,SimConstants)
                @timeit SimMetaData.HourGlass "08 Second NeighborLoop"                   NeighborLoop!(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants, SimParticles, SimThreadedArrays, ParticleRanges, CellDict, Stencil, Positionₙ⁺, ρₙ⁺, Pressure, Velocityₙ⁺, MotionLimiter, UniqueCellsView, EnumeratedIndices)
                @timeit SimMetaData.HourGlass "Reduction"                                ReductionStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ)

            
                @timeit SimMetaData.HourGlass "09 Final LimitDensityAtBoundary"          LimitDensityAtBoundary!(Density, SimConstants.ρ₀, MotionLimiter)
            
                @timeit SimMetaData.HourGlass "10 Final Density"                         DensityEpsi!(Density, dρdtI, ρₙ⁺, dt)
            
                @timeit SimMetaData.HourGlass "11 Update To Final TimeStep"              FullTimeStep(SimMetaData, SimKernel, SimConstants, SimParticles, ∇Cᵢ, ∇◌rᵢ, dt)
            
                @timeit SimMetaData.HourGlass "12 Update MetaData"                       UpdateMetaData!(SimMetaData, dt)

            end
        end
        
        return nothing
    end
    
    ###===
    function RunSimulation(;SimGeometry::Vector{Geometry{Dimensions, FloatType}}, #Don't further specify type for now
        SimMetaData::SimulationMetaData{Dimensions, FloatType},
        SimConstants::SimulationConstants,
        SimKernel::SPHKernelInstance,
        SimLogger::SimulationLogger,
        SimParticles::StructArray,
        SimViscosity::SV,
        SimDensityDiffusion::SDD,
        ParticleNormalsPath::Union{Nothing,String} = nothing
        ) where {Dimensions,FloatType,SV<:SPHViscosity,SDD<:SPHDensityDiffusion}

        # Unpack the relevant simulation meta data
        @unpack HourGlass = SimMetaData;

        # Vector of time steps
        TimeSteps = Vector{FloatType}()
        
        dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ = AllocateSupportDataStructures(SimParticles.Position)

        # Implement this properly later in shaa Allah
        # if !isnothing(ParticleNormalsPath)
            if SimMetaData.FlagMDBCSimple
                _, GhostPoints, GhostNormals = LoadBoundaryNormals(Val(Dimensions), FloatType, ParticleNormalsPath)
            

                #TODO: In the future decide on one of the two in shaa Allah
                for gi ∈ eachindex(GhostPoints)
                    SimParticles.GhostPoints[gi]  = GhostPoints[gi]
                    SimParticles.GhostNormals[gi] = GhostNormals[gi]
                end
            end
        # end

        if !SimMetaData.FlagShifting
            resize!(∇Cᵢ , 0)
            resize!(∇◌rᵢ, 0)
        end

        if SimMetaData.FlagLog
            InitializeLogger(SimLogger, SimConstants, SimMetaData, SimKernel, SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)
        end

        # To generate first line
        if SimMetaData.FlagLog
            LogStep(SimLogger, SimMetaData, HourGlass)
            SimMetaData.StepsTakenForLastOutput = SimMetaData.Iteration
        end
        
        NumberOfPoints = length(SimParticles)::Int
        Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
    
        SimThreadedArrays = AllocateThreadedArrays(SimMetaData, SimParticles, dρdtI, ∇Cᵢ, ∇◌rᵢ)
    
        # Produce sorting related variables
        ParticleRanges         = zeros(Int, NumberOfPoints + 1 + 1) # +1 for the last particle, +1 for dummy entry
        UniqueCells            = zeros(CartesianIndex{Dimensions}, NumberOfPoints)
        CellDict               = Dict{CartesianIndex{Dimensions}, Int}()
        Stencil                = ConstructStencil(Val(Dimensions))
        _, SortingScratchSpace = Base.Sort.make_scratch(nothing, eltype(SimParticles), NumberOfPoints)

        output = SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)

        # Save initial state, use 1 else this cannot be used to index fid vector
        SimMetaData.OutputIterationCounter = 1
        output.save_particles(SimMetaData.OutputIterationCounter)
        output.save_grid(SimMetaData.OutputIterationCounter, UniqueCells, SimParticles)


        # Assuming group markers are sequential
        MotionDefinition = Vector{Union{Nothing, MotionDetails{Dimensions, FloatType}}}(undef, maximum(SimParticles.GroupMarker))

        for geom in SimGeometry
            group_marker = geom.GroupMarker
            if geom.Motion !== nothing
                MotionDefinition[group_marker] = geom.Motion
            else
                MotionDefinition[group_marker] = nothing
            end
        end

        # Normal run and save data
        generate_showvalues(Iteration, TotalTime, TimeLeftInSeconds) = () -> [(:(Iteration),format(FormatExpr("{1:d}"),  Iteration)), (:(TotalTime),format(FormatExpr("{1:3.3f}"), TotalTime)), (:(TimeLeftInSeconds),format(FormatExpr("{1:3.1f} [s]"), TimeLeftInSeconds))]
        

        if !SimLogger.ToConsole
            @timeit HourGlass "14 Next TimeStep" next!(
                SimMetaData.ProgressSpecification;
                showvalues = generate_showvalues(
                    SimMetaData.Iteration,
                    SimMetaData.TotalTime,
                    1e6,
                ),
            )
        end

        @inbounds while true

            @timeit SimMetaData.HourGlass "00 SimulationLoop" SimulationLoop(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants, SimParticles, Stencil, ParticleRanges, UniqueCells, CellDict, SortingScratchSpace, SimThreadedArrays, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ, MotionDefinition)
            push!(TimeSteps, SimMetaData.CurrentTimeStep)

            if SimMetaData.FlagLog
                LogStep(SimLogger, SimMetaData, HourGlass)
                SimMetaData.StepsTakenForLastOutput = SimMetaData.Iteration
            end
    
            SimMetaData.OutputIterationCounter += 1

            UniqueCellsView = view(UniqueCells, 1:SimMetaData.IndexCounter)
            @sync Threads.@spawn begin
                @timeit SimMetaData.HourGlass "13A Save Particle Data" output.save_particles(SimMetaData.OutputIterationCounter)
                @timeit SimMetaData.HourGlass "13A Save CellGrid Data" output.save_grid(SimMetaData.OutputIterationCounter, UniqueCellsView, SimParticles)
            end
    
            if !SimLogger.ToConsole
                TimeLeftInSeconds = (SimMetaData.SimulationTime - SimMetaData.TotalTime) *
                                    (TimerOutputs.tottime(HourGlass) / 1e9 / SimMetaData.TotalTime)
                @timeit HourGlass "14 Next TimeStep" next!(
                    SimMetaData.ProgressSpecification;
                    showvalues = generate_showvalues(
                        SimMetaData.Iteration,
                        SimMetaData.TotalTime,
                        TimeLeftInSeconds,
                    ),
                )
            end
    
            if SimMetaData.TotalTime > SimMetaData.SimulationTime
                
                # At end of simulation
                @timeit SimMetaData.HourGlass "13B Close Data Streams" output.close_files()

                if !SimLogger.ToConsole
                    finish!(SimMetaData.ProgressSpecification)
                end
                show(HourGlass,sortby=:name)
                show(HourGlass)

                AutoOpenParaview(SimMetaData, output.variable_names)

                # Time steps line plot
                UnicodeTimeStepsGraph = lineplot(1:length(TimeSteps), TimeSteps, title="Time Steps [s] as a function of iteration", name="Time Steps", xlabel="Iterations [-]", ylabel="Time Step Size [s]")

                if SimMetaData.FlagLog
                    LogFinal(SimLogger, HourGlass)
             
                    with_logger(SimLogger.Logger) do
                        @info ""
                        show(SimLogger.LoggerIo, UnicodeTimeStepsGraph)
                    end

                    close(SimLogger.LoggerIo)
                    AutoOpenLogFile(SimLogger, SimMetaData)
                end

                break
            end
        end
    end
    

end
