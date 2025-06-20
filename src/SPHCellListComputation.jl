module SPHCellListComputation

using Parameters, FastPow, StaticArrays, Base.Threads, ChunkSplitters
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

import StructArrays: StructArray, foreachfield
import LinearAlgebra: dot, norm, diagm, diag, cond, det
import Parameters: @unpack
import FastPow: @fastpow
import ProgressMeter: next!, finish!
using Format
using TimerOutputs
using Logging, LoggingExtras
using HDF5
using UnicodePlots
using LinearAlgebra
using Bumper

Base.@propagate_inbounds function ComputeInteractions!(SimDensityDiffusion, SimViscosity, SimKernel, SimMetaData, SimConstants, SimParticles, SimThreadedArrays, Position, Density, Pressure, Velocity, i, j, MotionLimiter, ichunk)
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

        Dᵢ, Dⱼ = compute_density_diffusion(SimDensityDiffusion, SimKernel, SimConstants, SimParticles, xᵢⱼ, ∇ᵢWᵢⱼ, i, j, MotionLimiter)

        SimThreadedArrays.dρdtIThreaded[ichunk][i] += dρdt⁺ + Dᵢ
        SimThreadedArrays.dρdtIThreaded[ichunk][j] += dρdt⁻ + Dⱼ


        Pᵢ      =  Pressure[i]
        Pⱼ      =  Pressure[j]
        Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
        f_ab    = tensile_correction(SimKernel, Pᵢ, ρᵢ, Pⱼ, ρⱼ, q, dx)
        dvdt⁺   = - m₀ * (Pfac + f_ab) *  ∇ᵢWᵢⱼ

        visc_term, _ = compute_viscosity(SimViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, i, j)

        uₘ = dvdt⁺ + visc_term
        SimThreadedArrays.AccelerationThreaded[ichunk][i] += uₘ
        SimThreadedArrays.AccelerationThreaded[ichunk][j] -= uₘ 

        
        if FlagOutputKernelValues
            Wᵢⱼ  = @fastpow @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)
            SimThreadedArrays.KernelThreaded[ichunk][i]         += Wᵢⱼ
            SimThreadedArrays.KernelThreaded[ichunk][j]         += Wᵢⱼ
            SimThreadedArrays.KernelGradientThreaded[ichunk][i] +=  ∇ᵢWᵢⱼ
            SimThreadedArrays.KernelGradientThreaded[ichunk][j] += -∇ᵢWᵢⱼ
        end

        if SimMetaData.FlagShifting
            Wᵢⱼ  = @fastpow SPHKernels.Wᵢⱼ(SimKernel, q)
    
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


function ResetStep!(SimMetaData, SimThreadedArrays, dρdtI, Acceleration, Kernel, KernelGradient, ∇Cᵢ, ∇◌rᵢ)
    
    ResetArrays!(dρdtI, Acceleration)

    if SimMetaData.FlagOutputKernelValues
        ResetArrays!(Kernel, KernelGradient)
    end

    if SimMetaData.FlagShifting
        ResetArrays!(∇Cᵢ, ∇◌rᵢ)
    end

    foreachfield(f -> map!(v -> fill!(v, zero(eltype(v))), f, f), SimThreadedArrays)

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
    Position       = SimParticles.Position
    Density        = SimParticles.Density
    Velocity       = SimParticles.Velocity
    Acceleration   = SimParticles.Acceleration
    GravityFactor  = SimParticles.GravityFactor
    MotionLimiter  = SimParticles.MotionLimiter

    @inbounds @simd ivdep for i in eachindex(Position)
        Acceleration[i]  +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
        Positionₙ⁺[i]     =  Position[i]   + Velocity[i]   * dt₂  * MotionLimiter[i]
        Velocityₙ⁺[i]     =  Velocity[i]   + Acceleration[i]  *  dt₂ * MotionLimiter[i]
        ρₙ⁺[i]            =  Density[i]    + dρdtI[i]       *  dt₂
    end


    return nothing
end

function FullTimeStep(SimMetaData, SimKernel, SimConstants, SimParticles, ∇Cᵢ, ∇◌rᵢ, dt)
    Position       = SimParticles.Position
    Velocity       = SimParticles.Velocity
    Acceleration   = SimParticles.Acceleration
    GravityFactor  = SimParticles.GravityFactor
    MotionLimiter  = SimParticles.MotionLimiter
  
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
end
