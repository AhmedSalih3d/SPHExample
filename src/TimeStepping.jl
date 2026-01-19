module TimeStepping

export Δt, next_output_time, ProgressMotion, HalfTimeStep, FullTimeStep

using LinearAlgebra
using Parameters
using Base.Threads
using Bumper
using ..SimulationEquations
using ..SimulationGeometry
using ..SimulationMetaDataConfiguration

"""
    Δt(max_acceleration, SimulationConstants, SPHKernel)

Calculates the adaptive time step for the simulation based on Courant-Friedrichs-Lewy (CFL)
and force-based criteria.

# Arguments
- `max_acceleration`: Maximum acceleration magnitude across particles.
- `SimulationConstants`: Struct containing simulation parameters like `c₀` (speed of sound) and `CFL` number.
- `SPHKernel`: Struct containing kernel parameters like `h` (smoothing length) and `η²`.

# Returns
- The calculated time step `dt`.
"""
function Δt(max_acceleration, SimulationConstants, SPHKernel)
    @unpack c₀, CFL = SimulationConstants
    @unpack h   = SPHKernel

    dt_speed = h / c₀
    dt_force = sqrt(h / max_acceleration)
    return CFL * min(dt_speed, dt_force)
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

function ProgressMotion(_SimParticles, _dt₂, ::Nothing, _SimMetaData)
    return nothing
end

function ProgressMotion(SimParticles, dt₂, MotionsDefinition, SimMetaData)
    @unpack Position, Velocity = SimParticles
    ParticleMarker  = SimParticles.GroupMarker
    ParticleType    = SimParticles.Type
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

function HalfTimeStep(::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                          SimConstants, SimParticles, Positionₙ⁺,
                          Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂) where {Dimensions, FloatType, SMode, KMode, BMode, LMode}
    @unpack Position, Density, Velocity, Acceleration, GravityFactor, MotionLimiter = SimParticles

    @inbounds @simd ivdep for i in eachindex(Position)
        Acceleration[i]  +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
        Positionₙ⁺[i]     =  Position[i]   + Velocity[i]   * dt₂  * MotionLimiter[i]
        Velocityₙ⁺[i]     =  Velocity[i]   + Acceleration[i]  *  dt₂ * MotionLimiter[i]
        ρₙ⁺[i]            =  Density[i]    + dρdtI[i]       *  dt₂
    end

    return nothing
end

function FullTimeStep(::SimulationMetaData{D,T,NoShifting,K,B,L}, SimKernel,
                          SimConstants, SimParticles, ∇Cᵢ, ∇◌rᵢ, dt) where {D,T,
                                                                             K<:KernelOutputMode,
                                                                             B<:MDBCMode,
                                                                             L<:LogMode}
    @unpack Position, Velocity, Acceleration, GravityFactor, MotionLimiter = SimParticles
    @inbounds @simd ivdep for i in eachindex(Position)
        Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
        Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
        Position[i]       +=  (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt) * MotionLimiter[i]
    end
    return nothing
end

function FullTimeStep(::SimulationMetaData{D,T,S,K,B,L}, SimKernel, SimConstants,
                          SimParticles, ∇Cᵢ, ∇◌rᵢ, dt) where {D,T,S<:ShiftingMode,
                                                             K<:KernelOutputMode,
                                                             B<:MDBCMode,
                                                             L<:LogMode}
    @unpack Position, Velocity, Acceleration, GravityFactor, MotionLimiter = SimParticles
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
    return nothing
end

end
