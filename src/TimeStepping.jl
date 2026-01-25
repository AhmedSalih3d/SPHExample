module TimeStepping

export Δt, next_output_time, ProgressMotion, HalfTimeStep, FullTimeStep, UpdateTimeStep

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

"""
    UpdateTimeStep(AccelerationMax, SimConstants, SimKernel)

Computes and returns the updated time step based on maximum acceleration across all particles.

# Arguments
- `AccelerationMax`: Array of acceleration magnitudes for each particle.
- `SimConstants`: Struct containing simulation parameters.
- `SimKernel`: Struct containing kernel parameters.

# Returns
- The calculated adaptive time step `dt`.
"""
function UpdateTimeStep(AccelerationMax, SimConstants, SimKernel)
    max_acceleration = maximum(AccelerationMax)
    return Δt(max_acceleration, SimConstants, SimKernel)
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
        if ParticleType[i] == Moving || ParticleType[i] == FixedMoving
            motion = MotionsDefinition[ParticleMarker[i]]

            if motion !== nothing
                ShouldMove = (motion.StartTime <= SimMetaData.TotalTime) &&
                             (SimMetaData.TotalTime <= (motion.StartTime + motion.Duration))

                # Retrieve motion parameters
                MotionVel = motion.Velocity
                MotionDir = motion.Direction
                MotionFactor = ShouldMove ? one(MotionVel) : zero(MotionVel)
                PositionFactor = MotionPositionFactorValue(typeof(MotionVel), ParticleType[i])

                # Update Velocity and Position
                Velocity[i] = MotionFactor * MotionVel * MotionDir
                if motion.MovePosition && ShouldMove && PositionFactor != zero(PositionFactor)
                    Position[i] += Velocity[i] * dt₂ * PositionFactor
                end
            end
        end
    end

    return nothing
end

function HalfTimeStep(::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
                          SimConstants, SimParticles, Positionₙ⁺,
                          Velocityₙ⁺, ρₙ⁺, dρdtI, dt₂) where {Dimensions, FloatType, SMode, KMode, BMode, LMode}
    @unpack Position, Density, Velocity, Acceleration = SimParticles
    ParticleType = SimParticles.Type
    AccelerationScalarType = eltype(eltype(Acceleration))

    @inbounds @simd ivdep for i in eachindex(Position)
        MotionLimiterFactor = MotionLimiterValue(AccelerationScalarType, ParticleType[i])
        GravityFactor = GravityFactorValue(AccelerationScalarType, ParticleType[i])
        Acceleration[i]  +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor)
        Positionₙ⁺[i]     =  Position[i]   + Velocity[i]   * dt₂  * MotionLimiterFactor
        Velocityₙ⁺[i]     =  Velocity[i]   + Acceleration[i]  *  dt₂ * MotionLimiterFactor
        ρₙ⁺[i]            =  Density[i]    + dρdtI[i]       *  dt₂
    end

    return nothing
end

function FullTimeStep(::SimulationMetaData{D,T,NoShifting,K,B,L}, SimKernel,
                          SimConstants, SimParticles, Velocityₙ⁺, ∇Cᵢ, ∇◌rᵢ, dt) where {D,T,
                                                                             K<:KernelOutputMode,
                                                                             B<:MDBCMode,
                                                                             L<:LogMode}
    @unpack Position, Velocity, Acceleration = SimParticles
    ParticleType = SimParticles.Type
    AccelerationScalarType = eltype(eltype(Acceleration))
    @inbounds @simd ivdep for i in eachindex(Position)
        MotionLimiterFactor = MotionLimiterValue(AccelerationScalarType, ParticleType[i])
        GravityFactor = GravityFactorValue(AccelerationScalarType, ParticleType[i])
        Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor)
        Velocity[i]       +=  Acceleration[i] * dt * MotionLimiterFactor
        Position[i]       +=  (Velocityₙ⁺[i] * dt) * MotionLimiterFactor
    end
    return nothing
end

function FullTimeStep(::SimulationMetaData{D,T,S,K,B,L}, SimKernel, SimConstants,
                          SimParticles, Velocityₙ⁺, ∇Cᵢ, ∇◌rᵢ, dt) where {D,T,S<:ShiftingMode,
                                                             K<:KernelOutputMode,
                                                             B<:MDBCMode,
                                                             L<:LogMode}
    @unpack Position, Velocity, Acceleration = SimParticles
    ParticleType = SimParticles.Type
    AccelerationScalarType = eltype(eltype(Acceleration))
    A     = 0.005# Value between 1 to 6 advised
    A_FST = 0; # zero for internal flows
    A_FSM = length(first(Position)); #2d, 3d val different
    @inbounds @simd ivdep for i in eachindex(Position)
        MotionLimiterFactor = MotionLimiterValue(AccelerationScalarType, ParticleType[i])
        GravityFactor = GravityFactorValue(AccelerationScalarType, ParticleType[i])
        Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor)
        Velocity[i]       +=  Acceleration[i] * dt * MotionLimiterFactor

        δxᵢ = zero(Acceleration[i])
        A_FSC                  = (∇◌rᵢ[i] - A_FST)/(A_FSM - A_FST)
        if (∇◌rᵢ[i] - A_FST) < 0
            δxᵢ = -A_FSC * A * SimKernel.h * norm(Velocity[i]) * dt * ∇Cᵢ[i]
        elseif (∇◌rᵢ[i] - A_FST) >= 0
            δxᵢ = -A * SimKernel.h * norm(Velocity[i]) * dt * ∇Cᵢ[i]
        end

        Position[i]           += (Velocityₙ⁺[i] * dt + δxᵢ) * MotionLimiterFactor
    end
    return nothing
end

end
