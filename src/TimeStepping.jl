module TimeStepping

export Δt, FinalizeTimeStep, UpdateTimeStepBuffers!, next_output_time, ProgressMotion, HalfTimeStep, FullTimeStep

using LinearAlgebra
using Parameters
using Base.Threads
using Bumper
using ..SimulationEquations
using ..SimulationGeometry
using ..SimulationMetaDataConfiguration

@inline function UpdateTimeStepBuffers!(::Nothing, ::Nothing, ::Nothing, index, visc_sum,
                                           position, velocity, acceleration, sim_kernel)
    return nothing
end

@inline function UpdateTimeStepBuffers!(max_visc, max_speed, min_dt_force, index, visc_sum,
                                           position, velocity, acceleration, sim_kernel)
    h             = sim_kernel.h
    a_mag         = norm(acceleration)
    curr_dt_force = a_mag > 0 ? sqrt(h / a_mag) : Inf
    speed_mag     = norm(velocity)
    @inbounds begin
        max_visc[index] = visc_sum
        max_speed[index] = speed_mag
        min_dt_force[index] = curr_dt_force
    end
    return nothing
end

"""
    FinalizeTimeStep(max_visc, max_speed, min_dt_force, SimulationConstants, SPHKernel)

Compute the CFL-limited time step from the per-particle buffers.
"""
function FinalizeTimeStep(max_visc, max_speed, min_dt_force, SimulationConstants, SPHKernel)
    @unpack c₀, CFL = SimulationConstants
    @unpack h = SPHKernel

    global_visc     = maximum(max_visc)
    global_speed    = maximum(max_speed)
    global_dt_force = minimum(min_dt_force)

    dt2 = h / (max(c₀, global_speed) + h * global_visc)
    return CFL * min(global_dt_force, dt2)
end

"""
    Δt(Position, Velocity, Acceleration, Pressure, Density, SimulationConstants, SPHKernel)

Calculates the adaptive time step for the simulation based on Courant-Friedrichs-Lewy (CFL),
viscous, and force-based criteria.

# Arguments
- `Position`: Vector of position vectors for each particle.
- `Velocity`: Vector of velocity vectors for each particle.
- `Acceleration`: Vector of acceleration vectors for each particle.
- `Pressure`: Vector of pressures for each particle.
- `Density`: Vector of densities for each particle.
- `SimulationConstants`: Struct containing simulation parameters like `c₀` (speed of sound) and `CFL` number.
- `SPHKernel`: Struct containing kernel parameters like `h` (smoothing length) and `η²`.

# Returns
- The calculated time step `dt`.
"""
function Δt(Position, Velocity, Acceleration, Pressure, Density, SimulationConstants, SPHKernel)
    @unpack c₀, CFL = SimulationConstants
    @unpack h   = SPHKernel

    N = length(Position)
    n_chunks = Threads.nthreads()
    chunk_size = cld(N, n_chunks)

    @no_escape begin
        v_buffer = @alloc(Float64, n_chunks)
        d_buffer = @alloc(Float64, n_chunks)
        c_buffer = @alloc(Float64, n_chunks)

        @sync for i in 1:n_chunks
            Threads.@spawn begin
                idx_start = (i - 1) * chunk_size + 1
                idx_end   = min(i * chunk_size, N)

                t_vel = 0.0
                t_dt   = Inf
                t_c    = 0.0

                if idx_start <= idx_end
                    @inbounds for j in idx_start:idx_end
                        v = Velocity[j]
                        a = Acceleration[j]

                        t_vel = max(t_vel, norm(v))
                        t_c = max(t_c, CleaningWaveSpeed(Pressure[j], Density[j], SimulationConstants))

                        a_mag = norm(a)
                        if a_mag > 0
                            t_dt = min(t_dt, sqrt(h / a_mag))
                        end
                    end
                end

                v_buffer[i] = t_vel
                d_buffer[i] = t_dt
                c_buffer[i] = t_c
            end
        end

        max_wave = max(c₀, maximum(v_buffer), maximum(c_buffer))
        CFL * min(minimum(d_buffer), h / max_wave)
    end
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
                          Velocityₙ⁺, ρₙ⁺, Ψₙ⁺, dρdtI, dΨdtI, dt₂) where {Dimensions, FloatType, SMode, KMode, BMode, LMode}
    @unpack Position, Density, Velocity, Acceleration, GravityFactor, MotionLimiter, Psi = SimParticles

    @inbounds @simd ivdep for i in eachindex(Position)
        Acceleration[i]  +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
        Positionₙ⁺[i]     =  Position[i]   + Velocity[i]   * dt₂  * MotionLimiter[i]
        Velocityₙ⁺[i]     =  Velocity[i]   + Acceleration[i]  *  dt₂ * MotionLimiter[i]
        ρₙ⁺[i]            =  Density[i]    + dρdtI[i]       *  dt₂
        Ψₙ⁺[i]            =  Psi[i]        + dΨdtI[i]       *  dt₂
    end

    return nothing
end

function FullTimeStep(::SimulationMetaData{D,T,NoShifting,K,B,L}, SimKernel,
                          SimConstants, SimParticles, ∇Cᵢ, ∇◌rᵢ, dΨdtI, dt) where {D,T,
                                                                             K<:KernelOutputMode,
                                                                             B<:MDBCMode,
                                                                             L<:LogMode}
    @unpack Position, Velocity, Acceleration, GravityFactor, MotionLimiter, Psi = SimParticles
    @inbounds @simd ivdep for i in eachindex(Position)
        Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
        Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
        Position[i]       +=  (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt) * MotionLimiter[i]
        Psi[i]            +=  dΨdtI[i] * dt * MotionLimiter[i]
    end
    return nothing
end

function FullTimeStep(::SimulationMetaData{D,T,S,K,B,L}, SimKernel, SimConstants,
                          SimParticles, ∇Cᵢ, ∇◌rᵢ, dΨdtI, dt) where {D,T,S<:ShiftingMode,
                                                             K<:KernelOutputMode,
                                                             B<:MDBCMode,
                                                             L<:LogMode}
    @unpack Position, Velocity, Acceleration, GravityFactor, MotionLimiter, Psi = SimParticles
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
        Psi[i]                += dΨdtI[i] * dt * MotionLimiter[i]
    end
    return nothing
end

end
