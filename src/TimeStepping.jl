module TimeStepping

export Δt, FinalizeTimeStep, UpdateTimeStepBuffers!, next_output_time, ProgressMotion

using LinearAlgebra
using Parameters
using Base.Threads
using Bumper

@inline function UpdateTimeStepBuffers!(::Nothing, ::Nothing, index, position,
                                           velocity, acceleration, sim_kernel)
    return nothing
end

@inline function UpdateTimeStepBuffers!(max_visc, min_dt_force, index, position,
                                           velocity, acceleration, sim_kernel)
    h = sim_kernel.h
    η² = sim_kernel.η²
    r_sq = sqrt(dot(position, position))^2
    curr_visc = abs(h * dot(velocity, position) / (r_sq + η²))
    a_mag = norm(acceleration)
    curr_dt_force = a_mag > 0 ? sqrt(h / a_mag) : typemax(eltype(min_dt_force))
    @inbounds begin
        max_visc[index] = curr_visc
        min_dt_force[index] = curr_dt_force
    end
    return nothing
end

"""
    FinalizeTimeStep(max_visc, min_dt_force, SimulationConstants, SPHKernel)

Compute the CFL-limited time step from the per-particle buffers.
"""
function FinalizeTimeStep(max_visc, min_dt_force, SimulationConstants, SPHKernel)
    @unpack c₀, CFL = SimulationConstants
    @unpack h = SPHKernel

    global_visc = maximum(max_visc)
    global_dt_force = minimum(min_dt_force)

    dt2 = h / (c₀ + global_visc)
    return CFL * min(global_dt_force, dt2)
end

"""
    Δt(Position, Velocity, Acceleration, SimulationConstants, SPHKernel)

Calculates the adaptive time step for the simulation based on Courant-Friedrichs-Lewy (CFL),
viscous, and force-based criteria.

# Arguments
- `Position`: Vector of position vectors for each particle.
- `Velocity`: Vector of velocity vectors for each particle.
- `Acceleration`: Vector of acceleration vectors for each particle.
- `SimulationConstants`: Struct containing simulation parameters like `c₀` (speed of sound) and `CFL` number.
- `SPHKernel`: Struct containing kernel parameters like `h` (smoothing length) and `η²`.

# Returns
- The calculated time step `dt`.
"""
function Δt(Position, Velocity, Acceleration, SimulationConstants, SPHKernel)
    @unpack c₀, CFL = SimulationConstants
    @unpack h, η²   = SPHKernel

    N = length(Position)
    n_chunks = Threads.nthreads()
    chunk_size = cld(N, n_chunks)

    @no_escape begin
        v_buffer = @alloc(Float64, n_chunks)
        d_buffer = @alloc(Float64, n_chunks)

        @sync for i in 1:n_chunks
            Threads.@spawn begin
                idx_start = (i - 1) * chunk_size + 1
                idx_end   = min(i * chunk_size, N)

                t_visc = 0.0
                t_dt   = Inf

                if idx_start <= idx_end
                    @inbounds for j in idx_start:idx_end
                        r = Position[j]
                        v = Velocity[j]
                        a = Acceleration[j]

                        r_sq = sqrt(dot(r, r))^2
                        curr_visc = abs(h * dot(v, r) / (r_sq + η²))
                        t_visc = max(t_visc, curr_visc)

                        a_mag = norm(a)
                        if a_mag > 0
                            t_dt = min(t_dt, sqrt(h / a_mag))
                        end
                    end
                end

                v_buffer[i] = t_visc
                d_buffer[i] = t_dt
            end
        end

        CFL * min(minimum(d_buffer), h / (c₀ + maximum(v_buffer)))
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

end
