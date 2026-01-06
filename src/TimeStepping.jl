module TimeStepping

export Δt, FinalizeTimeStep, UpdateTimeStepBuffers!

using LinearAlgebra
using Parameters
using Base.Threads
using Bumper

@inline function UpdateTimeStepBuffers!(::Nothing, ::Nothing, index, position,
                                           velocity, acceleration, sim_kernel)
    return nothing
end

@inline function UpdateTimeStepBuffers!(max_speed, min_dt_force, index, position,
                                           velocity, acceleration, sim_kernel)
    h = sim_kernel.h
    curr_speed = norm(velocity)
    a_mag = norm(acceleration)
    curr_dt_force = a_mag > 0 ? sqrt(h / a_mag) : typemax(eltype(min_dt_force))
    @inbounds begin
        max_speed[index] = curr_speed
        min_dt_force[index] = curr_dt_force
    end
    return nothing
end

"""
    FinalizeTimeStep(max_speed, min_dt_force, SimulationConstants, SPHKernel)

Compute the CFL-limited time step from the per-particle buffers.
"""
function FinalizeTimeStep(max_speed, min_dt_force, SimulationConstants, SPHKernel)
    @unpack c₀, CFL, ν₀, dt_min = SimulationConstants
    @unpack h = SPHKernel

    global_speed = maximum(max_speed)
    global_dt_force = minimum(min_dt_force)

    dt_cfl = h / (c₀ + global_speed)
    dt_visc = ν₀ > 0 ? h^2 / ν₀ : typemax(eltype(min_dt_force))
    dt = CFL * min(global_dt_force, dt_cfl, dt_visc)
    return max(dt_min, dt)
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
    @unpack c₀, CFL, ν₀, dt_min = SimulationConstants
    @unpack h = SPHKernel

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

                t_speed = 0.0
                t_dt   = Inf

                if idx_start <= idx_end
                    @inbounds for j in idx_start:idx_end
                        v = Velocity[j]
                        a = Acceleration[j]

                        curr_speed = norm(v)
                        t_speed = max(t_speed, curr_speed)

                        a_mag = norm(a)
                        if a_mag > 0
                            t_dt = min(t_dt, sqrt(h / a_mag))
                        end
                    end
                end

                v_buffer[i] = t_speed
                d_buffer[i] = t_dt
            end
        end

        dt_cfl = h / (c₀ + maximum(v_buffer))
        dt_visc = ν₀ > 0 ? h^2 / ν₀ : typemax(eltype(d_buffer))
        dt = CFL * min(minimum(d_buffer), dt_cfl, dt_visc)
        max(dt_min, dt)
    end
end

end
