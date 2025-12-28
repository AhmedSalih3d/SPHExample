module TimeStepping

export ΔtWorkspace, ΔtAccumulator, Δt, finalize_time_step, reset_time_step_accumulator!,
       update_time_step_accumulator!

using LinearAlgebra
using Parameters
using Base.Threads

struct ΔtWorkspace
    tasks::Vector{Task}
    chunk_size::Int
end

struct ΔtAccumulator{T<:AbstractFloat}
    max_visc::Vector{T}
    min_dt_force::Vector{T}
end

"""
    ΔtAccumulator(nthreads, ::Type{T})

Create a thread-local accumulator for on-the-fly time step estimation.
"""
function ΔtAccumulator(nthreads::Int, ::Type{T}) where {T<:AbstractFloat}
    return ΔtAccumulator(fill(zero(T), nthreads), fill(typemax(T), nthreads))
end

"""
    ΔtAccumulator(::Type{T})

Create a thread-local accumulator sized to `Threads.maxthreadid()`.
"""
function ΔtAccumulator(::Type{T}) where {T<:AbstractFloat}
    return ΔtAccumulator(Threads.maxthreadid(), T)
end

"""
    reset_time_step_accumulator!(accumulator)

Reset the per-thread time step accumulators.
"""
function reset_time_step_accumulator!(accumulator::ΔtAccumulator{T}) where {T<:AbstractFloat}
    fill!(accumulator.max_visc, zero(T))
    fill!(accumulator.min_dt_force, typemax(T))
    return nothing
end

@inline function update_time_step_accumulator!(::Nothing, position, velocity,
                                               acceleration, sim_kernel)
    return nothing
end

@inline function update_time_step_accumulator!(accumulator::ΔtAccumulator, position,
                                               velocity, acceleration, sim_kernel)
    tid = threadid()
    if tid > length(accumulator.max_visc)
        return nothing
    end
    h = sim_kernel.h
    η² = sim_kernel.η²
    r_sq = dot(position, position)
    curr_visc = abs(h * dot(velocity, position) / (r_sq + η²))
    if curr_visc > accumulator.max_visc[tid]
        accumulator.max_visc[tid] = curr_visc
    end

    a_mag = norm(acceleration)
    if a_mag > 0
        curr_dt_force = sqrt(h / a_mag)
        if curr_dt_force < accumulator.min_dt_force[tid]
            accumulator.min_dt_force[tid] = curr_dt_force
        end
    end
    return nothing
end

"""
    finalize_time_step(accumulator, SimulationConstants, SPHKernel)

Compute the CFL-limited time step from the on-the-fly accumulators.
"""
function finalize_time_step(accumulator::ΔtAccumulator, SimulationConstants, SPHKernel)
    @unpack c₀, CFL = SimulationConstants
    @unpack h = SPHKernel

    global_visc = maximum(accumulator.max_visc)
    global_dt_force = minimum(accumulator.min_dt_force)

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
function Δt(workspace, Position, Velocity, Acceleration, SimulationConstants, SPHKernel)
    @unpack c₀, CFL = SimulationConstants
    @unpack h, η²   = SPHKernel

    N = length(Position)
    # Determine how many tasks to spawn. Usually nthreads() is best.
    n_chunks = Threads.nthreads()
    chunk_size = cld(N, n_chunks)

    @no_escape begin
        # Preallocate thread-local reduction buffers on the Bumper stack
        v_buffer = @alloc(Float64, n_chunks)
        d_buffer = @alloc(Float64, n_chunks)
        
        fill!(v_buffer, 0.0)
        fill!(d_buffer, Inf)

        # Use @sync to wait for all spawned tasks to complete
        @sync for i in 1:n_chunks
            Threads.@spawn begin
                # Each task knows exactly which index (i) it owns in the buffer
                idx_start = (i - 1) * chunk_size + 1
                idx_end   = min(i * chunk_size, N)

                # Local registers for the chunk to avoid memory traffic
                t_visc = 0.0
                t_dt   = Inf

                if idx_start <= idx_end
                    @inbounds for j in idx_start:idx_end
                        r = Position[j]
                        v = Velocity[j]
                        a = Acceleration[j]

                        # --- Viscous Logic ---
                        r_sq = sqrt(dot(r, r))
                        curr_visc = abs(h * dot(v, r) / (r_sq + η²))
                        t_visc = max(t_visc, curr_visc)

                        # --- Force Logic ---
                        a_mag = norm(a)
                        if a_mag > 0
                            t_dt = min(t_dt, sqrt(h / a_mag))
                        end
                    end
                end

                # Write results to the specific slot assigned to this task
                v_buffer[i] = t_visc
                d_buffer[i] = t_dt
            end
        end

        # Final Reduction: The block returns this value implicitly
        CFL * min(minimum(d_buffer), h / (c₀ + maximum(v_buffer)))
    end
end

end
