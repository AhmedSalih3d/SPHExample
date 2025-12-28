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
    @unpack tasks, chunk_size = workspace

    for i in eachindex(tasks)
        # Calculate start/end indices for this chunk
        idx_start = (i - 1) * chunk_size + 1
        idx_end = min(i * chunk_size, length(Position))

        # Spawn the task on any available thread
        tasks[i] = Threads.@spawn begin
            # Local accumulators for this specific task
            local_visc = 0.0
            local_dt_force = Inf

            # If this chunk has valid indices, run the loop
            if idx_start <= idx_end
                # @inbounds is safe here because we calculated indices carefully
                @inbounds for j in idx_start:idx_end
                    r = Position[j]
                    v = Velocity[j]
                    a = Acceleration[j]

                    # --- Viscous Logic ---
                    r_sq = dot(r, r)
                    # abs() is sufficient, removed redundant checks
                    curr_visc = abs(h * dot(v, r) / (r_sq + η²))
                    
                    if curr_visc > local_visc
                        local_visc = curr_visc
                    end

                    # --- Force Logic ---
                    a_mag = norm(a)
                    if a_mag > 0
                        curr_dt_force = sqrt(h / a_mag)
                        if curr_dt_force < local_dt_force
                            local_dt_force = curr_dt_force
                        end
                    end
                end
            end
            # The task returns its local results as a tuple
            (local_visc, local_dt_force)
        end
    end

    # 3. Reduce Results
    # Initialize global accumulators
    global_visc = 0.0
    global_dt_force = Inf

    # Wait for all tasks to finish and combine their results
    for t in tasks
        (l_visc, l_dt) = fetch(t)
        if l_visc > global_visc
            global_visc = l_visc
        end
        if l_dt < global_dt_force
            global_dt_force = l_dt
        end
    end

    # 4. Final Calculation
    dt2 = h / (c₀ + global_visc)
    dt = CFL * min(global_dt_force, dt2)

    return dt
end

end
