module TimeStepping

export Δt

using LinearAlgebra
using Parameters

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

    # 1. Determine Chunks
    # We split the work evenly across the available threads
    num_particles = length(Position)
    num_threads = Threads.nthreads()
    
    # Calculate chunk size (ceiling division to ensure we cover all particles)
    chunk_size = cld(num_particles, num_threads)

    # 2. Spawn Tasks
    # We create a vector of Tasks. Each task processes one chunk.
    tasks = Vector{Task}(undef, num_threads)

    for i in 1:num_threads
        # Calculate start/end indices for this chunk
        idx_start = (i - 1) * chunk_size + 1
        idx_end = min(i * chunk_size, num_particles)

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