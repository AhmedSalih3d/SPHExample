module TimeStepping

export Δt

using LinearAlgebra
using Parameters
using Bumper

function Δt(Position, Velocity, Acceleration, SimulationConstants, SPHKernel)
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