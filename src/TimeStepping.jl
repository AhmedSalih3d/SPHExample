module TimeStepping

export Δt

using LinearAlgebra
using Parameters
using Base.Threads
using Bumper

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

        fill!(v_buffer, 0.0)
        fill!(d_buffer, Inf)

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

                        r_sq = sqrt(dot(r, r))
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

end
