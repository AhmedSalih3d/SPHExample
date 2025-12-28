module TimeStepping

export Δt

using LinearAlgebra, Parameters, Polyester, Bumper

function Δt(Position, Velocity, Acceleration, SimulationConstants, SPHKernel)
    @unpack c₀, CFL = SimulationConstants
    @unpack h, η²   = SPHKernel

    N = length(Position)
    n_threads = Threads.nthreads()

    # 1. We allocate the reduction buffers normally (on the heap or as StaticArrays) 
    # because they need to exist across the threaded boundary.
    # Since it's only 2 Float64s per thread, this is negligible.
    visc_vals = zeros(Float64, n_threads)
    dt_vals   = fill(Inf, n_threads)

    # 2. Parallel loop
    @batch for i in 1:n_threads
        # Each thread gets its own Bumper scope
        @no_escape begin
            idx_start = (i - 1) * cld(N, n_threads) + 1
            idx_end   = min(i * cld(N, n_threads), N)

            t_visc = 0.0
            t_dt   = Inf

            if idx_start <= idx_end
                @inbounds for j in idx_start:idx_end
                    # If you had larger Bumper arrays (like A_gamma), 
                    # you would @alloc them here!
                    
                    r = Position[j]
                    v = Velocity[j]
                    a = Acceleration[j]

                    r_sq = dot(r, r)
                    curr_visc = abs(h * dot(v, r) / (r_sq + η²))
                    t_visc = max(t_visc, curr_visc)

                    a_mag = norm(a)
                    if a_mag > 0
                        t_dt = min(t_dt, sqrt(h / a_mag))
                    end
                end
            end
            
            # Save thread-local results back to the shared reduction arrays
            visc_vals[i] = t_visc
            dt_vals[i]   = t_dt
        end
    end

    # 3. Final Global Reduction (Implicit Return)
    CFL * min(minimum(dt_vals), h / (c₀ + maximum(visc_vals)))
end

end