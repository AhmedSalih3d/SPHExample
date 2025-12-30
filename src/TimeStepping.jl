module TimeStepping

export Δt, finalize_time_step, update_time_step_buffers!, viscous_term

using LinearAlgebra
using Parameters
using Base.Threads
using Bumper

@inline function update_time_step_buffers!(::Nothing, ::Nothing, index, viscous_max,
                                           acceleration, sim_kernel)
    return nothing
end

@inline function update_time_step_buffers!(max_visc, min_dt_force, index, viscous_max,
                                           acceleration, sim_kernel)
    h = sim_kernel.h
    a_mag = norm(acceleration)
    curr_dt_force = a_mag > 0 ? sqrt(h / a_mag) : typemax(eltype(min_dt_force))
    @inbounds begin
        max_visc[index] = viscous_max
        min_dt_force[index] = curr_dt_force
    end
    return nothing
end

@inline function viscous_term(position_a, velocity_a, position_b, velocity_b, h, η²)
    r_ab = position_a - position_b
    v_ab = velocity_a - velocity_b
    r_sq = dot(r_ab, r_ab)
    return abs(h * dot(v_ab, r_ab) / (r_sq + η²))
end

"""
    finalize_time_step(max_visc, min_dt_force, SimulationConstants, SPHKernel)

Compute the CFL-limited time step from the per-particle buffers.
"""
function finalize_time_step(max_visc, min_dt_force, SimulationConstants, SPHKernel)
    @unpack c₀, CFL = SimulationConstants
    @unpack h = SPHKernel

    global_visc = maximum(max_visc)
    global_dt_force = minimum(min_dt_force)

    dt2 = h / (c₀ + global_visc)
    return CFL * min(global_dt_force, dt2)
end

"""
    Δt(Position, Velocity, Acceleration, SimulationConstants, SPHKernel,
       ParticleRanges, CellDict, NeighborCellLists, Cells)

Calculates the adaptive time step for the simulation based on Courant-Friedrichs-Lewy (CFL),
viscous, and force-based criteria.

# Arguments
- `Position`: Vector of position vectors for each particle.
- `Velocity`: Vector of velocity vectors for each particle.
- `Acceleration`: Vector of acceleration vectors for each particle.
- `SimulationConstants`: Struct containing simulation parameters like `c₀` (speed of sound) and `CFL` number.
- `SPHKernel`: Struct containing kernel parameters like `h` (smoothing length) and `η²`.
- `ParticleRanges`: Cell list particle ranges for neighbor access.
- `CellDict`: Mapping from cell coordinates to cell list indices.
- `NeighborCellLists`: Precomputed neighbor cell indices.
- `Cells`: Cell assignment for each particle.

# Returns
- The calculated time step `dt`.
"""
function Δt(Position, Velocity, Acceleration, SimulationConstants, SPHKernel,
            ParticleRanges, CellDict, NeighborCellLists, Cells)
    @unpack c₀, CFL = SimulationConstants
    @unpack h, η²   = SPHKernel

    N = length(Position)
    n_threads = Threads.nthreads()

    @no_escape begin
        max_visc_buffer = @alloc(Float64, n_threads)
        min_dt_buffer = @alloc(Float64, n_threads)
        fill!(max_visc_buffer, 0.0)
        fill!(min_dt_buffer, Inf)

        Threads.@threads for i in 1:N
            thread_idx = Threads.threadid()
            local_visc = max_visc_buffer[thread_idx]
            local_min_dt = min_dt_buffer[thread_idx]

            a_mag = norm(Acceleration[i])
            if a_mag > 0
                local_min_dt = min(local_min_dt, sqrt(h / a_mag))
            end

            cell_index = Cells[i]
            cell_list_index = get(CellDict, cell_index, 1)
            same_cell_start = ParticleRanges[cell_list_index]
            same_cell_end = ParticleRanges[cell_list_index + 1] - 1
            neighbor_cell_indices = NeighborCellLists[cell_list_index]

            @inbounds for j in same_cell_start:(i - 1)
                local_visc = max(
                    local_visc,
                    viscous_term(Position[i], Velocity[i], Position[j], Velocity[j], h, η²),
                )
            end
            @inbounds for j in (i + 1):same_cell_end
                local_visc = max(
                    local_visc,
                    viscous_term(Position[i], Velocity[i], Position[j], Velocity[j], h, η²),
                )
            end
            for neighbor_idx in neighbor_cell_indices
                start_index = ParticleRanges[neighbor_idx]
                end_index = ParticleRanges[neighbor_idx + 1] - 1
                @inbounds for j in start_index:end_index
                    local_visc = max(
                        local_visc,
                        viscous_term(
                            Position[i],
                            Velocity[i],
                            Position[j],
                            Velocity[j],
                            h,
                            η²,
                        ),
                    )
                end
            end

            max_visc_buffer[thread_idx] = local_visc
            min_dt_buffer[thread_idx] = local_min_dt
        end

        global_visc = maximum(max_visc_buffer)
        global_dt_force = minimum(min_dt_buffer)

        CFL * min(global_dt_force, h / (c₀ + global_visc))
    end
end

end
