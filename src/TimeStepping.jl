module TimeStepping

export Δt, finalize_time_step, update_time_step_buffers!

using LinearAlgebra
using Parameters
using Base.Threads
using Bumper

@inline function update_time_step_buffers!(::Nothing, ::Nothing, index, position,
                                           velocity, acceleration, sim_kernel)
    return nothing
end

@inline function update_time_step_buffers!(max_visc, min_dt_force, index, position,
                                           velocity, acceleration, sim_kernel)
    h = sim_kernel.h
    η² = sim_kernel.η²
    r_sq = dot(position, position)
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

    t_visc = 0.0
    t_dt   = Inf
    @inbounds for j in eachindex(Position)
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
    return CFL * min(t_dt, h / (c₀ + t_visc))
end

end
