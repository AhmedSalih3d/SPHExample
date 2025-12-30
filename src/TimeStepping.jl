module TimeStepping

export Δt, FinalizeTimeStep, UpdateTimeStepBuffers!, ViscousTerm

using LinearAlgebra
using Parameters
using Bumper

@inline function UpdateTimeStepBuffers!(::Nothing, ::Nothing, _index, _viscous_max,
                                           _acceleration, _sim_kernel)
    return nothing
end

@inline function UpdateTimeStepBuffers!(max_visc, min_dt_force, index, viscous_max,
                                           acceleration, sim_kernel)
    h = sim_kernel.h
    # Compute force-based dt from acceleration; if invalid, fall back to a large value.
    a_mag = norm(acceleration)
    curr_dt_force = typemax(eltype(min_dt_force))
    if isfinite(a_mag) && a_mag > 0
        curr_dt_force = sqrt(h / a_mag)
    end
    # Keep the viscous contribution finite; invalid values drop to zero.
    safe_viscous = zero(viscous_max)
    if isfinite(viscous_max)
        safe_viscous = viscous_max
    end
    @inbounds begin
        max_visc[index] = safe_viscous
        min_dt_force[index] = curr_dt_force
    end
    return nothing
end

@inline function ViscousTerm(position_a, velocity_a, position_b, velocity_b, h, η²)
    r_ab = position_a - position_b
    v_ab = velocity_a - velocity_b
    # Use norm(r_ab)^2 to match the reference formulation (distance squared).
    r_sq = norm(r_ab)^2
    denom = r_sq + η²
    term = h * dot(v_ab, r_ab) / denom
    # Guard against invalid/degenerate denominators to avoid NaNs.
    viscous = zero(term)
    if isfinite(term) && denom > 0
        viscous = abs(term)
    end
    return viscous
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
    Δt(max_visc, min_dt_force, SimulationConstants, SPHKernel)

Calculates the adaptive time step for the simulation based on Courant-Friedrichs-Lewy (CFL),
viscous, and force-based criteria using precomputed buffers.

# Arguments
- `max_visc`: Per-particle viscous maxima buffer.
- `min_dt_force`: Per-particle force-based time-step buffer.
- `SimulationConstants`: Struct containing simulation parameters like `c₀` (speed of sound) and `CFL` number.
- `SPHKernel`: Struct containing kernel parameters like `h` (smoothing length) and `η²`.

# Returns
- The calculated time step `dt`.
"""
function Δt(max_visc, min_dt_force, SimulationConstants, SPHKernel)
    return FinalizeTimeStep(max_visc, min_dt_force, SimulationConstants, SPHKernel)
end

end
