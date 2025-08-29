module TimeStepping

export Δt

using LinearAlgebra

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
    (; c₀, CFL) = SimulationConstants
    (; h, η²) = SPHKernel

    # Viscous timestep constraint
    # More idiomatic and potentially faster version of max_visc
    visc = maximum(zip(Velocity, Position)) do (v, r)
        abs(h * dot(v, r) / (dot(r, r) + η²))
    end

    # Force-based timestep constraint
    # Use a generator expression with `minimum` for conciseness and efficiency
    # The norm of the acceleration is calculated for each particle.
    dt1 = minimum(sqrt(h / norm(acc)) for acc in Acceleration; init=Inf)

    # Courant-like speed of sound condition
    dt2 = h / (c₀ + visc)

    # Final timestep is the minimum of the two, scaled by the CFL number
    dt = CFL * min(dt1, dt2)

    return dt
end

end