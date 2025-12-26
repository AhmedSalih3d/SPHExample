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

    visc_values = fill(0.0, Threads.nthreads())
    dt1_values = fill(Inf, Threads.nthreads())
    Threads.@threads for i in eachindex(Position)
        tid = Threads.threadid()
        v = Velocity[i]
        r = Position[i]
        acc = Acceleration[i]

        visc_values[tid] = max(
            visc_values[tid],
            abs(h * dot(v, r) / (dot(r, r) + η²)),
        )

        acc_norm = norm(acc)
        if acc_norm != 0
            dt1_values[tid] = min(dt1_values[tid], sqrt(h / acc_norm))
        end
    end

    visc = maximum(visc_values)
    dt1 = minimum(dt1_values)

    # Courant-like speed of sound condition
    dt2 = h / (c₀ + visc)

    # Final timestep is the minimum of the two, scaled by the CFL number
    dt = CFL * min(dt1, dt2)

    return dt
end

end
