# Analytical checks shared by the example, tests and comparison benchmark.
using SPHExampleGPU
using LinearAlgebra: norm
using CSV

"""
    poiseuille_velocity(y, time = Inf; height = 0.001, lower = -height/2, acceleration = 0.0001, viscosity = 1e-6, terms = 128)

Analytical streamwise velocity between stationary no-slip plates at `y = lower`
and `y = lower + height`, starting from rest under constant acceleration. `viscosity`
is kinematic (m²/s). Finite times use the odd sine-series solution of
`∂u/∂t = acceleration + viscosity * ∂²u/∂y²`; `Inf` gives the steady parabola.
"""
function poiseuille_velocity(y, time = Inf; height = 0.001, lower = -height / 2, acceleration = 0.0001, viscosity = 1e-6, terms::Integer = 128)
    isfinite(height) && height > 0 || throw(ArgumentError("height must be finite and positive"))
    isfinite(viscosity) && viscosity > 0 || throw(ArgumentError("viscosity must be finite and positive"))
    isfinite(acceleration) || throw(ArgumentError("acceleration must be finite"))
    distance = y - lower
    isfinite(distance) && 0 <= distance <= height || throw(ArgumentError("y must lie between the plates"))
    !isnan(time) && time >= 0 || throw(ArgumentError("time must be nonnegative"))
    terms > 0 || throw(ArgumentError("terms must be positive"))
    (iszero(time) || iszero(distance) || distance == height) && return 0.0
    steady = acceleration * distance * (height - distance) / (2viscosity)
    isinf(time) && return steady
    transient = sum(1:terms) do k
        n = 2k - 1
        sinpi(n * distance / height) * exp(-viscosity * (n * π / height)^2 * time) / n^3
    end
    return steady - 4acceleration * height^2 / (viscosity * π^3) * transient
end

"""
    poiseuille_profile(result; path = nothing)

Compare every fluid particle with the analytical startup profile at the final
simulation time. Optionally write the particle coordinates, numerical/reference
velocities and density to `path`. Returns relative L2 and maximum velocity errors.
"""
function poiseuille_profile(result; path = nothing)
    result.final_state.complete || throw(ArgumentError("the simulation did not complete"))
    result.diagnostics.frozen_particles == 0 || throw(ArgumentError("the channel contains frozen particles"))
    p = result.final_state.particles
    fluid = findall(==(Fluid), p.Type)
    isempty(fluid) && throw(ArgumentError("the channel contains no fluid"))
    time = result.final_state.time
    rows = [(; id = p.ID[i], x = p.Position[i][1], y = p.Position[i][2], u = p.Velocity[i][1],
        v = p.Velocity[i][2], reference_u = poiseuille_velocity(p.Position[i][2], time),
        density = p.Density[i]) for i in fluid]
    path === nothing || CSV.write(path, rows)
    reference = getproperty.(rows, :reference_u)
    difference = getproperty.(rows, :u) - reference
    return (; time, fluid_particles = length(fluid), relative_l2 = norm(difference) / norm(reference),
        max_error = maximum(abs, difference), max_transverse = maximum(abs, getproperty.(rows, :v)),
        density_min = minimum(p.Density[fluid]), density_max = maximum(p.Density[fluid]))
end
