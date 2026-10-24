module SimulationConstantsConfiguration

using ..SPHKernels
using Base: @kwdef

export SimulationConstants

"""
    SimulationConstants{T}(; kwargs...)
    SimulationConstants{T}(dimensions::Integer; kwargs...)

SimulationConstants is a parameterized struct representing the constants and parameters used in a simulation. These constants are essential for configuring a Smoothed Particle Hydrodynamics (SPH) simulation.

Pass `2` or `3` as the positional `dimensions` argument to default the particle
mass to `ρ₀ * dx^dimensions`. In 2D, `ρ₀ * dx^2` assumes unit out-of-plane
depth (one metre when using SI units). In 3D, the default is `ρ₀ * dx^3`.
An explicit `m₀` overrides either default. The keyword-only constructor retains
the 2D default for compatibility. Both constructors return `SimulationConstants{T}`;
the dimension only selects the default mass.

# Fields
- `ρ₀::T`: Reference density. Default is 1000.
- `dx::T`: Initial particle distance (grid spacing). Default is 0.02.
- `m₀::T`: Initial particle mass. Defaults to `ρ₀ * dx^dimensions` with the dimension-aware constructor, or `ρ₀ * dx^2` otherwise.
- `α::T`: Artificial viscosity parameter. Default is 0.01.
- `g::T`: Gravitational constant (positive). Default is 9.81.
- `c₀::T`: Speed of sound (must be at least 10 times the highest velocity in the simulation). Default is computed based on `g`.
- `γ::T`: Adiabatic index. Default is 7.
- `γ⁻¹::T`: Reciprocal of `γ`.
- `δᵩ::T`: Coefficient for density diffusion. Default is 0.1.
- `CFL::T`: CFL (Courant-Friedrichs-Lewy) number (positive). Default is 0.2.
- `Cb::T`: Equation-of-state coefficient. Default is `c₀^2 * ρ₀ / γ`.
- `Cb⁻¹::T`: Reciprocal of `Cb`.
- `ν₀::T`: Kinematic viscosity. Default is 1e-6.
- `BlinConstant::T`: Blin model constant. Default is 0.0066.
- `SmagorinskyConstant::T`: Smagorinsky model constant. Default is 0.12.

# Example
```julia
using SPHExampleGPU

# A 3D simulation, with particle mass computed from the reference density and spacing.
constants = SimulationConstants{Float32}(3; ρ₀=1017, dx=0.03, α=0.02)
```
"""
@kwdef struct SimulationConstants{T<:AbstractFloat}
    ρ₀::T  = 1000
    dx::T  = 0.02
    m₀::T  = ρ₀ * dx^2
    α::T   = 0.01
    g::T   = 9.81
    c₀::T  = sqrt(g * 2) * 20
    γ::T   = 7
    γ⁻¹::T = 1 / γ
    δᵩ::T  = 0.1
    CFL::T = 0.2
    Cb::T  = (c₀^2 * ρ₀) / γ
    Cb⁻¹::T = inv(Cb)
    ν₀::T  = 1e-6
    BlinConstant::T            = 0.0066
    SmagorinskyConstant::T     = 0.12
end

function SimulationConstants{T}(dimensions::Integer; ρ₀ = 1000, dx = 0.02, m₀ = nothing, kwargs...) where {T<:AbstractFloat}
    dimensions in (2, 3) || throw(ArgumentError("SimulationConstants supports only 2D or 3D; got dimensions = $dimensions"))
    mass = m₀ === nothing ? ρ₀ * dx^dimensions : m₀
    return SimulationConstants{T}(; ρ₀, dx, m₀ = mass, kwargs...)
end
end
