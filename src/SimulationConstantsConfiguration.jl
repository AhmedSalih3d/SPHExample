module SimulationConstantsConfiguration

using ..SPHKernels
using Base: @kwdef

export SimulationConstants

"""
    struct SimulationConstants{T<:Real, TI<:Integer}

SimulationConstants is a parameterized struct representing the constants and parameters used in a simulation. These constants are essential for configuring a Smoothed Particle Hydrodynamics (SPH) simulation.

# Fields
- `ρ₀::T`: Reference density. Default is 1000.
- `dx::T`: Initial particle distance (grid spacing). Default is 0.02.
- `h::T`: Smoothing length. Default is computed as 1.2 times the square root of 2 times `dx`.
- `m₀::T`: Initial mass of particles. Default is computed as `ρ₀ * dx^2`.
- `αD::T`: Normalization constant for the SPH kernel. Default is computed as `7 / (4 * π * H^2)`.
- `α::T`: Artificial viscosity parameter. Default is 0.01.
- `g::T`: Gravitational constant (positive). Default is 9.81.
- `c₀::T`: Speed of sound (must be at least 10 times the highest velocity in the simulation). Default is computed based on `g`.
- `γ::TI`: Adiabatic index (positive integer). Default is 7.
- `dt::T`: Initial time step. Default is 1e-5.
- `δᵩ::T`: Coefficient for density diffusion. Default is 0.1.
- `CFL::T`: CFL (Courant-Friedrichs-Lewy) number (positive). Default is 0.2.
- `η²::T`: Eta squared (positive). Default is computed as `(0.01 * H)^2`.

# Example
```julia
using SimulationConstantsConfiguration

# Create a SimulationConstants instance with custom parameters
constants = SimulationConstants(ρ₀=1017, dx=0.03, α=0.02)
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
end
