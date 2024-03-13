module SimulationConstantsConfiguration

using Parameters

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
@with_kw struct SimulationConstants{T<:AbstractFloat}
    ρ₀::T  = 1000                 ; @assert ρ₀   > 0 "Density (ρ₀) must be positive"
    dx::T  = 0.02                 ; @assert dx   > 0 "Grid spacing (dx) must be positive"
    h::T   = 1.2 * sqrt(2) * dx   ; @assert h    > 0 "Smoothing length (h) must be positive"
    H::T   = 2h                   ; @assert H    > 0 "Kernel support domain (CutOff) (H) must be positive"
    H²::T  = H^2                  ; @assert H²   > 0 "CutOffSquared  (H^2) must be positive"
    h⁻¹::T = 1/h                  ; @assert h⁻¹  > 0 "Inverse smoothing length (h⁻¹) must be positive"
    m₀::T  = ρ₀ * dx^2            ; @assert m₀   > 0 "Particle mass (m₀) must be positive"
    αD::T  = 7 / (4 * π * h^2)    ; @assert αD   > 0 "Alpha parameter (αD) must be positive"
    α::T   = 0.01                 ; @assert α    > 0 "Artificial viscosity (α) must be positive"
    g::T   = 9.81                 ; @assert g   >= 0 "Gravitational constant (g) must be positive"
    c₀::T  = sqrt(g * 2) * 20     ; @assert c₀   > 0 "Speed of sound (c₀) must be positive"
    γ::T  = 7                     ; @assert γ    > 0 "Adiabatic index (γ) must be positive"
    γ⁻¹::T  = 1/γ                 ; @assert γ⁻¹  > 0 "Inverse adiabatic index (γ⁻¹) must be positive"
    dt::T  = 1e-5                 ; @assert dt   > 0 "Time step (dt) must be positive"
    δᵩ::T  = 0.1                  ; @assert δᵩ   > 0 "Density variation (δᵩ) must be positive"
    CFL::T = 0.2                  ; @assert CFL  > 0 "CFL condition (CFL) must be positive"
    η²::T  = (0.01 * h)^2         ; @assert η²   > 0 "Eta squared (η²) must be positive"
    Cb::T  = (c₀^2 * ρ₀)/γ        ; @assert Cb  >= 0 "Cb (pressure coefficient) must be positive"
    Cb⁻¹::T  = inv(Cb)            ; @assert Cb⁻¹>= 0 "Inverse Cb (inverse pressure coefficient) must be positive"
    ν₀::T    = 1e-6               ; @assert ν₀  >= 0 "Kinematic viscosity must be positive"
    BlinConstant::T                              = 0.0066
    SmagorinskyConstant::T                       = 0.12
end

end
