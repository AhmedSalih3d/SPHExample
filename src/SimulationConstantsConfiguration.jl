module SimulationConstantsConfiguration

using Parameters
using ..SPHKernels

export SimulationConstants

"""
    SimulationConstants

Holds global simulation constants shared across models.
Viscosity parameters are configured in `SPHViscosity` models, not here.
"""
@with_kw struct SimulationConstants{T<:AbstractFloat}
    ρ₀::T   = 1000             ; @assert ρ₀   > 0 "Density (ρ₀) must be positive"
    dx::T   = 0.02             ; @assert dx   > 0 "Grid spacing (dx) must be positive"
    m₀::T   = ρ₀ * dx^2        ; @assert m₀   > 0 "Particle mass (m₀) must be positive"
    g::T    = 9.81             ; @assert g   >= 0 "Gravitational constant (g) must be non-negative"
    c₀::T   = sqrt(g * 2) * 20 ; @assert c₀   > 0 "Speed of sound (c₀) must be positive"
    γ::T    = 7                ; @assert γ    > 0 "Adiabatic index (γ) must be positive"
    γ⁻¹::T  = 1 / γ            ; @assert γ⁻¹  > 0 "Inverse adiabatic index (γ⁻¹) must be positive"
    δᵩ::T   = 0.1              ; @assert δᵩ   > 0 "Density variation (δᵩ) must be positive"
    CFL::T  = 0.2              ; @assert CFL  > 0 "CFL condition (CFL) must be positive"
    A::T    = 0.01             ; @assert A    > 0 "Shifting factor (A) must be positive"
    Cb::T   = (c₀^2 * ρ₀) / γ  ; @assert Cb  >= 0 "Cb (pressure coefficient) must be non-negative"
    Cb⁻¹::T = inv(Cb)          ; @assert Cb⁻¹ >= 0 "Inverse Cb (Cb⁻¹) must be non-negative"
end

end
