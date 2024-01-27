module SimulationConstantsConfiguration

using Parameters

export SimulationConstants

@with_kw struct SimulationConstants{T<:Real, TI<:Integer}
    ρ₀::T  = 1000                 ; @assert ρ₀  > 0 "Density (ρ₀) must be positive"
    dx::T  = 0.02                 ; @assert dx  > 0 "Grid spacing (dx) must be positive"
    H::T   = 1.2 * sqrt(2) * dx   ; @assert H   > 0 "Smoothing length (H) must be positive"
    m₀::T  = ρ₀ * dx^2            ; @assert m₀  > 0 "Particle mass (m₀) must be positive"
    αD::T  = 7 / (4 * π * H^2)    ; @assert αD  > 0 "Alpha parameter (αD) must be positive"
    α::T   = 0.01                 ; @assert α   > 0 "Artificial viscosity (α) must be positive"
    g::T   = 9.81                 ; @assert g   > 0 "Gravitational constant (g) must be positive"
    c₀::T  = sqrt(g * 2) * 20     ; @assert c₀  > 0 "Speed of sound (c₀) must be positive"
    γ::TI  = 7                    ; @assert γ   > 0 "Adiabatic index (γ) must be positive"
    dt::T  = 1e-5                 ; @assert dt  > 0 "Time step (dt) must be positive"
    δᵩ::T  = 0.1                  ; @assert δᵩ  > 0 "Density variation (δᵩ) must be positive"
    CFL::T = 0.2                  ; @assert CFL > 0 "CFL condition (CFL) must be positive"
    η²::T  = (0.01 * H)^2         ; @assert η²  > 0 "Eta squared (η²) must be positive"
end

end
