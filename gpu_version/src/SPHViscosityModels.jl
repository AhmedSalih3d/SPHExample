module SPHViscosityModels

using StaticArrays, LinearAlgebra

export SPHViscosity, ZeroViscosity, ArtificialViscosity, Laminar, LaminarSPS, compute_viscosity

"""
    abstract type SPHViscosity end

Abstract supertype for all SPH viscosity models. Concrete models implement
`compute_viscosity` for their formulation.
"""
abstract type SPHViscosity end

"Represents a simulation with no viscous forces."
struct ZeroViscosity <: SPHViscosity end

"""
    ArtificialViscosity()

Monaghan style artificial viscosity for shock capturing and preventing
particle interpenetration.
"""
struct ArtificialViscosity <: SPHViscosity end

"""
    Laminar()

Standard laminar viscosity governed by the kinematic viscosity `ν₀`.
"""
struct Laminar <: SPHViscosity end

"""
    LaminarSPS()

Hybrid model combining `Laminar` viscosity with a Smagorinsky type
sub-particle scale turbulence closure.
"""
struct LaminarSPS <: SPHViscosity end


"""
    compute_viscosity(model, SimKernel, SimConstants, SimParticles,
                      xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j)

Compute the viscous acceleration between particles `i` and `j` for the
selected viscosity `model`. Returns `(Πᵢ, Πⱼ)`.

The interaction kernel supplies the densities of the state being evaluated
(the predictor density during the corrector loop) together with their
precomputed reciprocals, so models neither reload nor divide by density.
Same signature as the CPU package. `SimParticles` is a NamedTuple of the
evaluated state's device arrays (`Position`, `Density`, `Velocity`,
`Pressure`, `Type`) for custom models that need additional fields; the
built in models never touch it.
"""

# No viscosity: return zero contributions.
@inline function compute_viscosity(::ZeroViscosity, SimKernel, SimConstants, SimParticles,
                                   xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j)
    return zero(xᵢⱼ), zero(xᵢⱼ)
end

# Artificial viscosity formulation.
@inline function compute_viscosity(::ArtificialViscosity, SimKernel, SimConstants, SimParticles,
                                   xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j)
    (; m₀, α, c₀) = SimConstants
    (; h, η²) = SimKernel

    v_dot_x = dot(vᵢⱼ, xᵢⱼ)
    if v_dot_x < 0
        ρ̄ = (ρᵢ + ρⱼ) / 2
        μᵢⱼ = h * v_dot_x / (d² + η²)

        Π = -m₀ * (-α * c₀ * μᵢⱼ) / ρ̄ * ∇ᵢWᵢⱼ
        return Π, -Π
    end

    return zero(xᵢⱼ), zero(xᵢⱼ)
end

# Laminar viscosity formulation.
@inline function compute_viscosity(::Laminar, SimKernel, SimConstants, SimParticles,
                                   xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j)
    (; m₀, ν₀) = SimConstants
    (; η²) = SimKernel

    term = (4 * m₀ * ν₀ * dot(xᵢⱼ, ∇ᵢWᵢⱼ)) / ((ρᵢ + ρⱼ) * (d² + η²))
    return term * vᵢⱼ, -term * vᵢⱼ
end

@inline radial_viscosity(::ZeroViscosity, kernel, constants, x, v, gradient, ρᵢ, ρⱼ, inverse_distance) = zero(x)

@inline function radial_viscosity(::ArtificialViscosity, kernel, constants, x, v, gradient, ρᵢ, ρⱼ, inverse_distance)
    v_dot_x = dot(v, x)
    if v_dot_x < 0
        mean_density = (ρᵢ + ρⱼ) / 2
        μ = kernel.h * v_dot_x * inverse_distance
        return -constants.m₀ * (-constants.α * constants.c₀ * μ) / mean_density * gradient
    end
    return zero(x)
end

@inline function radial_viscosity(::Laminar, kernel, constants, x, v, gradient, ρᵢ, ρⱼ, inverse_distance)
    term = (4 * constants.m₀ * constants.ν₀ * dot(x, gradient)) * inverse_distance / (ρᵢ + ρⱼ)
    return term * v
end

# LaminarSPS: with sub-grid scale stresses.
@inline function compute_viscosity(::LaminarSPS, SimKernel, SimConstants, SimParticles,
                                   xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j)
    (; m₀, dx, SmagorinskyConstant, BlinConstant) = SimConstants

    t1, t2 = compute_viscosity(Laminar(), SimKernel, SimConstants, SimParticles,
                               xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j)

    T        = eltype(xᵢⱼ)
    Iᴹ       = diagm(one.(xᵢⱼ))
    third    = T(1)/3
    twothird = T(2)/3
    #julia> a .- a'
    # 3×3 SMatrix{3, 3, Float64, 9} with indices SOneTo(3)×SOneTo(3):
    # 0.0  0.0  0.0
    # 0.0  0.0  0.0
    # 0.0  0.0  0.0
    # Strain *rate* tensor is the gradient of velocity. The kernel's vᵢⱼ is the
    # velocity difference of the evaluated state, so vⱼ - vᵢ == -vᵢⱼ.
    Sᵢ = ∇vᵢ =  (m₀ * ρⱼ⁻¹) * (-vᵢⱼ) * ∇ᵢWᵢⱼ'
    norm_Sᵢ  = sqrt(2 * sum(Sᵢ .^ 2))
    νtᵢ      = (SmagorinskyConstant * dx)^2 * norm_Sᵢ
    trace_Sᵢ = sum(diag(Sᵢ))
    τᶿᵢ      = 2*νtᵢ*ρᵢ * (Sᵢ - third * trace_Sᵢ * Iᴹ) - twothird * ρᵢ * BlinConstant * dx^2 * norm_Sᵢ^2 * Iᴹ
    Sⱼ = ∇vⱼ =  (m₀ * ρᵢ⁻¹) * vᵢⱼ * -∇ᵢWᵢⱼ'
    norm_Sⱼ  = sqrt(2 * sum(Sⱼ .^ 2))
    νtⱼ      = (SmagorinskyConstant * dx)^2 * norm_Sⱼ
    trace_Sⱼ = sum(diag(Sⱼ))
    τᶿⱼ      = 2*νtⱼ*ρⱼ * (Sⱼ - third * trace_Sⱼ * Iᴹ) - twothird * ρⱼ * BlinConstant * dx^2 * norm_Sⱼ^2 * Iᴹ

    # MATHEMATICALLY THIS IS DOT PRODUCT TO GO FROM TENSOR TO VECTOR, BUT USE * IN JULIA TO REPRESENT IT
    dτdtᵢ = (m₀ * ρᵢ⁻¹ * ρⱼ⁻¹) * (τᶿᵢ + τᶿⱼ) *  ∇ᵢWᵢⱼ
    dτdtⱼ = -dτdtᵢ

    return t1 + dτdtᵢ, t2 + dτdtⱼ
end

end  # module SPHViscosityModels
