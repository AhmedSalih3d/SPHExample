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
                      xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², i, j)

Compute the viscous acceleration between particles `i` and `j` for the
selected viscosity `model`. Returns `(Πᵢ, Πⱼ)`.
"""

# No viscosity: return zero contributions.
@inline function compute_viscosity(::ZeroViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², i, j)
    return zero(xᵢⱼ), zero(xᵢⱼ)
end

# Artificial viscosity formulation.
@inline function compute_viscosity(::ArtificialViscosity, SimKernel, SimConstants, SimParticles,
                                   xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², i, j)
    (; m₀, α, c₀) = SimConstants
    (; h, η²) = SimKernel

    ρᵢ = SimParticles.Density[i]
    ρⱼ = SimParticles.Density[j]

    v_dot_x = dot(vᵢⱼ, xᵢⱼ)
    if v_dot_x < 0
        ρ̄ = 0.5 * (ρᵢ + ρⱼ)
        μᵢⱼ = h * v_dot_x / (d² + η²)

        Π = -m₀ * (-α * c₀ * μᵢⱼ) / ρ̄ * ∇ᵢWᵢⱼ
        return Π, -Π
    end

    return zero(xᵢⱼ), zero(xᵢⱼ)
end

# Laminar viscosity formulation.
@inline function compute_viscosity(::Laminar, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², i, j)
    (; m₀, ν₀) = SimConstants
    (; η²) = SimKernel

    dᵢⱼ =  sqrt(abs(d²))
    ρᵢ  = SimParticles.Density[i]
    ρⱼ  = SimParticles.Density[j]

    term = (4 * m₀ * ν₀ * dot(xᵢⱼ, ∇ᵢWᵢⱼ)) / ((ρᵢ + ρⱼ) + (d² + η²))
    return term * vᵢⱼ, -term * vᵢⱼ
end

# LaminarSPS: with sub-grid scale stresses.
@inline function compute_viscosity(::LaminarSPS, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², i, j)
    (; m₀, dx, SmagorinskyConstant, BlinConstant) = SimConstants
    
    t1,t2 = compute_viscosity(Laminar(), SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², i, j)
    

    ρᵢ  = SimParticles.Density[i]
    ρⱼ  = SimParticles.Density[j]

    vᵢ  = SimParticles.Velocity[i]
    vⱼ  = SimParticles.Velocity[j]


    Iᴹ       = diagm(one.(xᵢⱼ))
    #julia> a .- a'
    # 3×3 SMatrix{3, 3, Float64, 9} with indices SOneTo(3)×SOneTo(3):
    # 0.0  0.0  0.0
    # 0.0  0.0  0.0
    # 0.0  0.0  0.0
    # Strain *rate* tensor is the gradient of velocity
    Sᵢ = ∇vᵢ =  (m₀/ρⱼ) * (vⱼ - vᵢ) * ∇ᵢWᵢⱼ'
    norm_Sᵢ  = sqrt(2 * sum(Sᵢ .^ 2))
    νtᵢ      = (SmagorinskyConstant * dx)^2 * norm_Sᵢ
    trace_Sᵢ = sum(diag(Sᵢ))
    τᶿᵢ      = 2*νtᵢ*ρᵢ * (Sᵢ - (1/3) * trace_Sᵢ * Iᴹ) - (2/3) * ρᵢ * BlinConstant * dx^2 * norm_Sᵢ^2 * Iᴹ
    Sⱼ = ∇vⱼ =  (m₀/ρᵢ) * (vᵢ - vⱼ) * -∇ᵢWᵢⱼ'
    norm_Sⱼ  = sqrt(2 * sum(Sⱼ .^ 2))
    νtⱼ      = (SmagorinskyConstant * dx)^2 * norm_Sⱼ
    trace_Sⱼ = sum(diag(Sⱼ))
    τᶿⱼ      = 2*νtⱼ*ρⱼ * (Sⱼ - (1/3) * trace_Sⱼ * Iᴹ) - (2/3) * ρⱼ * BlinConstant * dx^2 * norm_Sⱼ^2 * Iᴹ

    # MATHEMATICALLY THIS IS DOT PRODUCT TO GO FROM TENSOR TO VECTOR, BUT USE * IN JULIA TO REPRESENT IT
    dτdtᵢ = (m₀/(ρⱼ * ρᵢ)) * (τᶿᵢ + τᶿⱼ) *  ∇ᵢWᵢⱼ 
    dτdtⱼ = -dτdtᵢ

    return t1 + dτdtᵢ, t2 + dτdtⱼ
end

end  # module SPHViscosityModels
