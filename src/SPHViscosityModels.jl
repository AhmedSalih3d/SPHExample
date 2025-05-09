module SPHViscosityModels

using StaticArrays, LinearAlgebra, Parameters

export SPHViscosity, ZeroViscosity, ArtificialViscosity, Laminar, LaminarSPS, compute_viscosity

# Abstract type and specific viscosity models
abstract type SPHViscosity end

struct ZeroViscosity        <: SPHViscosity end
struct ArtificialViscosity  <: SPHViscosity end
struct Laminar              <: SPHViscosity end
struct LaminarSPS           <: SPHViscosity end

# No viscosity: return zero contributions.
@inline function compute_viscosity(::ZeroViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, i, j)
    return zero(xᵢⱼ), zero(xᵢⱼ)
end

# Artificial viscosity formulation.
@inline function compute_viscosity(::ArtificialViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, i, j)
    @unpack ρ₀, m₀, α, γ, g, c₀, δᵩ, Cb, Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants
    @unpack h, η² = SimKernel

    dᵢⱼ =  sqrt(abs(dot(xᵢⱼ,xᵢⱼ)))
    ρᵢ  = SimParticles.Density[i]
    ρⱼ  = SimParticles.Density[j]

    invd²η²   =  1.0 / (dᵢⱼ*dᵢⱼ+η²)
    ρ̄ = (ρᵢ + ρⱼ) * 0.5
    cond = dot(vᵢⱼ, xᵢⱼ)
    flag = cond < 0 ? one(eltype(cond)) : zero(eltype(cond))
    μ = h * cond * invd²η²
    Π = -m₀ * (flag * (-α * c₀ * μ) / ρ̄) * ∇ᵢWᵢⱼ
    return Π, -Π
end

# Laminar viscosity formulation.
@inline function compute_viscosity(::Laminar, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, i, j)
    @unpack ρ₀, m₀, α, γ, g, c₀, δᵩ, Cb, Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants
    @unpack η² = SimKernel

    dᵢⱼ =  sqrt(abs(dot(xᵢⱼ,xᵢⱼ)))
    ρᵢ  = SimParticles.Density[i]
    ρⱼ  = SimParticles.Density[j]

    term = (4 * m₀ * ν₀ * dot(xᵢⱼ, ∇ᵢWᵢⱼ)) / ((ρᵢ + ρⱼ) + (dᵢⱼ*dᵢⱼ + η²))
    return term * vᵢⱼ, -term * vᵢⱼ
end

# LaminarSPS: with sub-grid scale stresses.
@inline function compute_viscosity(::LaminarSPS, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, i, j)
    @unpack ρ₀, m₀, α, γ, g, c₀, δᵩ, Cb, Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants
    @unpack η² = SimKernel
    
    t1,t2 = compute_viscosity(Laminar(), SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, i, j)
    

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
