module SPHViscosityModels

using StaticArrays, LinearAlgebra, Parameters

export SPHViscosity, ZeroViscosity, ArtificialViscosity, Laminar, LaminarSPS, compute_viscosity

abstract type SPHViscosity end

struct ZeroViscosity <: SPHViscosity end

@with_kw struct ArtificialViscosity{T<:Union{Float32, Float64}} <: SPHViscosity
    α::T
end

@with_kw struct Laminar{T<:Union{Float32, Float64}} <: SPHViscosity
    ν::T = 1e-6
end


@with_kw struct LaminarSPS{T<:Union{Float32, Float64}} <: SPHViscosity
    ν::T                   = 1e-6
    SmagorinskyConstant::T = 0.12
    BlinConstant::T        = 0.0066
end


# No viscosity: return zero contributions.
@inline function compute_viscosity(::ZeroViscosity, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², i, j)
    return zero(xᵢⱼ), zero(xᵢⱼ)
end

# Artificial viscosity formulation.
@inline function compute_viscosity(SimViscosity::ArtificialViscosity, SimKernel, SimConstants, SimParticles,
                                   xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², i, j)
    @unpack m₀, c₀ = SimConstants
    @unpack h, η²     = SimKernel
    α = SimViscosity.α

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
@inline function compute_viscosity(SimViscosity::Laminar, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², i, j)
    @unpack m₀ = SimConstants
    @unpack η²     = SimKernel
    ν = SimViscosity.ν

    dᵢⱼ =  sqrt(abs(d²))
    ρᵢ  = SimParticles.Density[i]
    ρⱼ  = SimParticles.Density[j]

    term = (4 * m₀ * ν * dot(xᵢⱼ, ∇ᵢWᵢⱼ)) / ((ρᵢ + ρⱼ) + (d² + η²))
    return term * vᵢⱼ, -term * vᵢⱼ
end

# LaminarSPS: with sub-grid scale stresses.
@inline function compute_viscosity(SimViscosity::LaminarSPS, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², i, j)
    @unpack m₀, dx = SimConstants
    @unpack ν, SmagorinskyConstant, BlinConstant = SimViscosity
    
    t1,t2 = compute_viscosity(Laminar(ν = ν), SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, d², i, j)
    

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
