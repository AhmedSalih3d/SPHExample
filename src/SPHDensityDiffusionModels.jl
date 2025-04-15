module SPHDensityDiffusionModels

using StaticArrays, LinearAlgebra, Parameters
using .SimulationEquations
#---------------------------------------------------------------
# Exported
#---------------------------------------------------------------
export  SPHDensityDiffusion, 
        ZeroDensityDiffusion, 
        ZeroGravityLinearDensityDiffusion,
        LinearDensityDiffusion,
        ZeroGravityComplexDensityDiffusion,
        ComplexDensityDiffusion,
        compute_density_diffusion


#---------------------------------------------------------------
# Abstract supertype
#---------------------------------------------------------------
abstract type SPHDensityDiffusion end

#---------------------------------------------------------------
# 1) ZeroDensityDiffusion(): ignore all diffusion
#---------------------------------------------------------------
"""
        ZeroDensityDiffusion()
A model that always returns zero(). No extra density diffusion
and ignores all other parameters.
"""
struct ZeroDensityDiffusion <: SPHDensityDiffusion end

@inline function compute_density_diffusion(
        ::ZeroDensityDiffusion,
        SimKernel,
        SimConstants,
        SimParticles,
        xᵢⱼ,
        ∇ᵢWᵢⱼ,
        i,
        j,
        MotionLimiter
)
        return zero(xᵢⱼ)
end

#---------------------------------------------------------------
# 2) ZeroGravityLinearDensityDiffusion(): 
#---------------------------------------------------------------
"""
A linear density diffusion approach, but there is no hydrostatic 
term, and we skip ρᵢⱼᴴ enitrely.
"""
struct ZeroGravityLinearDensityDiffusion <: SPHDensityDiffusion end

@inline function compute_density_diffusion(
        ::ZeroGravityLinearDensityDiffusion,
        SimKernel,
        SimConstants,
        SimParticles,
        xᵢⱼ,
        ∇ᵢWᵢⱼ,
        i,
        j,
        MotionLimiter
)

        @unpack ρ₀, m₀, c₀, δᵩ, Cb, Cb⁻¹, γ    = SimConstants
        @unpack h, η²                          = SimKernel

        ρᵢ  = SimParticles.Density[i]
        ρⱼ  = SimParticles.Density[j]

        # g == 0 => skip any hydrostatic parts
        
        dᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)

        invdᵢⱼ²η² = one(eltype(ρᵢ)) / (dᵢⱼ² + η²)

        ρⱼᵢ = ρⱼ - ρᵢ
        ψᵢⱼ = 2 * ρⱼᵢ * (-xᵢⱼ) * invdᵢⱼ²η²

        Dᵢ  = δᵩ * h * c₀ * (m₀/ρⱼ) * dot(ψᵢⱼ, ∇ᵢWᵢⱼ)
        Dⱼ  = -Dᵢ


        return Dᵢ, Dⱼ
end

#---------------------------------------------------------------
# 3) LinearDensityDiffusion(): Linear approach, uses gravity from
# SimConstants.g
#---------------------------------------------------------------
"""
        LinearDensityDiffusion()

Uses a linear relationship for the hydrostatic correction.
"""
struct LinearDensityDiffusion <: SPHDensityDiffusion end

@inline function compute_density_diffusion(
        ::LinearDensityDiffusion,
        SimKernel,
        SimConstants,
        SimParticles,
        xᵢⱼ,
        ∇ᵢWᵢⱼ,
        i,
        j,
        MotionLimiter
)

        @unpack ρ₀, m₀, c₀, δᵩ, Cb, Cb⁻¹, γ, g = SimConstants
        @unpack h, η²                          = SimKernel

        Linear_ρ_factor = (1/(Cb*γ))*ρ₀

        ρᵢ  = SimParticles.Density[i]
        ρⱼ  = SimParticles.Density[j]

        Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
        ρᵢⱼᴴ  = Pᵢⱼᴴ * Linear_ρ_factor

        
        dᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)

        invdᵢⱼ²η² = one(eltype(ρᵢ)) / (dᵢⱼ² + η²)

        ρⱼᵢ = ρⱼ - ρᵢ
        ψᵢⱼ = 2 * (ρⱼᵢ - ρᵢⱼᴴ)  * (-xᵢⱼ) * invdᵢⱼ²η²

        MLcond = MotionLimiter[i] * MotionLimiter[j]

        Dᵢ  = δᵩ * h * c₀ * (m₀/ρⱼ) * dot(ψᵢⱼ, ∇ᵢWᵢⱼ) * MLcond
        Dⱼ  = -Dᵢ

        return Dᵢ, Dⱼ
end

#---------------------------------------------------------------
# 3) LinearDensityDiffusion(): Linear approach, uses gravity from
# SimConstants.g
#---------------------------------------------------------------
"""
        ComplexDensityDiffusion()

Uses a 'complex' relationship for the hydrostatic correction. In essence the inverse
hydrostatic equation of state.
"""
struct ComplexDensityDiffusion <: SPHDensityDiffusion end

@inline function compute_density_diffusion(
        ::ComplexDensityDiffusion,
        SimKernel,
        SimConstants,
        SimParticles,
        xᵢⱼ,
        ∇ᵢWᵢⱼ,
        i,
        j,
        MotionLimiter
)

        @unpack ρ₀, m₀, c₀, δᵩ, Cb, Cb⁻¹, γ, g = SimConstants
        @unpack h, η²                          = SimKernel

        ρᵢ  = SimParticles.Density[i]
        ρⱼ  = SimParticles.Density[j]

        # In theory these two equations are not completely symmetric.
        # In practice it is 'good' enough and saves a lot of time to
        # not do it the mathematically correct way.
        Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
        ρᵢⱼᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pᵢⱼᴴ, Cb⁻¹)
        # ρᵢⱼᴴ = ρ₀ * ( Estimate7thRoot( 1 + (Pᵢⱼᴴ * Cb⁻¹)) - 1)
        # ρⱼᵢᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pⱼᵢᴴ, Cb⁻¹)
        
        dᵢⱼ² = dot(xᵢⱼ, xᵢⱼ)

        invdᵢⱼ²η² = one(eltype(ρᵢ)) / (dᵢⱼ² + η²)

        ρⱼᵢ = ρⱼ - ρᵢ
        ψᵢⱼ = 2 * (ρⱼᵢ - ρᵢⱼᴴ)  * (-xᵢⱼ) * invdᵢⱼ²η²

        MLcond = MotionLimiter[i] * MotionLimiter[j]

        Dᵢ  = δᵩ * h * c₀ * (m₀/ρⱼ) * dot(ψᵢⱼ, ∇ᵢWᵢⱼ) * MLcond
        Dⱼ  = -Dᵢ

        return Dᵢ, Dⱼ
end

end #module SPHDensityDiffusionModels