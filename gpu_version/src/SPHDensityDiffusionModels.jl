module SPHDensityDiffusionModels

using StaticArrays, LinearAlgebra
using ..SimulationEquations
using ..SimulationGeometry: Fluid
#---------------------------------------------------------------
# Exported
#---------------------------------------------------------------
export  SPHDensityDiffusion, 
        ZeroDensityDiffusion, 
        ZeroGravityLinearDensityDiffusion,
        LinearDensityDiffusion,
        ZeroGravityComplexDensityDiffusion,
        ComplexDensityDiffusion,
        FourtakasDensityDiffusion,
        compute_density_diffusion


#---------------------------------------------------------------
# Abstract supertype
#---------------------------------------------------------------
abstract type SPHDensityDiffusion end

"""
    compute_density_diffusion(model, SimKernel, SimConstants, SimParticles,
                              xᵢⱼ, ∇ᵢWᵢⱼ, d², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j, ParticleType)

Density diffusion contribution of the pair `(i, j)` for the selected `model`.
Returns `(Dᵢ, Dⱼ)`.

The interaction kernel supplies the densities of the state being evaluated
(the predictor density during the corrector loop) and their precomputed
reciprocals, so models neither reload nor divide by density. Same signature
as the CPU package. `SimParticles` is a NamedTuple of the evaluated state's
device arrays for custom models; the built in models only read
`ParticleType`.
"""
function compute_density_diffusion end

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
        d²,
        ρᵢ,
        ρⱼ,
        ρᵢ⁻¹,
        ρⱼ⁻¹,
        i,
        j,
        ParticleType
)
        return zero(d²), zero(d²)
end

#---------------------------------------------------------------
# 2) ZeroGravityLinearDensityDiffusion(): 
#---------------------------------------------------------------
"""
A linear density diffusion approach, but there is no hydrostatic 
term, and we skip ρᵢⱼᴴ enitrely.
"""
struct ZeroGravityLinearDensityDiffusion <: SPHDensityDiffusion end

@inline linear_density_difference(::ZeroGravityLinearDensityDiffusion, constants, x, ρᵢ, ρⱼ) = ρⱼ - ρᵢ

@inline function compute_density_diffusion(
        ::ZeroGravityLinearDensityDiffusion,
        SimKernel,
        SimConstants,
        SimParticles,
        xᵢⱼ,
        ∇ᵢWᵢⱼ,
        d²,
        ρᵢ,
        ρⱼ,
        ρᵢ⁻¹,
        ρⱼ⁻¹,
        i,
        j,
        ParticleType
)

        (; ρ₀, m₀, c₀, δᵩ, Cb, Cb⁻¹, γ) = SimConstants
        (; h, η²) = SimKernel

        # g == 0 => skip any hydrostatic parts
        
        invdᵢⱼ²η² = one(eltype(ρᵢ)) / (d² + η²)

        difference = linear_density_difference(ZeroGravityLinearDensityDiffusion(), SimConstants, xᵢⱼ, ρᵢ, ρⱼ)
        ψᵢⱼ = 2 * difference * (-xᵢⱼ) * invdᵢⱼ²η²

        Dᵢ  = δᵩ * h * c₀ * (m₀ * ρⱼ⁻¹) * dot(ψᵢⱼ, ∇ᵢWᵢⱼ)
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

@inline function linear_density_difference(::LinearDensityDiffusion, constants, x, ρᵢ, ρⱼ)
    (; Cb, γ, ρ₀, g) = constants
    factor = (1 / (Cb * γ)) * ρ₀
    pressure = ρ₀ * (-g) * -x[end]
    hydrostatic = pressure * factor
    return (ρⱼ - ρᵢ) - hydrostatic
end

@inline function compute_density_diffusion(
        ::LinearDensityDiffusion,
        SimKernel,
        SimConstants,
        SimParticles,
        xᵢⱼ,
        ∇ᵢWᵢⱼ,
        d²,
        ρᵢ,
        ρⱼ,
        ρᵢ⁻¹,
        ρⱼ⁻¹,
        i,
        j,
        ParticleType
)

        (; ρ₀, m₀, c₀, δᵩ, Cb, Cb⁻¹, γ, g) = SimConstants
        (; h, η²) = SimKernel

        invdᵢⱼ²η² = one(eltype(ρᵢ)) / (d² + η²)

        difference = linear_density_difference(LinearDensityDiffusion(), SimConstants, xᵢⱼ, ρᵢ, ρⱼ)
        ψᵢⱼ = 2 * difference * (-xᵢⱼ) * invdᵢⱼ²η²

        # The term only acts between two fluid particles. Gate it with a select
        # rather than an early `return` at the top of the function: measured on an
        # RTX A1000 (Float32, 9 interleaved rounds) the CPU style early exit was
        # within +-2% of the old MotionLimiter multiply on every case, because the
        # branch diverges inside the pair loop for fluid particles with boundary
        # neighbours, while the select made the pair kernel 4-10% faster in 3D and
        # was neutral in 2D. See gpu_version/README.md.
        both_fluid = (ParticleType[i] == Fluid) & (ParticleType[j] == Fluid)

        Dᵢ  = both_fluid ? δᵩ * h * c₀ * (m₀ * ρⱼ⁻¹) * dot(ψᵢⱼ, ∇ᵢWᵢⱼ) : zero(d²)
        Dⱼ  = -Dᵢ

        return Dᵢ, Dⱼ
end

"""
    radial_density_diffusion(model, kernel, constants, x, d², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j, types, factor, inverse_distance)

Scalar diffusion using the symmetric reciprocal-density weight
`(ρᵢ⁻¹ + ρⱼ⁻¹) / 2`. Swapping particle roles reverses the returned pair
contributions. This symmetrizes the reference linear diffusion operator;
it is not only a reassociation of its floating-point arithmetic.
"""
@inline function radial_density_diffusion(::ZeroDensityDiffusion, kernel, constants,
                                         x, d², ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j, types, factor, inverse_distance)
    return zero(d²), zero(d²)
end

@inline function radial_density_diffusion(model::Union{LinearDensityDiffusion, ZeroGravityLinearDensityDiffusion},
                                         kernel, constants, x, d²::T, ρᵢ, ρⱼ, ρᵢ⁻¹, ρⱼ⁻¹, i, j,
                                         types, factor, inverse_distance) where {T}
    difference = linear_density_difference(model, constants, x, ρᵢ, ρⱼ)
    # Both vectors are multiples of x: psi dot grad(W) = -A * factor * |x|^2.
    contraction = -(T(2) * difference * inverse_distance) * factor * d²
    inverse_density = T(1//2) * (ρᵢ⁻¹ + ρⱼ⁻¹)
    value = constants.δᵩ * kernel.h * constants.c₀ * (constants.m₀ * inverse_density) * contraction
    enabled = !(model isa LinearDensityDiffusion) || ((types[i] == Fluid) & (types[j] == Fluid))
    Dᵢ = enabled ? value : zero(T)
    return Dᵢ, -Dᵢ
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

"""
    FourtakasDensityDiffusion()

Target-oriented Fourtakas inner density diffusion (DualSPHysics DDT2).
The GPU interaction pass suppresses the complete contribution when the target's support
contains a prescribed wall. This differs from `ComplexDensityDiffusion`,
which retains the CPU reference's pair orientation and fluid-fluid terms
near walls. Existing diffusion models are unchanged.
"""
struct FourtakasDensityDiffusion <: SPHDensityDiffusion end

@inline function fourtakas_diffusion(kernel, constants, x, gradient, distance_squared, density_i, density_j, inverse_j, ::Val{Fast}=Val(false)) where {Fast}
    base = one(density_i) + (constants.ρ₀ * constants.g / constants.Cb) * x[end]
    root = Fast ? Base.FastMath.pow_fast(base,constants.γ⁻¹) : base^constants.γ⁻¹
    hydrostatic = constants.ρ₀ * root - constants.ρ₀
    coefficient = constants.δᵩ * kernel.H * constants.c₀ *
        ((density_j - density_i) - hydrostatic) / (distance_squared + kernel.η²)
    return -coefficient * dot(x, gradient) * constants.m₀ / density_j
end

@inline function compute_density_diffusion(::FourtakasDensityDiffusion, kernel, constants, particles,
        x, gradient, distance_squared, density_i, density_j, inverse_i, inverse_j, i, j, types)
    (types[i] == Fluid && types[j] == Fluid) || return zero(density_i), zero(density_j)
    first = fourtakas_diffusion(kernel, constants, x, gradient, distance_squared, density_i, density_j, inverse_j)
    second = fourtakas_diffusion(kernel, constants, -x, -gradient, distance_squared, density_j, density_i, inverse_i)
    return first, second
end

@inline function compute_density_diffusion(
        ::ComplexDensityDiffusion,
        SimKernel,
        SimConstants,
        SimParticles,
        xᵢⱼ,
        ∇ᵢWᵢⱼ,
        d²,
        ρᵢ,
        ρⱼ,
        ρᵢ⁻¹,
        ρⱼ⁻¹,
        i,
        j,
        ParticleType
)

        (; ρ₀, m₀, c₀, δᵩ, Cb, Cb⁻¹, γ, g) = SimConstants
        (; h, η²) = SimKernel

        # In theory these two equations are not completely symmetric.
        # In practice it is 'good' enough and saves a lot of time to
        # not do it the mathematically correct way.
        Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
        ρᵢⱼᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pᵢⱼᴴ, Cb⁻¹)
        # ρᵢⱼᴴ = ρ₀ * ( Estimate7thRoot( 1 + (Pᵢⱼᴴ * Cb⁻¹)) - 1)
        # ρⱼᵢᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pⱼᵢᴴ, Cb⁻¹)
        
        invdᵢⱼ²η² = one(eltype(ρᵢ)) / (d² + η²)

        ρⱼᵢ = ρⱼ - ρᵢ
        ψᵢⱼ = 2 * (ρⱼᵢ - ρᵢⱼᴴ)  * (-xᵢⱼ) * invdᵢⱼ²η²

        # Same fluid-fluid select gate as `LinearDensityDiffusion`, see there.
        both_fluid = (ParticleType[i] == Fluid) & (ParticleType[j] == Fluid)

        Dᵢ  = both_fluid ? δᵩ * h * c₀ * (m₀ * ρⱼ⁻¹) * dot(ψᵢⱼ, ∇ᵢWᵢⱼ) : zero(d²)
        Dⱼ  = -Dᵢ

        return Dᵢ, Dⱼ
end

end #module SPHDensityDiffusionModels
