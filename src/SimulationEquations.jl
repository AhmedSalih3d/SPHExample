module SimulationEquations

export EquationOfState, EquationOfStateGamma7, Pressure!, DensityEpsi!, LimitDensityAtBoundary!, ConstructGravitySVector, InverseHydrostaticEquationOfState, Estimate7thRoot

using StaticArrays
using Parameters
using FastPow
using ..SimulationGeometry

@inline function EquationOfStateGamma7(ρ,c₀,ρ₀)
    return @fastpow ((c₀^2*ρ₀)/7) * ((ρ/ρ₀)^7 - 1)
end

# Equation of State in Weakly-Compressible SPH
function EquationOfState(ρ,c₀,γ,ρ₀)
    return ((c₀^2*ρ₀)/γ) * ((ρ/ρ₀)^γ - 1)
end

@inline function Pressure!(Press, Density, SimulationConstants)
    @unpack c₀,γ,ρ₀ = SimulationConstants
    @inbounds for i ∈ eachindex(Press,Density)
        # Press[i] = EquationOfState(Density[i],c₀,γ,ρ₀)
        Press[i] = EquationOfStateGamma7(Density[i],c₀,ρ₀)
    end
end

# This is to handle the special factor multiplied on density in the time stepping procedure, when
# using symplectic time stepping
@inline function DensityEpsi!(Density, dρdtIₙ⁺,ρₙ⁺,Δt)
    @inbounds for i in eachindex(Density)
        epsi = - (dρdtIₙ⁺[i] / ρₙ⁺[i]) * Δt
        Density[i] *= (2 - epsi) / (2 + epsi)
    end
end

# This version of the function uses ParticleType instead of BoundaryBool
@inline function LimitDensityAtBoundary!(Density, ρ₀, ParticleType)
    @inbounds for i in eachindex(Density)
        if (Density[i] < ρ₀) && (ParticleType[i] != Fluid)
            Density[i] = ρ₀
        end
    end
end

@inline function ConstructGravitySVector(_::SVector{N, T}, value) where {N, T}
    return SVector{N, T}(ntuple(i -> i == N ? value : 0, N))
end

@inline function Estimate7thRoot(x::T) where {T<:AbstractFloat}
    return copysign(abs(x)^(inv(T(7))), x)
end
@inline Estimate7thRoot(x::Real) = Estimate7thRoot(float(x))
@inline InverseHydrostaticEquationOfState(ρ₀, P, invCb) = ρ₀ * (Estimate7thRoot(one(P * invCb) + (P * invCb)) - one(P * invCb))

end
