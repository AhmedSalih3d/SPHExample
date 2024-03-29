module SimulationEquations

export EquationOfState, EquationOfStateGamma7, Pressure!, DensityEpsi!, LimitDensityAtBoundary!, ConstructGravitySVector, InverseHydrostaticEquationOfState

using StaticArrays
using Parameters
using FastPow

@inline function EquationOfStateGamma7(ρ,c₀,ρ₀)
    return @fastpow ((c₀^2*ρ₀)/7) * ((ρ/ρ₀)^7 - 1)
end

# Equation of State in Weakly-Compressible SPH
function EquationOfState(ρ,c₀,γ,ρ₀)
    return ((c₀^2*ρ₀)/γ) * ((ρ/ρ₀)^γ - 1)
end

@inline function Pressure!(Press, Density, SimulationConstants)
    @unpack c₀,γ,ρ₀ = SimulationConstants
    @inbounds Base.Threads.@threads for i ∈ eachindex(Press,Density)
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

# This version of the function using !Bool(MotionLimiter) instead of BoundaryBool
@inline function LimitDensityAtBoundary!(Density,ρ₀, MotionLimiter)
    @inbounds for i in eachindex(Density)
        if (Density[i] < ρ₀) * !Bool(MotionLimiter[i])
            Density[i] = ρ₀
        end
    end
end

@inline function ConstructGravitySVector(_::SVector{N, T}, value) where {N, T}
    return SVector{N, T}(ntuple(i -> i == N ? value : 0, N))
end

#https://discourse.julialang.org/t/can-this-be-written-even-faster-cpu/109924/28
@inline function Estimate7thRoot(x)
    # todo tune the magic constant
    # initial guess based on fast inverse sqrt trick but adjusted to compute x^(1/7)
    t = copysign(reinterpret(Float64, 0x36cd000000000000 + reinterpret(UInt64,abs(x))÷7), x)
    @fastmath for _ in 1:2
        # newton's method for t^3 - x/t^4 = 0
        t2 = t*t
        t3 = t2*t
        t4 = t2*t2
        xot4 = x/t4
        t = t - t*(t3 - xot4)/(4*t3 + 3*xot4)
    end
    t
end
@inline InverseHydrostaticEquationOfState(ρ₀, P, invCb) = ρ₀ * ( Estimate7thRoot( 1 + (P * invCb)) - 1)

end
