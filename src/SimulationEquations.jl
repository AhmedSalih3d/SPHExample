module SimulationEquations

export Wᵢⱼ, ∑ⱼWᵢⱼ!, Optim∇ᵢWᵢⱼ, ∑ⱼ∇ᵢWᵢⱼ!, Pressure, Pressure!, ∂Πᵢⱼ∂t!, ∂ρᵢ∂tDDT!, ∂vᵢ∂t!, DensityEpsi!, LimitDensityAtBoundary!, updatexᵢⱼ!

using CellListMap
using StaticArrays
using LinearAlgebra
using Parameters
using LoopVectorization

# Function to calculate Kernel Value
function Wᵢⱼ(αD,q)
    return αD*(1-q/2)^4*(2*q + 1)
end

# Function to calculate kernel value in both "particle i" format and "list of interactions" format
# Please notice how when using CellListMap since it is based on a "list of interactions", for each 
# interaction we must add the contribution to both the i'th and j'th particle!
function ∑ⱼWᵢⱼ!(Kernel, KernelL, I, J, D, SimulationConstants)
    @unpack αD, h⁻¹ = SimulationConstants
    
    # Calculation
    @tturbo for iter in eachindex(D)
        d = D[iter]

        q = d * h⁻¹

        W = Wᵢⱼ(αD,q)

        KernelL[iter] = W
    end

    # Reduction
    for iter in eachindex(I,J)
        i = I[iter]
        j = J[iter]
        
        Kernel[i] += KernelL[iter]
        Kernel[j] += KernelL[iter]
    end

    return nothing
end
# Original implementation of kernel gradient
# function ∇ᵢWᵢⱼ(αD,q,xᵢⱼ,h)
#     # Skip distances outside the support of the kernel:
#     if q < 0.0 || q > 2.0
#         return SVector(0.0,0.0,0.0)
#     end

#     gradWx = αD * 1/h * (5*(q-2)^3*q)/8 * (xᵢⱼ[1] / (q*h+1e-6))
#     gradWy = αD * 1/h * (5*(q-2)^3*q)/8 * (xᵢⱼ[2] / (q*h+1e-6))
#     gradWz = αD * 1/h * (5*(q-2)^3*q)/8 * (xᵢⱼ[3] / (q*h+1e-6)) 

#     return SVector(gradWx,gradWy,gradWz)
# end

# This is a much faster version of ∇ᵢWᵢⱼ
function Optim∇ᵢWᵢⱼ(αD,q,xᵢⱼ,h) 
    # Skip distances outside the support of the kernel:
    if 0 < q < 2
        Fac = αD*5*(q-2)^3*q / (8h*(q*h+1e-6)) 
    else
        Fac = 0.0 # or return zero(xᵢⱼ) 
    end
    return Fac .* xᵢⱼ
end

# Function to calculate kernel gradient value in both "particle i" format and "list of interactions" format
# Please notice how when using CellListMap since it is based on a "list of interactions", for each 
# interaction we must add the contribution to both the i'th and j'th particle!
function ∑ⱼ∇ᵢWᵢⱼ!(KernelGradientIˣ,KernelGradientIʸ,KernelGradientIᶻ,KernelGradientLˣ,KernelGradientLʸ,KernelGradientLᶻ, I, J, D, xᵢⱼˣ, xᵢⱼʸ, xᵢⱼᶻ, SimulationConstants)
    @unpack αD, h, h⁻¹, η² = SimulationConstants
 
    @tturbo for iter in eachindex(I)
        i = I[iter]; j = J[iter]; d = D[iter]

        q = clamp(d * h⁻¹, 0.0, 2.0)

        Fac = αD*5*(q-2)^3*q / (8h*(q*h+η²)) 

        ∇ᵢWᵢⱼˣ = Fac * xᵢⱼˣ[iter]  
        ∇ᵢWᵢⱼʸ = Fac * xᵢⱼʸ[iter]  
        ∇ᵢWᵢⱼᶻ = Fac * xᵢⱼᶻ[iter] 

        KernelGradientLˣ[iter] =  ∇ᵢWᵢⱼˣ
        KernelGradientLʸ[iter] =  ∇ᵢWᵢⱼʸ
        KernelGradientLᶻ[iter] =  ∇ᵢWᵢⱼᶻ
    end


    for iter in eachindex(I,J)
        i = I[iter]
        j = J[iter]

        ∇ᵢWᵢⱼˣ                 =  KernelGradientLˣ[iter]
        ∇ᵢWᵢⱼʸ                 =  KernelGradientLʸ[iter]
        ∇ᵢWᵢⱼᶻ                 =  KernelGradientLᶻ[iter]

        KernelGradientIˣ[i]   +=  ∇ᵢWᵢⱼˣ
        KernelGradientIˣ[j]   += -∇ᵢWᵢⱼˣ
        KernelGradientIʸ[i]   +=  ∇ᵢWᵢⱼʸ
        KernelGradientIʸ[j]   += -∇ᵢWᵢⱼʸ
        KernelGradientIᶻ[i]   +=  ∇ᵢWᵢⱼᶻ
        KernelGradientIᶻ[j]   += -∇ᵢWᵢⱼᶻ
    end

    return nothing
end


# Equation of State in Weakly-Compressible SPH
function Pressure(ρ,c₀,γ,ρ₀)
    return ((c₀^2*ρ₀)/γ) * ((ρ/ρ₀)^γ - 1)
end

@inline @inbounds function Pressure!(Press, Density, SimulationConstants)
    @unpack c₀,γ,ρ₀ = SimulationConstants
    @tturbo for i ∈ eachindex(Press,Density)
        Press[i] = Pressure(Density[i],c₀,γ,ρ₀)
    end
end

# The artificial viscosity term
function ∂Πᵢⱼ∂t!(viscI, list,xᵢⱼ,ρ,v,WgL,SimulationConstants)
    @unpack h, α, c₀, m₀, η² = SimulationConstants

    for (iter,L) in enumerate(list)
        i = L[1]; j = L[2]; d = L[3]
        
        ρᵢ    = ρ[i]
        ρⱼ    = ρ[j]
        vᵢⱼ   = v[i] - v[j]
        xᵢⱼ⁰  = xᵢⱼ[iter] #xᵢⱼ   = points[i] - points[j]
        d²    = d*d
        ρᵢⱼ   = (ρᵢ+ρⱼ)*0.5

        cond      = dot(vᵢⱼ,xᵢⱼ⁰)

        cond_bool = cond < 0

        μᵢⱼ = h*cond/(d²+η²)
        Πᵢⱼ = cond_bool*(-α*c₀*μᵢⱼ)/ρᵢⱼ
        visc_val = -Πᵢⱼ*m₀*WgL[iter]
        
        viscI[i] +=  visc_val
        viscI[j] += -visc_val
    end

    return nothing
end


@inline function fancy7th(x)
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

#https://discourse.julialang.org/t/can-this-be-written-even-faster-cpu/109924/28
#faux(ρ₀, P, invCb, γ⁻¹) = ρ₀ * ( ^( 1 + (P * invCb), γ⁻¹) - 1)
@inline faux(ρ₀, P, invCb, γ⁻¹) = ρ₀ * (expm1(γ⁻¹ * log1p(P * invCb)))
@inline faux_fancy(ρ₀, P, Cb) = ρ₀ * ( fancy7th( 1 + (P * Cb)) - 1)
#faux(ρ₀, P, invCb) = ρ₀ * ( fancy7th( 1 + (P * invCb)) - 1)

# The density derivative function INCLUDING density diffusion
function ∂ρᵢ∂tDDT!(dρdtI, list, xᵢⱼ,xᵢⱼʸ,ρ,v,WgL,MotionLimiter, drhopLp, drhopLn, SimulationConstants)
    @unpack h,m₀,δᵩ,c₀,γ,g,ρ₀,η²,γ⁻¹ = SimulationConstants

    # Generate the needed constants
    Cb    = (c₀^2*ρ₀)/γ
    invCb = inv(Cb)

    @tturbo for iter in eachindex(list)
        Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼʸ[iter]
        ρᵢⱼᴴ  = faux_fancy(ρ₀, Pᵢⱼᴴ, invCb)
        Pⱼᵢᴴ  = -Pᵢⱼᴴ
        ρⱼᵢᴴ  = faux_fancy(ρ₀, Pⱼᵢᴴ, invCb)
        
        drhopLp[iter] = ρᵢⱼᴴ
        drhopLn[iter] = ρⱼᵢᴴ
    end

    for (iter,L) in enumerate(list)
        i = L[1]; j = L[2]; d = L[3]

        #xⱼᵢ   = points[j] - points[i]
        xⱼᵢ   = -xᵢⱼ[iter]
        r²    = d*d
        ρᵢ    = ρ[i]
        ρⱼ    = ρ[j]
        ρⱼᵢ   = ρⱼ - ρᵢ
        vᵢⱼ   = v[i] - v[j]
        ∇ᵢWᵢⱼ = WgL[iter]
        
        # First part of continuity equation
        FirstPartOfContinuity = dot(m₀*vᵢⱼ,∇ᵢWᵢⱼ) # =dot(m₀*-vᵢⱼ,-∇ᵢWᵢⱼ)

        # Follow the implementation here: https://arxiv.org/abs/2110.10076
        # Implement for particle i
        # Pᵢⱼᴴ = ρ₀ * (-g) * xⱼᵢ[2]
        # ρᵢⱼᴴ = ρ₀ * ( ^( 1 + (Pᵢⱼᴴ/Cb), γ⁻¹) - 1)
        ρᵢⱼᴴ = drhopLp[iter]
        Ψᵢⱼ  = 2 * (ρⱼᵢ - ρᵢⱼᴴ) * xⱼᵢ/(r²+η²)
        Dᵢ   = δᵩ * h * c₀ * (m₀/ρⱼ) * dot(Ψᵢⱼ,∇ᵢWᵢⱼ)

        dρdtI[i] += FirstPartOfContinuity + Dᵢ * MotionLimiter[i]

        # Implement for particle j
        # Pⱼᵢᴴ = -Pᵢⱼᴴ
        # ρⱼᵢᴴ = ρ₀ * ( ^( 1 + (Pⱼᵢᴴ/Cb), γ⁻¹) - 1)
        ρⱼᵢᴴ = drhopLn[iter]
        Ψⱼᵢ  = 2 * (-ρⱼᵢ - ρⱼᵢᴴ) * (-xⱼᵢ)/(r²+η²)
        Dⱼ   = δᵩ * h * c₀ * (m₀/ρᵢ) * dot(Ψⱼᵢ,-∇ᵢWᵢⱼ)

        dρdtI[j] += FirstPartOfContinuity + Dⱼ * MotionLimiter[i]
    end

    return nothing
end

# The momentum equation without any dissipation - we add the dissipation using artificial viscosity (∂Πᵢⱼ∂t)
function ∂vᵢ∂t!(dvdtI, list,ρ,WgL,press, SimulationConstants)
    @unpack m₀, c₀,γ,ρ₀ = SimulationConstants

    for (iter,L) in enumerate(list)
        i = L[1]; j = L[2]

        ρᵢ    = ρ[i]
        ρⱼ    = ρ[j]
        Pᵢ    = press[i] #Pᵢ    = Pressure(ρᵢ,c₀,γ,ρ₀)
        Pⱼ    = press[j] #Pⱼ    = Pressure(ρⱼ,c₀,γ,ρ₀)
        ∇ᵢWᵢⱼ = WgL[iter]

        Pfac  = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)


        dvdt  = - m₀ * Pfac *  ∇ᵢWᵢⱼ

        dvdtI[i]    +=  dvdt
        dvdtI[j]    += -dvdt
    end

    return nothing
end

# This is to handle the special factor multiplied on density in the time stepping procedure, when
# using symplectic time stepping
function DensityEpsi!(Density, dρdtIₙ⁺,ρₙ⁺,Δt)
    for i in eachindex(Density)
        epsi = - (dρdtIₙ⁺[i] / ρₙ⁺[i]) * Δt
        Density[i] *= (2 - epsi) / (2 + epsi)
    end
end

# This function is used to limit density at boundary to ρ₀ to avoid suctions at walls. Walls should
# only push and not attract so this is fine!
function LimitDensityAtBoundary!(Density,BoundaryBool,ρ₀)
    for i in eachindex(Density)
        if (Density[i] < ρ₀) * Bool(BoundaryBool[i])
            Density[i] = ρ₀
        end
    end
end

@inline @inbounds function updatexᵢⱼ!(xᵢⱼˣ, xᵢⱼʸ, xᵢⱼᶻ, I, J, Positionˣ, Positionʸ, Positionᶻ)
    @tturbo for iter ∈ eachindex(I,J)
        i = I[iter]; j = J[iter]; 
        
        xᵢⱼˣ[iter] = Positionˣ[i] - Positionˣ[j]
        xᵢⱼʸ[iter] = Positionʸ[i] - Positionʸ[j]
        xᵢⱼᶻ[iter] = Positionᶻ[i] - Positionᶻ[j]
    end
end
# Another implementation
# function LimitDensityAtBoundary!(Density, BoundaryBool, ρ₀)
#     # Element-wise operation to set Density values
#     Density .= max.(Density, ρ₀ .* BoundaryBool)
# end


end