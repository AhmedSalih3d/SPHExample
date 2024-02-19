module SimulationEquations

export Wᵢⱼ, ∑ⱼWᵢⱼ!, Optim∇ᵢWᵢⱼ, ∑ⱼWᵢⱼ!∑ⱼ∇ᵢWᵢⱼ!, EquationOfState, Pressure!, ∂Πᵢⱼ∂t!, ∂ρᵢ∂tDDT!, ∂vᵢ∂t!, DensityEpsi!, LimitDensityAtBoundary!, updatexᵢⱼ!, ArtificialViscosityMomentumEquation!, DimensionalData

using CellListMap
using StaticArrays
import LinearAlgebra: sqrt, norm #Don't import to not get LinearAlgebra.I in scope and bug out
using Parameters
using LoopVectorization
using Bumper
using Polyester
using Base.Threads
using StructArrays
using StaticArrays
using ChunkSplitters

struct DimensionalData{D, T <: AbstractFloat}
    vectors::Tuple{Vararg{Vector{T}, D}}
    V::StructArray{SVector{D, T}, 1, Tuple{Vararg{Vector{T}, D}}}

    # General constructor for vectors
    function DimensionalData(vectors::Vector{T}...) where {T}
        D = length(vectors)
        V = StructArray{SVector{D, T}}(vectors)
        new{D, T}(Tuple(vectors), V)
    end

    # Constructor for initializing with all zeros, adapting to dimension D
    function DimensionalData{D, T}(len::Int) where {D, T}
        vectors = ntuple(d -> zeros(T, len), D) # Create D vectors of zeros
        V = StructArray{SVector{D, T}}(vectors)
        new{D, T}(vectors, V)
    end
end

# Overwrite resizing and fill functions for DimensionalData
Base.resize!(data::DimensionalData,n::Int) = resize!(data.V,n) 
reset!(data::DimensionalData)              = fill!(data.V,zero(eltype(data.V)))
Base.length(data::DimensionalData)         = length(data.V)

# Threaded reduction instead of single thread.
# Variables are taken as examples
ReductionFunctionChunk!(dρdtI, I, J, drhopLp) = ReductionFunctionChunk!(dρdtI, I, J, drhopLp, drhopLp)
function ReductionFunctionChunk!(dρdtI, I, J, drhopLp, drhopLn, op1=+, op2=+)
    XT = eltype(dρdtI); XL = length(dρdtI); X0 = zero(XT)
    nchunks = nthreads() 
    
    @inbounds @no_escape begin
        local_X = @alloc(XT, XL, nchunks)

        fill!(local_X,X0)

        # Directly iterate over the chunks
        @batch for ichunk in 1:nchunks
            chunk_inds = getchunk(I, ichunk; n=nchunks)
            for idx in chunk_inds
                i = I[idx]
                j = J[idx]

                # Accumulate the contributions into the correct place
                local_X[i, ichunk] = local_X[i, ichunk] + op1(drhopLp[idx])
                local_X[j, ichunk] = local_X[j, ichunk] + op2(drhopLn[idx])
            end
        end

        # Reduction step
        @tturbo for ix in 1:XL
            for chunk in 1:nchunks
                dρdtI[ix] += local_X[ix, chunk]
            end
        end
    end
    
    # # Reduction - Single threaded
    # @inbounds for iter in eachindex(I,J)
    #     i = I[iter]
    #     j = J[iter]

    #     dρdtI[i] +=  drhopLp[iter]
    #     dρdtI[j] +=  drhopLn[iter]
    # end

    return nothing
end

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
@generated function ∑ⱼWᵢⱼ!∑ⱼ∇ᵢWᵢⱼ!(KernelGradientI::DimensionalData{dims}, KernelGradientL::DimensionalData, Kernel, KernelL, I, J, D, xᵢⱼ::DimensionalData, SimulationConstants) where {dims}
    quote
        @unpack αD, h, h⁻¹, η² = SimulationConstants
    
        @tturbo for iter in eachindex(I)
            i = I[iter]; j = J[iter]; d = D[iter]

            q = clamp(d * h⁻¹, 0.0, 2.0)
            Fac = αD*5*(q-2)^3*q / (8h*(q*h+η²)) 
    
            W = Wᵢⱼ(αD,q)
    
            KernelL[iter] = W

            Base.Cartesian.@nexprs $dims dᵅ -> begin 
                KernelGradientL.vectors[dᵅ][iter] = Fac * xᵢⱼ.vectors[dᵅ][iter]
            end
        end

        # Reducing kernel values
        ReductionFunctionChunk!(Kernel,I,J,KernelL)
        Base.Cartesian.@nexprs $dims dᵅ -> begin
            ReductionFunctionChunk!(KernelGradientI.vectors[dᵅ],I,J,KernelGradientL.vectors[dᵅ], KernelGradientL.vectors[dᵅ], +, -)
        end

        # for iter in eachindex(I,J)
        #     i = I[iter]
        #     j = J[iter]
    
        #     Base.Cartesian.@nexprs $dims dᵅ -> begin
        #         KernelGradientI.vectors[dᵅ][i]   +=  KernelGradientL.vectors[dᵅ][iter]
        #         KernelGradientI.vectors[dᵅ][j]   -=  KernelGradientL.vectors[dᵅ][iter]
        #     end
        # end

        return nothing
    end
end

# Equation of State in Weakly-Compressible SPH
function EquationOfState(ρ,c₀,γ,ρ₀)
    return ((c₀^2*ρ₀)/γ) * ((ρ/ρ₀)^γ - 1)
end

@inline @inbounds function Pressure!(Press, Density, SimulationConstants)
    @unpack c₀,γ,ρ₀ = SimulationConstants
    @tturbo for i ∈ eachindex(Press,Density)
        Press[i] = EquationOfState(Density[i],c₀,γ,ρ₀)
    end
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
@generated function ∂ρᵢ∂tDDT!(dρdtI, I, J, D , xᵢⱼ::DimensionalData{dims} , Density , Velocity::DimensionalData, KernelGradientL::DimensionalData, MotionLimiter, drhopLp, drhopLn, SimulationConstants) where {dims}
    quote
        @unpack h,m₀,δᵩ,c₀,γ,g,ρ₀,η²,γ⁻¹ = SimulationConstants

        # Generate the needed constants
        Cb      = (c₀^2*ρ₀)/γ
        invCb   = inv(Cb)
        δₕ_h_c₀ = δᵩ * h * c₀


        # Follow the implementation here: https://arxiv.org/abs/2110.10076
        @tturbo for iter in eachindex(I,J,D)
            i = I[iter]; j = J[iter]; d = D[iter]
            
                Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ.vectors[dims][iter]
                ρᵢⱼᴴ  = faux_fancy(ρ₀, Pᵢⱼᴴ, invCb)
                Pⱼᵢᴴ  = -Pᵢⱼᴴ
                ρⱼᵢᴴ  = faux_fancy(ρ₀, Pⱼᵢᴴ, invCb)

                r²    = d*d
                ρᵢ    = Density[i]
                ρⱼ    = Density[j]
                ρⱼᵢ   = ρⱼ - ρᵢ

                FacRhoI = 2 * ( ρⱼᵢ - ρᵢⱼᴴ) * inv(r²+η²)
                FacRhoJ = 2 * (-ρⱼᵢ - ρⱼᵢᴴ) * inv(r²+η²)

            Base.Cartesian.@nexprs $dims dᵅ -> begin
                drhopLp[iter] += (m₀ * (  Velocity.vectors[dᵅ][i] - Velocity.vectors[dᵅ][j])  + δₕ_h_c₀ * (m₀/ρⱼ) * FacRhoI *  -xᵢⱼ.vectors[dᵅ][iter] * MotionLimiter[i]) *  KernelGradientL.vectors[dᵅ][iter]
                drhopLn[iter] += (m₀ * (-(Velocity.vectors[dᵅ][i] - Velocity.vectors[dᵅ][j])) + δₕ_h_c₀ * (m₀/ρᵢ) * FacRhoJ *   xᵢⱼ.vectors[dᵅ][iter] * MotionLimiter[j]) * -KernelGradientL.vectors[dᵅ][iter]
            end
        end

        ReductionFunctionChunk!(dρdtI, I, J, drhopLp, drhopLn)

        return nothing
    end
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

# The momentum equation without any dissipation - we add the dissipation using artificial viscosity (∂Πᵢⱼ∂t)
@generated function ArtificialViscosityMomentumEquation!(I,J, D, dvdtI::DimensionalData, dvdtL::DimensionalData,Density,KernelGradientL::DimensionalData, xᵢⱼ::DimensionalData{dims}, Velocity::DimensionalData, Press, GravityFactor, SimulationConstants) where {dims}
    quote
        @unpack m₀, c₀,γ,ρ₀,α,h,η²,g = SimulationConstants
        # Calculation
        @tturbo for iter in eachindex(I)
            i = I[iter]; j = J[iter]; d = D[iter]
            ρᵢ    = Density[i]
            ρⱼ    = Density[j]
            Pᵢ    = Press[i] #Pᵢ    = Pressure(ρᵢ,c₀,γ,ρ₀)
            Pⱼ    = Press[j] #Pⱼ    = Pressure(ρⱼ,c₀,γ,ρ₀)
            ρ̄ᵢⱼ   = (ρᵢ+ρⱼ)*0.5
            Pfac  = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
            d²    = d*d

            Base.Cartesian.@nexprs $dims dᵅ -> begin
                ∇ᵢWᵢⱼᵈ =  KernelGradientL.vectors[dᵅ][iter]
                vᵢⱼᵈ      = Velocity.vectors[dᵅ][i] - Velocity.vectors[dᵅ][j]
                xᵢⱼᵈ      = xᵢⱼ.vectors[dᵅ][iter]
                cond      = vᵢⱼᵈ * xᵢⱼᵈ
                cond_bool = cond < 0.0
                μᵢⱼᵈ      = h*cond/(d²+η²)
                Πᵢⱼᵈ      = cond_bool*(-α*c₀*μᵢⱼᵈ)/ρ̄ᵢⱼ
                # Finally combine contributions
                dvdtL.vectors[dᵅ][iter] = - m₀ * ( Pfac + Πᵢⱼᵈ) * ∇ᵢWᵢⱼᵈ
            end
        end

        Base.Cartesian.@nexprs $dims dᵅ -> begin
            ReductionFunctionChunk!(dvdtI.vectors[dᵅ],I,J,dvdtL.vectors[dᵅ], dvdtL.vectors[dᵅ], +, -)
        end

        # # Reduction
        # for iter in eachindex(I,J)
        #     i = I[iter]
        #     j = J[iter]

        #     Base.Cartesian.@nexprs $dims dᵅ -> begin
        #         dvdtI.vectors[dᵅ][i] += dvdtL.vectors[dᵅ][iter]
        #         dvdtI.vectors[dᵅ][j] -= dvdtL.vectors[dᵅ][iter]
        #     end
        # end

        # Add gravity to fluid particles
        @tturbo for i in eachindex(GravityFactor)
            dvdtI.vectors[dims][i] += g * GravityFactor[i]
        end

        return nothing
    end
end

# Define a generated function to dynamically create expressions based on D
@generated function updatexᵢⱼ!(xᵢⱼ::DimensionalData{dims}, Position::DimensionalData, I, J) where {dims}
    quote
        @tturbo for iter ∈ eachindex(I,J)
            i, j = I[iter], J[iter]
            Base.Cartesian.@nexprs $dims dᵅ -> begin
            xᵢⱼ.vectors[dᵅ][iter] = Position.vectors[dᵅ][i] - Position.vectors[dᵅ][j]  # Compute the difference for the d-th dimension
            end
        end
    end
end

end