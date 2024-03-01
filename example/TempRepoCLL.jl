using BenchmarkTools
import CellListMap
using StaticArrays
using Parameters
using Plots; using Measures
using StructArrays
import LinearAlgebra: dot
include("../src/ProduceVTP.jl")

@with_kw struct CLL{I,T,D}
    Points::StructVector{SVector{D,T}}
    MaxValidIndex::Base.RefValue{Int} = Ref(0)
    CutOff::T
    CutOffSquared::T                         = CutOff^2
    Padding::I                               = 2
    HalfPad::I                               = convert(typeof(Padding),Padding//2)
    ZeroOffset::I                            = 1 #Since we start from 0 when generating cells

    Stencil::Vector{NTuple{D, I}}            = neighbors(Val(getsvecD(eltype(Points))) )
    
    Cells::Vector{NTuple{D, I}}              = ExtractCells(Points,CutOff,Val(getsvecD(eltype(Points))))
    UniqueCells::Vector{NTuple{D, I}}        = unique(Cells)
    Nmax::I                                  = maximum(reinterpret(Int,@view(Cells[:]))) + ZeroOffset
    Layout::Array{Vector{I}, D}              = GenerateM(Nmax,ZeroOffset,HalfPad,Padding,Cells,Val(getsvecD(eltype(Points))))

    
    ListOfInteractions::Vector{Tuple{I,I,T}} = Vector{Tuple{Int,Int,getsvecT(eltype(Points))}}(undef,CalculateTotalPossibleNumberOfInteractions(UniqueCells,Layout,Stencil,HalfPad))
end
@inline getspecs(::Type{CLL{I,T,D}}) where {I,T,D} = (typeINT = I, typeFLT = T, dimensions=D)
@inline getsvecDT(::Type{SVector{d,T}}) where {d,T} = (dimensions=d, type=T)
@inline getsvecD(::Type{SVector{d,T}}) where {d,T} = d
@inline getsvecT(::Type{SVector{d,T}}) where {d,T} = T

@inline function distance_condition(p1::AbstractVector{T}, p2::AbstractVector{T}) where T <: AbstractFloat
    d2 = sum(@. (p1 - p2)^2)
    return d2
end

# https://jaantollander.com/post/searching-for-fixed-radius-near-neighbors-with-cell-lists-algorithm-in-julia-language/#definition
function neighbors(v::Val{d}) where d
    n_ = CartesianIndices(ntuple(_->-1:1,v))
    half_length = length(n_) ÷ 2
    n  = n_[1:half_length]
    
    n_svec = Vector{NTuple{d,Int}}(undef,length(n)) #zeros(SVector{d,eltype(d)},length(n))

    for i ∈ eachindex(n_svec)
        val       = n[i]
        n_svec[i] = (val.I)
    end

    return n_svec
end


function ExtractCells(p,R,::Val{d}) where d
    n = length(p)
    cells = Vector{NTuple{d,Int}}(undef,n)

    for i = 1:n
        vs = Int.(fld.(p[i],R))
        cells[i] = tuple(vs...)
    end

    return cells
end

function GenerateM(Nmax,ZeroOffset,HalfPad,Padding,cells,v::Val{d}) where d
    Msize = ntuple(_ -> Nmax+Padding,v)
    M     = Array{Vector{Int}}(undef,Msize)

    #sizehint! is a genius function
    # but it actually does not improve performance anymore lol
    for i = 1:prod(size(M))
        arr  = Vector{Int}()
        #sizehint!(arr,100)
        @inbounds M[i] = arr
    end

    iter = 0

    for ind ∈ cells
        iter += 1
        @inbounds push!(M[(ind .+ ZeroOffset .+ HalfPad)...],iter)
    end

    return M
end

function CalculateTotalPossibleNumberOfInteractions(UniqueCells,Layout,Stencil,HalfPad)
    # We use the same loop as in the actual algorithm for now..
    # In future try to simplify like this not working exactly..
      # M = TheCLL.Layout
    # S = TheCLL.Stencil
    # RealNL = 0
    # @inbounds for Cind_ ∈ TheCLL.UniqueCells
    #     Cind = (Cind_ .+ 1 .+ TheCLL.HalfPad)

    #     NumberOfParticlesInCell  = length(TheCLL.Layout[Cind...])
    #     for Sind_ ∈ TheCLL.Stencil
    #         Sind = (Cind .+ Sind_)
    #         RealNL += NumberOfParticlesInCell * length(TheCLL.Layout[Sind...])
    #         RealNL += length(TheCLL.Layout[Sind...])
    #     end
    # end

    RealNL = 0

    @inbounds for Cind_ ∈ UniqueCells
            
        Cind = (Cind_ .+ 1 .+ HalfPad)

        # The indices in the cell are:
        indices_in_cell = Layout[Cind...]
        n_idx_cells = length(indices_in_cell)
        for ki = 1:n_idx_cells-1
            k_idx = indices_in_cell[ki]
              for kj = (ki+1):n_idx_cells
                k_1up = indices_in_cell[kj]
                RealNL += 1
            end
        end

        for Sind ∈ Stencil
            Sind = (Cind .+ Sind)
            indices_in_cell_plus  =Layout[Sind...]
            # Here a double loop to compare indices_in_cell[k] to all possible neighbours
            for k1 ∈ eachindex(indices_in_cell)
                k1_idx = indices_in_cell[k1]
                for k2 ∈ eachindex(indices_in_cell_plus)
                    k2_idx = indices_in_cell_plus[k2]
                    RealNL += 1
                end
            end
        end
    end

    return RealNL
end


function sim_step(i,j,d2,h⁻¹,Kernel,KernelGradient, Position,Density,Velocity)
    d  = sqrt(d2)

    xᵢ  = Position[i]
    xⱼ  = Position[j]
    xᵢⱼ = xᵢ - xⱼ

    q  = clamp(d  * h⁻¹,0.0,2.0)
    W  = αD*(1-q/2)^4*(2*q + 1)

    Kernel[i] += W
    Kernel[j] += W

    Fac = αD*5*(q-2)^3*q / (8h*(q*h+1e-6))
    ∇ᵢWᵢⱼ = Fac * xᵢⱼ
    KernelGradient[i] +=  ∇ᵢWᵢⱼ
    KernelGradient[j] += -∇ᵢWᵢⱼ

    d² = d*d

    ρᵢ    = Density[i]
    ρⱼ    = Density[j]


    vᵢ      = Velocity[i]
    vⱼ      = Velocity[j]
    vᵢⱼ     = vᵢ - vⱼ

    dρdt⁺   = dot(m₀ *   vᵢⱼ ,  ∇ᵢWᵢⱼ)
    dρdt⁻   = dot(m₀ *  -vᵢⱼ , -∇ᵢWᵢⱼ)

    Pᵢ      = EquationOfState(ρᵢ,c₀,γ,ρ₀)
    Pⱼ      = EquationOfState(ρⱼ,c₀,γ,ρ₀)

    ρ̄ᵢⱼ     = (ρᵢ+ρⱼ)*0.5
    Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)

    cond      = dot(vᵢⱼ, xᵢⱼ)
    cond_bool = cond < 0.0
    μᵢⱼ       = h*cond/(d²+η²)
    Πᵢⱼ       = cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ

    dvdt⁺ = - m₀ * ( Pfac + Πᵢⱼ) *  ∇ᵢWᵢⱼ + SVector(0, g * GravityFactor[i])
    dvdt⁻ = - m₀ * ( Pfac + Πᵢⱼ) * -∇ᵢWᵢⱼ + SVector(0, g * GravityFactor[j])
    
    # Time stepping start - this makes suction
    DensityNew[i]  += dρdt⁺ * (dt/2) #Starts as  ρᵢ and becomes  ρᵢᴺ
    DensityNew[j]  += dρdt⁻ * (dt/2) #Starts as  ρⱼ and becomes  ρⱼᴺ

    if BoundaryBool[i]
        DensityNew[i] = clamp(DensityNew[i],1000.0,2000.0)
    elseif BoundaryBool[j]
        DensityNew[j] = clamp(DensityNew[j],1000.0,2000.0)
    end


    vᵢᴺ  = vᵢ + dvdt⁺ * (dt/2) * MotionLimiter[i]
    vⱼᴺ  = vⱼ + dvdt⁻ * (dt/2) * MotionLimiter[j]

    xᵢᴺ  = xᵢ + vᵢᴺ * (dt/2) * MotionLimiter[i]
    xⱼᴺ  = xⱼ + vⱼᴺ * (dt/2) * MotionLimiter[j]

    # Update and go through same steps as above

    xᵢⱼᴺ = xᵢᴺ - xⱼᴺ

    vᵢⱼᴺ     = vᵢᴺ - vⱼᴺ

    dρdtᴺ⁺   = dot(m₀ *   vᵢⱼᴺ , ∇ᵢWᵢⱼ)
    dρdtᴺ⁻   = dot(m₀ *  -vᵢⱼᴺ ,-∇ᵢWᵢⱼ)

    DensityDerivativeHalfStep[k_idx]  += dρdtᴺ⁺
    DensityDerivativeHalfStep[k_1up]  += dρdtᴺ⁻

    Pᵢᴺ     = EquationOfState(DensityNew[i],c₀,γ,ρ₀)
    Pⱼᴺ     = EquationOfState(DensityNew[j],c₀,γ,ρ₀)

    ρ̄ᵢⱼᴺ     = (DensityNew[i]+DensityNew[j])*0.5
    Pfacᴺ    = (Pᵢᴺ+Pⱼᴺ)/(DensityNew[i]*DensityNew[j])

    cond      = dot(vᵢⱼᴺ, xᵢⱼᴺ)
    cond_bool = cond < 0.0
    μᵢⱼᴺ       = h*cond/(d²+η²)
    Πᵢⱼᴺ       = cond_bool*(-α*c₀*μᵢⱼᴺ)/ρ̄ᵢⱼᴺ

    dvdtᴺ⁺ = - m₀ * ( Pfacᴺ + Πᵢⱼᴺ) *  ∇ᵢWᵢⱼ + SVector(0, g * GravityFactor[i])
    dvdtᴺ⁻ = - m₀ * ( Pfacᴺ + Πᵢⱼᴺ) * -∇ᵢWᵢⱼ + SVector(0, g * GravityFactor[j])

    VelocityNew[i] += dvdtᴺ⁺ * dt * MotionLimiter[i]
    VelocityNew[j] += dvdtᴺ⁻ * dt * MotionLimiter[j]

    PositionNew[i] += ((VelocityNew[i] - vᵢ)/2) * dt * MotionLimiter[i]
    PositionNew[j] += ((VelocityNew[j] - vⱼ)/2) * dt * MotionLimiter[j]
end

using SPHExample 
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
@inline faux_fancy(ρ₀, P, Cb) = ρ₀ * ( fancy7th( 1 + (P * Cb)) - 1)

function EquationOfState(ρ,c₀,γ,ρ₀)
    return ((c₀^2*ρ₀)/γ) * ((ρ/ρ₀)^γ - 1)
end

function CustomCLL(PositionNew, DensityNew, VelocityNew, SimConstants, MotionLimiter, BoundaryBool, GravityFactor, Position, Kernel, KernelGradient, Density, Velocity, DensityHalfStep, DensityDerivativeHalfStep)
    @unpack ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η² = SimConstants


    Kernel                    .*= 0
    KernelGradient.V          .*= 0
    DensityDerivativeHalfStep .*= 0

    dt = 7.65e-5

    Cb      = (c₀^2*ρ₀)/γ
    invCb   = inv(Cb)
    δₕ_h_c₀ = δᵩ * h * c₀

    R = 2*h
    TheCLL = CLL(Points=Position.V,CutOff=R)

    nl    = 0

    @inbounds for Cind_ ∈ TheCLL.UniqueCells
            
        Cind = (Cind_ .+ 1 .+ TheCLL.HalfPad)

            # The indices in the cell are:
            indices_in_cell = TheCLL.Layout[Cind...]

            n_idx_cells = length(indices_in_cell)
            for ki = 1:n_idx_cells-1
                k_idx = indices_in_cell[ki]
                  for kj = (ki+1):n_idx_cells
                    k_1up = indices_in_cell[kj]
                    d2 = distance_condition(Position.V[k_idx],Position.V[k_1up])

                    if d2 <= TheCLL.CutOffSquared
                        d  = sqrt(d2)

                        xᵢ  = Position.V[k_idx]
                        xⱼ  = Position.V[k_1up]
                        xᵢⱼ = xᵢ - xⱼ

                        q  = clamp(d  * h⁻¹,0.0,2.0)
                        W  = αD*(1-q/2)^4*(2*q + 1)
                        Kernel[k_idx] += W
                        Kernel[k_1up] += W

                        Fac = αD*5*(q-2)^3*q / (8h*(q*h+1e-6))
                        ∇ᵢWᵢⱼ = Fac * xᵢⱼ
                        KernelGradient.V[k_idx] +=  ∇ᵢWᵢⱼ
                        KernelGradient.V[k_1up] += -∇ᵢWᵢⱼ

                        d² = d*d

                        ρᵢ    = Density[k_idx]
                        ρⱼ    = Density[k_1up]
   

                        vᵢ      = Velocity.V[k_idx]
                        vⱼ      = Velocity.V[k_1up]
                        vᵢⱼ     = vᵢ - vⱼ

                        dρdt⁺   = dot(m₀ *   vᵢⱼ ,  ∇ᵢWᵢⱼ)
                        dρdt⁻   = dot(m₀ *  -vᵢⱼ , -∇ᵢWᵢⱼ)

                        Pᵢ      = EquationOfState(ρᵢ,c₀,γ,ρ₀)
                        Pⱼ      = EquationOfState(ρⱼ,c₀,γ,ρ₀)

                        ρ̄ᵢⱼ     = (ρᵢ+ρⱼ)*0.5
                        Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)

                        cond      = dot(vᵢⱼ, xᵢⱼ)
                        cond_bool = cond < 0.0
                        μᵢⱼ       = h*cond/(d²+η²)
                        Πᵢⱼ       = cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ

                        dvdt⁺ = - m₀ * ( Pfac + Πᵢⱼ) *  ∇ᵢWᵢⱼ + SVector(0, g * GravityFactor[k_idx])
                        dvdt⁻ = - m₀ * ( Pfac + Πᵢⱼ) * -∇ᵢWᵢⱼ + SVector(0, g * GravityFactor[k_1up])
                        
                        # Time stepping start - this makes suction
                        DensityNew[k_idx]  += dρdt⁺ * (dt/2) #Starts as  ρᵢ and becomes  ρᵢᴺ
                        DensityNew[k_1up]  += dρdt⁻ * (dt/2) #Starts as  ρⱼ and becomes  ρⱼᴺ

                        if BoundaryBool[k_idx]
                            DensityNew[k_idx] = clamp(DensityNew[k_idx],1000.0,2000.0)
                        elseif BoundaryBool[k_1up]
                            DensityNew[k_1up] = clamp(DensityNew[k_1up],1000.0,2000.0)
                        end


                        vᵢᴺ  = vᵢ + dvdt⁺ * (dt/2) * MotionLimiter[k_idx]
                        vⱼᴺ  = vⱼ + dvdt⁻ * (dt/2) * MotionLimiter[k_1up]

                        xᵢᴺ  = xᵢ + vᵢᴺ * (dt/2) * MotionLimiter[k_idx]
                        xⱼᴺ  = xⱼ + vⱼᴺ * (dt/2) * MotionLimiter[k_1up]

                        # Update and go through same steps as above

                        xᵢⱼᴺ = xᵢᴺ - xⱼᴺ

                        vᵢⱼᴺ     = vᵢᴺ - vⱼᴺ

                        dρdtᴺ⁺   = dot(m₀ *   vᵢⱼᴺ , ∇ᵢWᵢⱼ)
                        dρdtᴺ⁻   = dot(m₀ *  -vᵢⱼᴺ ,-∇ᵢWᵢⱼ)

                        DensityDerivativeHalfStep[k_idx]  += dρdtᴺ⁺
                        DensityDerivativeHalfStep[k_1up]  += dρdtᴺ⁻

                        Pᵢᴺ     = EquationOfState(DensityNew[k_idx],c₀,γ,ρ₀)
                        Pⱼᴺ     = EquationOfState(DensityNew[k_1up],c₀,γ,ρ₀)

                        ρ̄ᵢⱼᴺ     = (DensityNew[k_idx]+DensityNew[k_1up])*0.5
                        Pfacᴺ    = (Pᵢᴺ+Pⱼᴺ)/(DensityNew[k_idx]*DensityNew[k_1up])

                        cond      = dot(vᵢⱼᴺ, xᵢⱼᴺ)
                        cond_bool = cond < 0.0
                        μᵢⱼᴺ       = h*cond/(d²+η²)
                        Πᵢⱼᴺ       = cond_bool*(-α*c₀*μᵢⱼᴺ)/ρ̄ᵢⱼᴺ

                        dvdtᴺ⁺ = - m₀ * ( Pfacᴺ + Πᵢⱼᴺ) *  ∇ᵢWᵢⱼ + SVector(0, g * GravityFactor[k_idx])
                        dvdtᴺ⁻ = - m₀ * ( Pfacᴺ + Πᵢⱼᴺ) * -∇ᵢWᵢⱼ + SVector(0, g * GravityFactor[k_1up])

                        VelocityNew.V[k_idx] += dvdtᴺ⁺ * dt * MotionLimiter[k_idx]
                        VelocityNew.V[k_1up] += dvdtᴺ⁻ * dt * MotionLimiter[k_1up]

                        PositionNew.V[k_idx] += ((VelocityNew.V[k_idx] - vᵢ)/2) * dt * MotionLimiter[k_idx]
                        PositionNew.V[k_1up] += ((VelocityNew.V[k_1up] - vⱼ)/2) * dt * MotionLimiter[k_1up]
                    end
                end
            end

            for Sind ∈ TheCLL.Stencil
                Sind = (Cind .+ Sind)
                indices_in_cell_plus  = TheCLL.Layout[Sind...]

                # Here a double loop to compare indices_in_cell[k] to all possible neighbours
                for k1 ∈ eachindex(indices_in_cell)
                    k1_idx = indices_in_cell[k1]
                    for k2 ∈ eachindex(indices_in_cell_plus)
                        k2_idx = indices_in_cell_plus[k2]
                        d2  = distance_condition(Position.V[k1_idx],Position.V[k2_idx])

                        if d2 <= TheCLL.CutOffSquared
                            d   = sqrt(d2)
                            d²  = d*d

                            xᵢ  = Position.V[k1_idx]
                            xⱼ  = Position.V[k2_idx]
                            xᵢⱼ = xᵢ - xⱼ

                            q   = clamp(d  * h⁻¹,0.0,2.0)
                            W   = αD*(1-q/2)^4*(2*q + 1)
                            Kernel[k1_idx] += W 
                            Kernel[k2_idx] += W 

                            Fac = αD*5*(q-2)^3*q / (8h*(q*h+1e-6))
                            ∇ᵢWᵢⱼ = Fac * xᵢⱼ
                            KernelGradient.V[k1_idx] +=  ∇ᵢWᵢⱼ
                            KernelGradient.V[k2_idx] += -∇ᵢWᵢⱼ
        
                            ρᵢ    = Density[k1_idx]
                            ρⱼ    = Density[k2_idx]
        
                            vᵢ      = Velocity.V[k1_idx]
                            vⱼ      = Velocity.V[k2_idx]
                            vᵢⱼ     = vᵢ - vⱼ
        
                            dρdt⁺   = dot(m₀ *   vᵢⱼ ,  ∇ᵢWᵢⱼ) 
                            dρdt⁻   = dot(m₀ *  -vᵢⱼ , -∇ᵢWᵢⱼ) 
        
                            Pᵢ      = EquationOfState(ρᵢ,c₀,γ,ρ₀)
                            Pⱼ      = EquationOfState(ρⱼ,c₀,γ,ρ₀)

                            ρ̄ᵢⱼ     = (ρᵢ+ρⱼ)*0.5
                            Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
        
                            cond      = dot(vᵢⱼ, xᵢⱼ)
                            cond_bool = cond < 0.0
                            μᵢⱼ       = h*cond/(d²+η²)
                            Πᵢⱼ       = cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ
        
                            dvdt⁺ = - m₀ * ( Pfac + Πᵢⱼ) *  ∇ᵢWᵢⱼ + SVector(0, g * GravityFactor[k1_idx])
                            dvdt⁻ = - m₀ * ( Pfac + Πᵢⱼ) * -∇ᵢWᵢⱼ + SVector(0, g * GravityFactor[k2_idx]) 

                            # Time stepping start
                            DensityNew[k1_idx]  += dρdt⁺ * (dt/2) #Starts as  ρᵢ and becomes  ρᵢᴺ
                            DensityNew[k2_idx]  += dρdt⁻ * (dt/2) #Starts as  ρⱼ and becomes  ρⱼᴺ

                            if BoundaryBool[k1_idx]
                                DensityNew[k1_idx] = clamp(DensityNew[k1_idx],1000,2000)
                            elseif BoundaryBool[k2_idx]
                                DensityNew[k2_idx] = clamp(DensityNew[k2_idx],1000,2000)
                            end

                            vᵢᴺ  = vᵢ + dvdt⁺ * (dt/2) * MotionLimiter[k1_idx]
                            vⱼᴺ  = vⱼ + dvdt⁻ * (dt/2) * MotionLimiter[k2_idx]

                            xᵢᴺ  = xᵢ + vᵢᴺ   * (dt/2) * MotionLimiter[k1_idx]
                            xⱼᴺ  = xⱼ + vⱼᴺ   * (dt/2) * MotionLimiter[k2_idx]

                            # Update and go through motion again, copy paste from above
                            # d² not updated!, neiter gradient

                            xᵢⱼᴺ = xᵢᴺ - xⱼᴺ
        
                            vᵢⱼᴺ     = vᵢᴺ - vⱼᴺ
        
                            dρdtᴺ⁺   = dot(m₀ *   vᵢⱼᴺ ,  ∇ᵢWᵢⱼ)
                            dρdtᴺ⁻   = dot(m₀ *  -vᵢⱼᴺ , -∇ᵢWᵢⱼ)
        
                            DensityDerivativeHalfStep[k1_idx]   += dρdtᴺ⁺
                            DensityDerivativeHalfStep[k2_idx]   += dρdtᴺ⁻    

                            Pᵢᴺ     = EquationOfState(DensityNew[k1_idx],c₀,γ,ρ₀)
                            Pⱼᴺ     = EquationOfState(DensityNew[k2_idx],c₀,γ,ρ₀)

                            ρ̄ᵢⱼᴺ     = (DensityNew[k1_idx]+DensityNew[k2_idx])*0.5
                            Pfacᴺ    = (Pᵢᴺ+Pⱼᴺ)/(DensityNew[k1_idx]*DensityNew[k2_idx])
        
                            cond      = dot(vᵢⱼᴺ, xᵢⱼᴺ)
                            cond_bool = cond < 0.0
                            μᵢⱼᴺ       = h*cond/(d²+η²)
                            Πᵢⱼᴺ       = cond_bool*(-α*c₀*μᵢⱼᴺ)/ρ̄ᵢⱼᴺ
        
                            dvdtᴺ⁺ = - m₀ * ( Pfacᴺ + Πᵢⱼᴺ) *  ∇ᵢWᵢⱼ + SVector(0, g * GravityFactor[k1_idx])
                            dvdtᴺ⁻ = - m₀ * ( Pfacᴺ + Πᵢⱼᴺ) * -∇ᵢWᵢⱼ + SVector(0, g * GravityFactor[k2_idx])


                            VelocityNew.V[k1_idx] += dvdtᴺ⁺ * dt * MotionLimiter[k1_idx]
                            VelocityNew.V[k2_idx] += dvdtᴺ⁻ * dt * MotionLimiter[k2_idx]

                            PositionNew.V[k1_idx] += ((VelocityNew.V[k1_idx] - vᵢ)/2) * dt * MotionLimiter[k1_idx]
                            PositionNew.V[k2_idx] += ((VelocityNew.V[k2_idx] - vⱼ)/2) * dt * MotionLimiter[k2_idx]
                        end
                    end
                end
            end
    end

    Position.V .= PositionNew.V
    Velocity.V .= VelocityNew.V
    # Density    .= DensityNew

    DensityEpsi!(Density, DensityDerivativeHalfStep, DensityNew, dt)

    println(sum(Kernel))
    println(sum(KernelGradient.V))
    # println(sum(Density))
    

    TheCLL.MaxValidIndex[] = nl
    # resize!(TheCLL.ListOfInteractions,nl)

    return nothing
end

to_3d(vec_2d) = [SVector(v..., 0.0) for v in vec_2d]

# For testing script properly
let 
    T = Float64
    FloatType = T
    D = 2
    Dimensions = D
    
    FluidCSV     = "./input/DSPH_DamBreak_Fluid_Dp0.02.csv"
    BoundCSV     = "./input/DSPH_DamBreak_Boundary_Dp0.02.csv"
    
    SimMetaData  = SimulationMetaData{D, T}(
                                    SimulationName="AllInOne", 
                                    SaveLocation=raw"E:\SecondApproach\Testing",
                                    # MaxIterations=10001,
                                    # OutputIteration=50,
    )
    # Initialze the constants to use
    SimConstants = SimulationConstants{T}()
    
    # Unpack the relevant simulation meta data
    @unpack HourGlass, SaveLocation, SimulationName, SilentOutput, ThreadsCPU = SimMetaData;
    
    # Unpack simulation constants
    @unpack ρ₀, dx, h, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η² = SimConstants
    
    # Load in the fluid and boundary particles. Return these points and both data frames
    # @inline is a hack here to remove all allocations further down due to uncertainty of the points type at compile time
    @inline points, density_fluid, density_bound  = LoadParticlesFromCSV(Dimensions,FloatType, FluidCSV,BoundCSV)
    Position           = DimensionalData(points.vectors...)
    # Read this as "GravityFactor * g", so -1 means negative acceleration for fluid particles
    GravityFactor            = [-ones(size(density_fluid,1)) ; zeros(size(density_bound,1))]
    
    # MotionLimiter is what allows fluid particles to move, while not letting the velocity of boundary
    # particles change
    MotionLimiter = [ ones(size(density_fluid,1)) ; zeros(size(density_bound,1))]
    
    # Based on MotionLimiter we assess which particles are boundary particles
    BoundaryBool  = .!Bool.(MotionLimiter)
    
    # Preallocate simulation arrays
    NumberOfPoints = length(points)
    
    Density           = deepcopy([density_fluid;density_bound])


    DensityHalfStep           = deepcopy(Density)
    DensityDerivativeHalfStep = zeros(FloatType, NumberOfPoints)

    Kernel            = zeros(FloatType, NumberOfPoints)
    KernelL           = zeros(FloatType, NumberOfPoints)
    dρdtI             = zeros(FloatType, NumberOfPoints)
    ρₙ⁺               = zeros(FloatType, NumberOfPoints)
    dρdtIₙ⁺           = zeros(FloatType, NumberOfPoints)
    
    drhopLp            = zeros(FloatType, NumberOfPoints)
    drhopLn            = zeros(FloatType, NumberOfPoints) 
    Pressureᵢ          = zeros(FloatType, NumberOfPoints)
    
    KernelGradient     = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
    Velocity           = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
    dvdtI              = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
    
    DensityNew         = deepcopy(Density)
    VelocityNew        = deepcopy(Velocity)
    PositionNew        = deepcopy(Position)
   
    CustomCLL(PositionNew, DensityNew, VelocityNew, SimConstants, MotionLimiter, BoundaryBool, GravityFactor, Position, Kernel, KernelGradient, Density, Velocity,  DensityHalfStep,  DensityDerivativeHalfStep)
    function f()
        foreach(rm, filter(endswith(".vtp"), readdir(SimMetaData.SaveLocation,join=true)))
        PolyDataTemplate(SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(0,6,"0") * ".vtp", to_3d(Position.V), ["Kernel","KernelGradient","Density","Velocity"], Kernel, KernelGradient.V, Density, Velocity.V)
        for iteration in 1:51
            CustomCLL(PositionNew, DensityNew, VelocityNew, SimConstants, MotionLimiter, BoundaryBool, GravityFactor, Position, Kernel, KernelGradient, Density, Velocity, DensityHalfStep,  DensityDerivativeHalfStep)
            if iteration % 1 == 0
                PolyDataTemplate(SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(iteration,6,"0") * ".vtp", to_3d(Position.V), ["Kernel","KernelGradient","Density","Velocity"], Kernel, KernelGradient.V, Density, Velocity.V)
                println(iteration)
            end
        end
    end

    @profview f()

    # @benchmark CustomCLL($PositionNew, $DensityNew, $VelocityNew, $SimConstants, $MotionLimiter, $BoundaryBool, $GravityFactor, $Position, $Kernel, $KernelGradient, $Density, $Velocity)
end



