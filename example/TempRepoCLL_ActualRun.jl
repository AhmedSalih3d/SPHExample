using SPHExample
import CellListMap: InPlaceNeighborList, neighborlist!, update!
using BenchmarkTools
import CellListMap
using StaticArrays
using Parameters
using Plots; using Measures
using StructArrays
import LinearAlgebra: dot
using Test
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


function sim_step(i , j, d2, SimConstants,  Kernel, KernelGradient, Position, Density, Velocity, dρdtI, dvdtI)
    @unpack h, m₀, h⁻¹,  α ,  αD, c₀, γ, ρ₀, g, η² = SimConstants
    
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

    dρdt⁺   = - ρᵢ * dot((m₀/ρⱼ) *  -vᵢⱼ ,  ∇ᵢWᵢⱼ)
    dρdt⁻   = - ρⱼ * dot((m₀/ρᵢ) *   vᵢⱼ , -∇ᵢWᵢⱼ)

    dρdtI[i] += dρdt⁺
    dρdtI[j] += dρdt⁻

    Pᵢ      = EquationOfState(ρᵢ,c₀,γ,ρ₀)
    Pⱼ      = EquationOfState(ρⱼ,c₀,γ,ρ₀)

    ρ̄ᵢⱼ     = (ρᵢ+ρⱼ)*0.5
    Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)

    cond      = dot(vᵢⱼ, xᵢⱼ)
    cond_bool = cond < 0.0
    μᵢⱼ       = h*cond/(d²+η²)
    Πᵢⱼ       = cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ

    dvdt⁺ = - m₀ * ( Pfac + Πᵢⱼ) *  ∇ᵢWᵢⱼ
    dvdt⁻ = - m₀ * ( Pfac + Πᵢⱼ) * -∇ᵢWᵢⱼ

    dvdtI[i] += dvdt⁺
    dvdtI[j] += dvdt⁻
end

function sim_step2(i , j, d2, SimConstants, Position, Density, Velocity, dρdtI, dvdtI)
    @unpack h, m₀, h⁻¹,  α ,  αD, c₀, γ, ρ₀, g, η² = SimConstants
    
    d  = sqrt(d2)

    xᵢ  = Position[i]
    xⱼ  = Position[j]
    xᵢⱼ = xᵢ - xⱼ

    q  = clamp(d  * h⁻¹,0.0,2.0)

    Fac = αD*5*(q-2)^3*q / (8h*(q*h+1e-6))
    ∇ᵢWᵢⱼ = Fac * xᵢⱼ

    d² = d*d

    ρᵢ    = Density[i]
    ρⱼ    = Density[j]

    vᵢ      = Velocity[i]
    vⱼ      = Velocity[j]
    vᵢⱼ     = vᵢ - vⱼ

    dρdt⁺   = - ρᵢ * dot((m₀/ρⱼ) *  -vᵢⱼ ,  ∇ᵢWᵢⱼ)
    dρdt⁻   = - ρⱼ * dot((m₀/ρᵢ) *   vᵢⱼ , -∇ᵢWᵢⱼ)

    dρdtI[i] += dρdt⁺
    dρdtI[j] += dρdt⁻

    Pᵢ      = EquationOfState(ρᵢ,c₀,γ,ρ₀)
    Pⱼ      = EquationOfState(ρⱼ,c₀,γ,ρ₀)

    ρ̄ᵢⱼ     = (ρᵢ+ρⱼ)*0.5
    Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)

    cond      = dot(vᵢⱼ, xᵢⱼ)
    cond_bool = cond < 0.0
    μᵢⱼ       = h*cond/(d²+η²)
    Πᵢⱼ       = cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ

    dvdt⁺ = - m₀ * ( Pfac + Πᵢⱼ) *  ∇ᵢWᵢⱼ
    dvdt⁻ = - m₀ * ( Pfac + Πᵢⱼ) * -∇ᵢWᵢⱼ

    dvdtI[i] += dvdt⁺
    dvdtI[j] += dvdt⁻
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
@inline faux_fancy(ρ₀, P, Cb) = ρ₀ * ( fancy7th( 1 + (P * Cb)) - 1)

function EquationOfState(ρ,c₀,γ,ρ₀)
    return ((c₀^2*ρ₀)/γ) * ((ρ/ρ₀)^γ - 1)
end

function CustomCLL(SimConstants, MotionLimiter, BoundaryBool, GravityFactor, Position, Kernel, KernelGradient, Density, Velocity, ρₙ⁺, Velocityₙ⁺, Positionₙ⁺, dρdtI, dρdtIₙ⁺, dvdtI, dvdtIₙ⁺)
    @unpack ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η² = SimConstants


    Kernel            .*= 0
    KernelGradient.V  .*= 0
    dρdtI             .*= 0
    dρdtIₙ⁺            .*= 0
    dvdtI.V           .*= 0
    dvdtIₙ⁺.V          .*= 0

    dt = 1e-5

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
                        nl += 1
                        sim_step(k_idx , k_1up, d2, SimConstants,  Kernel, KernelGradient.V, Position.V, Density, Velocity.V, dρdtI, dvdtI.V)
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
                            nl += 1
                            sim_step(k1_idx , k2_idx, d2, SimConstants,  Kernel, KernelGradient.V, Position.V, Density, Velocity.V, dρdtI, dvdtI.V)
                        end
                    end
                end
            end
    end


    dvdtI.vectors[end]  .+= g * GravityFactor


    @. Velocityₙ⁺.V   = Velocity.V   + dvdtI.V       * (dt/2)  * MotionLimiter
    @. Positionₙ⁺.V   = Position.V   + Velocityₙ⁺.V  * (dt/2)  * MotionLimiter
    @. ρₙ⁺            = Density    + dρdtI        * (dt/2) 
    LimitDensityAtBoundary!(ρₙ⁺,BoundaryBool,ρ₀)

    # Kernel            .*= 0
    # KernelGradient.V  .*= 0
    dρdtIₙ⁺            .*= 0
    dvdtIₙ⁺.V          .*= 0
    
    @inbounds for Cind_ ∈ TheCLL.UniqueCells
            
        Cind = (Cind_ .+ 1 .+ TheCLL.HalfPad)

            # The indices in the cell are:
            indices_in_cell = TheCLL.Layout[Cind...]

            n_idx_cells = length(indices_in_cell)
            for ki = 1:n_idx_cells-1
                k_idx = indices_in_cell[ki]
                  for kj = (ki+1):n_idx_cells
                    k_1up = indices_in_cell[kj]
                    d2 = distance_condition(Positionₙ⁺.V[k_idx],Positionₙ⁺.V[k_1up])

                    if d2 <= TheCLL.CutOffSquared
                        nl += 1
                        sim_step2(k_idx , k_1up, d2, SimConstants, Positionₙ⁺.V, ρₙ⁺, Velocityₙ⁺.V, dρdtIₙ⁺, dvdtIₙ⁺.V)
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
                        d2  = distance_condition(Positionₙ⁺.V[k1_idx],Positionₙ⁺.V[k2_idx])

                        if d2 <= TheCLL.CutOffSquared
                            nl += 1
                            sim_step2(k1_idx , k2_idx, d2, SimConstants, Positionₙ⁺.V, ρₙ⁺, Velocityₙ⁺.V, dρdtIₙ⁺, dvdtIₙ⁺.V)
                        end
                    end
                end
            end
    end

    dvdtIₙ⁺.vectors[end] .+= g * GravityFactor
    DensityEpsi!(Density,dρdtIₙ⁺,ρₙ⁺,dt)
    LimitDensityAtBoundary!(Density,BoundaryBool,ρ₀)

    @. Velocity.V += dvdtIₙ⁺.V * dt * MotionLimiter
    @. Position.V += ((Velocity.V + (Velocity.V - dvdtIₙ⁺.V * dt * MotionLimiter)) / 2) * dt * MotionLimiter


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
    Pressureᵢ          = zeros(FloatType, NumberOfPoints)
    
    drhopLp            = zeros(FloatType, NumberOfPoints)
    drhopLn            = zeros(FloatType, NumberOfPoints) 
    Pressureᵢ          = zeros(FloatType, NumberOfPoints)
    
    KernelGradient     = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
    Velocity           = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
    dvdtI              = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
    dvdtIₙ⁺            = DimensionalData{Dimensions,FloatType}(NumberOfPoints)

    Velocityₙ⁺ = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
    Positionₙ⁺ = DimensionalData{Dimensions,FloatType}(NumberOfPoints)
   
    # CustomCLL(SimConstants, MotionLimiter, BoundaryBool, GravityFactor, Position, Kernel, KernelGradient, Density, Velocity, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, dρdtI, dρdtIₙ⁺, dvdtI, dvdtIₙ⁺)
    function f()
        foreach(rm, filter(endswith(".vtp"), readdir(SimMetaData.SaveLocation,join=true)))
        PolyDataTemplate(SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(0,6,"0") * ".vtp", to_3d(Position.V), ["Kernel","KernelGradient","Density","Velocity", "Acceleration"], Kernel, KernelGradient.V, Density, Velocity.V, dvdtIₙ⁺.V)
        for iteration in 1:100000#:101
            CustomCLL(SimConstants, MotionLimiter, BoundaryBool, GravityFactor, Position, Kernel, KernelGradient, Density, Velocity, ρₙ⁺, Velocityₙ⁺, Positionₙ⁺, dρdtI,  dρdtIₙ⁺, dvdtI, dvdtIₙ⁺)
            if iteration % 50 == 0
                PolyDataTemplate(SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(iteration,6,"0") * ".vtp", to_3d(Position.V), ["Kernel","KernelGradient","Density","Velocity", "Acceleration"], Kernel, to_3d(KernelGradient.V), Density, Velocity.V, dvdtIₙ⁺.V)
                println(iteration)
            end
        end

        return nothing
    end
    f()
end



