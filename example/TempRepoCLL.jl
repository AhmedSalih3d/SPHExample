using BenchmarkTools
import CellListMap
using StaticArrays
using Parameters
using Plots; using Measures
using StructArrays
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

using SPHExample 

T = Float64
FloatType = T
D = 2
Dimensions = D

FluidCSV     = "./input/FluidPoints_Dp0.02.csv"
BoundCSV     = "./input/BoundaryPoints_Dp0.02.csv"

SimMetaData  = SimulationMetaData{D, T}(
                                SimulationName="MySimulation", 
                                SaveLocation=raw"E:\SecondApproach\Testing",
                                MaxIterations=10001,
                                OutputIteration=50,
)
# Initialze the constants to use
SimConstants = SimulationConstants{T}()

# Unpack the relevant simulation meta data
@unpack HourGlass, SaveLocation, SimulationName, MaxIterations, OutputIteration, SilentOutput, ThreadsCPU = SimMetaData;

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
Kernel            = zeros(FloatType, NumberOfPoints)
KernelL           = zeros(FloatType, NumberOfPoints)
dρdtI             = zeros(FloatType, NumberOfPoints)
ρₙ⁺               = zeros(FloatType, NumberOfPoints)
dρdtIₙ⁺           = zeros(FloatType, NumberOfPoints)

drhopLp            = zeros(FloatType, NumberOfPoints)
drhopLn            = zeros(FloatType, NumberOfPoints) 
Pressureᵢ          = zeros(FloatType, NumberOfPoints)

KernelGradient     = DimensionalData{Dimensions,FloatType}(NumberOfPoints)


function CustomCLL(p, SimConstants, Kernel, KernelGradient)
    @unpack ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η² = SimConstants
    R = 2*h
    TheCLL = CLL(Points=p,CutOff=R)

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
                    d2 = distance_condition(p[k_idx],p[k_1up])
                    d  = sqrt(d2)

                    xᵢⱼ = p[k_idx] - p[k_1up]

                    q  = clamp(d  * h⁻¹,0.0,2.0)
                    W  = αD*(1-q/2)^4*(2*q + 1)
                    Kernel[k_idx] += W
                    Kernel[k_1up] += W

                    Fac = αD*5*(q-2)^3*q / (8h*(q*h+1e-6))
                    KernelGradient.V[k_idx] +=  Fac * xᵢⱼ
                    KernelGradient.V[k_1up] += -Fac * xᵢⱼ
                    

                    #cond = d2 <= TheCLL.CutOffSquared
                    # # If cond true, we use nl + 1 as new index
                    # ind = ifelse(cond,nl+1,length(TheCLL.ListOfInteractions))
                    # TheCLL.ListOfInteractions[ind] = (k_idx,k_1up,sqrt(d2))
                    # # Then if cond true, update nl
                    # nl  = ifelse(cond,ind,nl)
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
                        d2  = distance_condition(p[k1_idx],p[k2_idx])
                        d   = sqrt(d2)
                        xᵢⱼ = p[k1_idx] - p[k2_idx]
                        q   = clamp(d  * h⁻¹,0.0,2.0)
                        W   = αD*(1-q/2)^4*(2*q + 1)
                        Kernel[k1_idx] += W 
                        Kernel[k2_idx] += W 

                        Fac = αD*5*(q-2)^3*q / (8h*(q*h+1e-6))
                        KernelGradient.V[k1_idx] +=  Fac * xᵢⱼ
                        KernelGradient.V[k2_idx] += -Fac * xᵢⱼ

                        # cond = d2 <= TheCLL.CutOffSquared
                        # # If cond true, we use nl + 1 as new index
                        # ind = ifelse(cond,nl+1,length(TheCLL.ListOfInteractions))
                        # TheCLL.ListOfInteractions[ind] = (k1_idx,k2_idx,sqrt(d2))
                        # # Then if cond true, update nl
                        # nl  = ifelse(cond,ind,nl)
                    end
                end
            end
    end

    TheCLL.MaxValidIndex[] = nl
    # resize!(TheCLL.ListOfInteractions,nl)

    return nothing
end

@profview CustomCLL(Position.V,SimConstants, Kernel, KernelGradient)

to_3d(vec_2d) = [SVector(v..., 0.0) for v in vec_2d]
 PolyDataTemplate(SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(SimMetaData.Iteration,6,"0") * ".vtp", to_3d(Position.V), ["Kernel","KernelGradient"], Kernel, KernelGradient.V)

@benchmark CustomCLL($Position.V,$SimConstants, $Kernel, $KernelGradient)