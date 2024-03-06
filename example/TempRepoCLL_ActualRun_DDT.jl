using SPHExample
using BenchmarkTools
using StaticArrays
using Parameters
using Plots; using Measures
using StructArrays
import LinearAlgebra: dot
using LoopVectorization
using Polyester
using JET
using Formatting
using ProgressMeter
using TimerOutputs
using FastPow
using ChunkSplitters
import Cthulhu as Deep

import Base.Threads: nthreads, @threads
include("../src/ProduceVTP.jl")


function ConstructGravitySVector(_::SVector{N, T}, value) where {N, T}
    return SVector{N, T}(ntuple(i -> i == N ? value : 0, N))
end


@with_kw struct CLL{D,T}
    Points::Vector{SVector{D,T}}

    CutOff::T
    CutOffSquared::T                             = CutOff^2
    Padding::Int64                               = 2
    HalfPad::Int64                               = convert(typeof(Padding),Padding//2)
    ZeroOffset::Int64                            = 1 #Since we start from 0 when generating cells

    Cells::Vector{NTuple{D, Int64}}              = ExtractCells(Points,CutOff,Val(getsvecD(eltype(Points))))
    Nmax::Int64                                  = maximum(reinterpret(Int,@view(Cells[:]))) + ZeroOffset #Find largest dimension in x,y,z for the Cells

    UniqueCells::Vector{NTuple{D, Int64}}        = union(Cells) #just do all cells for now, optimize later
    Layout::Array{Vector{Int64}, D}              = GenerateM(Nmax,ZeroOffset,HalfPad,Padding,Cells,Val(getsvecD(eltype(Points))))
end
@inline getsvecD(::Type{SVector{d,T}}) where {d,T} = d


@inline function distance_condition(p1::SVector, p2::SVector)
    return dot(p1 - p2, p1 - p2)
end

# https://jaantollander.com/post/searching-for-fixed-radius-near-neighbors-with-cell-lists-algorithm-in-julia-language/#definition
function neighbors(v::Val{d}) where d
    n_ = CartesianIndices(ntuple(_->-1:1,v))
    half_length = length(n_) ÷ 2
    n  = n_[1:half_length]
    
    # n_svec = Vector{NTuple{d,Int}}(undef,length(n)) #zeros(SVector{d,eltype(d)},length(n))

    # for i ∈ eachindex(n_svec)
    #     val       = n[i]
    #     n_svec[i] = (val.I)
    # end

    # return n_svec

    return n
end


function ExtractCells(p,R,::Val{d}) where d
    cells = Vector{NTuple{d,Int}}(undef,length(p))

    @inbounds @batch for i ∈ eachindex(p)
        vs = @. Int(fld(p[i],R))
        cells[i] = tuple(vs...)
    end

    return cells
end

function ExtractCells!(cells, p,R,::Val{d}) where d

    @inbounds @batch for i ∈ eachindex(p)
        vs = @. Int(fld(p[i],R))
        cells[i] = tuple(vs...)
    end

    return cells
end

function GenerateM(Nmax,ZeroOffset,HalfPad,Padding,cells,v::Val{d}) where d
    Msize = ntuple(_ -> Nmax+Padding,v)
    
    M     = Array{Vector{Int}}(undef,Msize)
    @inbounds @batch for i = 1:prod(size(M))
        arr  = Vector{Int}()
        M[i] = arr
    end

    iter = 0

    @inbounds for ind ∈ cells
        iter += 1
        push!(M[(ind .+ ZeroOffset .+ HalfPad)...],iter)
    end

    return M
end

function GenerateM!(M, ZeroOffset,HalfPad, cells)

    @inbounds @batch for i = 1:prod(size(M))
        empty!(M[i])
    end

    iter = 0
    @inbounds for ind ∈ cells
        iter += 1
        subind = @. ind + ZeroOffset + HalfPad
        if !checkbounds(Bool, M,subind...) continue end
        push!(M[subind...],iter)
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

@inline function ∇ᵢWᵢⱼ_(αD,h,xᵢⱼ,q, η²)
    @fastpow return (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 
end

# Combine into terms, this way can utilize slight symmetry
@inline function dρᵢdt_dρⱼdt_(ρᵢ,ρⱼ,m₀,vᵢⱼ,∇ᵢWᵢⱼ)
    #original: - ρᵢ * dot((m₀/ρⱼ) *  -vᵢⱼ ,  ∇ᵢWᵢⱼ)
    symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ) # = dot(vᵢⱼ , -∇ᵢWᵢⱼ)
    dρdt⁺          = - ρᵢ * (m₀/ρⱼ) *  symmetric_term
    dρdt⁻          = - ρⱼ * (m₀/ρᵢ) *  symmetric_term

    return dρdt⁺, dρdt⁻
end

#https://discourse.julialang.org/t/can-this-be-written-even-faster-cpu/109924/28
@inline faux_fancy(ρ₀, P, invCb) = ρ₀ * ( fancy7th( 1 + (P * invCb)) - 1)


@inbounds function sim_step(i , j, d2, SimConstants, ∇Cᵢ, ∇◌rᵢ, KernelGradient, Position, Density, Velocity, dρdtI, dvdtI, MotionLimiter,BoolDDT=true)
    @unpack h, m₀, h⁻¹,  α ,  αD, c₀, γ, ρ₀, Cb⁻¹, g, η² = SimConstants
    #https://discourse.julialang.org/t/sqrt-abs-x-is-even-faster-than-sqrt/58154/12
    d  = sqrt(abs(d2))
    d² = d*d

    xᵢ  = Position[i]
    xⱼ  = Position[j]
    xᵢⱼ = xᵢ - xⱼ

    invd²η² = inv(d²+η²) 
    

    q  = d  * h⁻¹ #clamp(d  * h⁻¹,0.0,2.0), not needed when checking d2 < CutOffSquared before hand

    @fastpow ∇ᵢWᵢⱼ = ∇ᵢWᵢⱼ_(αD,h,xᵢⱼ,q, η²)

    KernelGradient[i] +=  ∇ᵢWᵢⱼ
    KernelGradient[j] += -∇ᵢWᵢⱼ 

    ρᵢ      = Density[i]
    ρⱼ      = Density[j]

    vᵢ      = Velocity[i]
    vⱼ      = Velocity[j]
    vᵢⱼ     = vᵢ - vⱼ

    dρdt⁺, dρdt⁻ = dρᵢdt_dρⱼdt_(ρᵢ,ρⱼ,m₀,vᵢⱼ,∇ᵢWᵢⱼ)

    # Density diffusion
    if BoolDDT
        Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
        ρᵢⱼᴴ  = faux_fancy(ρ₀, Pᵢⱼᴴ, Cb⁻¹)
        Pⱼᵢᴴ  = -Pᵢⱼᴴ
        ρⱼᵢᴴ  = faux_fancy(ρ₀, Pⱼᵢᴴ, Cb⁻¹)
    
        ρⱼᵢ   = ρⱼ - ρᵢ

        MLcond = MotionLimiter[i] * MotionLimiter[j]
        Dᵢ  = h * c₀ * (m₀/ρⱼ) * 2 * ( ρⱼᵢ - ρᵢⱼᴴ) * invd²η² * dot(-xᵢⱼ,  ∇ᵢWᵢⱼ) * MLcond
        Dⱼ  = h * c₀ * (m₀/ρᵢ) * 2 * (-ρⱼᵢ - ρⱼᵢᴴ) * invd²η² * dot( xᵢⱼ, -∇ᵢWᵢⱼ) * MLcond

        dρdtI[i] += dρdt⁺ + Dᵢ
        dρdtI[j] += dρdt⁻ + Dⱼ
    else
        dρdtI[i] += dρdt⁺
        dρdtI[j] += dρdt⁻
    end

    Pᵢ      =  EquationOfStateGamma7(ρᵢ,c₀,ρ₀)
    Pⱼ      =  EquationOfStateGamma7(ρⱼ,c₀,ρ₀)

    ρ̄ᵢⱼ     = (ρᵢ+ρⱼ)*0.5
    Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)

    cond      = dot(vᵢⱼ, xᵢⱼ)
    cond_bool = cond < 0.0
    μᵢⱼ       = h*cond * invd²η²
    Πᵢⱼ       = cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ

    dvdt⁺ = - m₀ * ( Pfac + Πᵢⱼ) *  ∇ᵢWᵢⱼ
    dvdt⁻ = - dvdt⁺ #- m₀ * ( Pfac + Πᵢⱼ) * -∇ᵢWᵢⱼ

    dvdtI[i] += dvdt⁺
    dvdtI[j] += dvdt⁻

    # Particle Shifting
    ∇Cᵢ[i]   += (m₀/ρⱼ) *  ∇ᵢWᵢⱼ
    ∇Cᵢ[j]   += (m₀/ρᵢ) * -∇ᵢWᵢⱼ

    ∇◌rᵢ[i]  += (m₀/ρⱼ) * dot(xᵢⱼ , ∇ᵢWᵢⱼ)
    ∇◌rᵢ[j]  += (m₀/ρᵢ) * dot(-xᵢⱼ ,-∇ᵢWᵢⱼ)

    return nothing
end

@inbounds function neighbor_loop(TheCLL, LoopLayout, Stencil, SimConstants, ∇Cᵢ, ∇◌rᵢ, KernelGradient, Position, Density, Velocity, dρdtI, dvdtI, MotionLimiter, BoolDDT=true)
    nchunks = nthreads()
    @threads for ichunk in 1:nchunks
        for Cind_ ∈ getchunk(LoopLayout, ichunk; n=nchunks)
            for Ci ∈ Cind_
                Cind = LoopLayout[Ci]
                if !isassigned(TheCLL.Layout,Cind) continue end
                # The indices in the cell are:
                indices_in_cell = TheCLL.Layout[Cind]

                n_idx_cells = length(indices_in_cell)
                @inbounds for ki = 1:n_idx_cells-1
                    k_idx = indices_in_cell[ki]
                    @inbounds for kj = (ki+1):n_idx_cells
                        k_1up = indices_in_cell[kj]
                        d2 = distance_condition(Position[k_idx],Position[k_1up])

                        if d2 < TheCLL.CutOffSquared
                            @inbounds @inline sim_step(k_idx , k_1up, d2, SimConstants, ∇Cᵢ, ∇◌rᵢ, KernelGradient, Position, Density, Velocity, dρdtI, dvdtI, MotionLimiter, BoolDDT)
                        end
                    end
                end
        
                for Sind_ ∈ Stencil
                    Sind = Cind + Sind_
        
                    # Keep this in, because some cases break without it..
                    if !isassigned(TheCLL.Layout,Sind) continue end
        
                    indices_in_cell_plus  = TheCLL.Layout[Sind]
        
                    # Here a double loop to compare indices_in_cell[k] to all possible neighbours
                    for k1 ∈ eachindex(indices_in_cell)
                        k1_idx = indices_in_cell[k1]
                        for k2 ∈ eachindex(indices_in_cell_plus)
                            k2_idx = indices_in_cell_plus[k2]
                            d2  = distance_condition(Position[k1_idx],Position[k2_idx])
        
                            if d2 < TheCLL.CutOffSquared
                                @inbounds @inline sim_step(k1_idx , k2_idx, d2, SimConstants, ∇Cᵢ, ∇◌rᵢ, KernelGradient, Position, Density, Velocity, dρdtI, dvdtI, MotionLimiter, BoolDDT)
                            end
                        end
                    end
                end
            end
        end
    end
end



function updateCLL!(cll::CLL,Points)
    # Update Cells based on new positions of Points
    ExtractCells!(cll.Cells,Points, cll.CutOff, Val(getsvecD(eltype(Points))))
    
    GenerateM!(cll.Layout, cll.ZeroOffset, cll.HalfPad, cll.Cells)

    return nothing
end

@fastpow function EquationOfStateGamma7(ρ,c₀,ρ₀)
    return ((c₀^2*ρ₀)/7) * ((ρ/ρ₀)^7 - 1)
end

function EquationOfState(ρ,c₀,γ,ρ₀)
    return ((c₀^2*ρ₀)/γ) * ((ρ/ρ₀)^γ - 1)
end

function CustomCLL(TheCLL, LoopLayout, Stencil, SimConstants, SimMetaData, MotionLimiter, BoundaryBool, GravityFactor, ∇Cᵢ, ∇◌rᵢ, KernelGradient, Position, Density, Velocity, ρₙ⁺, Velocityₙ⁺, Positionₙ⁺, dρdtI, dρdtIₙ⁺, dvdtI, dvdtIₙ⁺; BoolDDT = true)
    @unpack ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, δᵩ, CFL, η² = SimConstants

    dt  = Δt(Position, Velocity, dvdtIₙ⁺, SimConstants)
    dt₂ = dt * 0.5

    ResetArrays!(∇Cᵢ, ∇◌rᵢ, KernelGradient, dρdtI, dvdtI)
    neighbor_loop(TheCLL, LoopLayout, Stencil, SimConstants, ∇Cᵢ, ∇◌rᵢ, KernelGradient, Position, Density, Velocity, dρdtI, dvdtI, MotionLimiter,  BoolDDT)

    # Make loop, no allocs
    @inbounds @batch for i in eachindex(dvdtI)
        dvdtI[i]       += ConstructGravitySVector(dvdtI[i], g * GravityFactor[i])
        Velocityₙ⁺[i]   = Velocity[i]   + dvdtI[i]       *  dt₂ * MotionLimiter[i]
        Positionₙ⁺[i]   = Position[i]   + Velocityₙ⁺[i]   * dt₂  * MotionLimiter[i]
        ρₙ⁺[i]          = Density[i]    + dρdtI[i]       *  dt₂
    end

    LimitDensityAtBoundary!(ρₙ⁺,BoundaryBool,ρ₀)

    ResetArrays!(∇Cᵢ, ∇◌rᵢ, KernelGradient, dρdtIₙ⁺, dvdtIₙ⁺)
    neighbor_loop(TheCLL, LoopLayout, Stencil, SimConstants, ∇Cᵢ, ∇◌rᵢ, KernelGradient, Positionₙ⁺, ρₙ⁺, Velocityₙ⁺, dρdtIₙ⁺, dvdtIₙ⁺, MotionLimiter,  BoolDDT)
    
    DensityEpsi!(Density,dρdtIₙ⁺,ρₙ⁺,dt)
    LimitDensityAtBoundary!(Density,BoundaryBool,ρ₀)

    @inbounds @batch for i in eachindex(dvdtIₙ⁺)
        dvdtIₙ⁺[i]            +=  ConstructGravitySVector(dvdtIₙ⁺[i], g * GravityFactor[i])
        Velocity[i]           += dvdtIₙ⁺[i] * dt * MotionLimiter[i]
        Position[i]           += ((Velocity[i] + (Velocity[i] - dvdtIₙ⁺[i] * dt * MotionLimiter[i])) / 2) * dt * MotionLimiter[i]
    end

    SimMetaData.Iteration      += 1
    SimMetaData.CurrentTimeStep = dt
    SimMetaData.TotalTime      += dt

    return nothing
end

to_3d(vec_2d) = [SVector(v..., 0.0) for v in vec_2d]

# For testing script properly
function RunSimulation(;FluidCSV::String,
    BoundCSV::String,
    SimMetaData::SimulationMetaData{Dimensions, FloatType},
    SimConstants::SimulationConstants,
    BoolDDT = true
) where {Dimensions,FloatType}

    # Unpack the relevant simulation meta data
    @unpack HourGlass, SaveLocation, SimulationName, SilentOutput, ThreadsCPU = SimMetaData;
    
    # Unpack simulation constants
    @unpack ρ₀, dx, h, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η² = SimConstants
    
    # Load in the fluid and boundary particles. Return these points and both data frames
    # @inline is a hack here to remove all allocations further down due to uncertainty of the points type at compile time
    @inline points, density_fluid, density_bound  = LoadParticlesFromCSV(Dimensions,FloatType, FluidCSV,BoundCSV)
    Position           = convert(Vector{SVector{Dimensions,FloatType}},points.V)
    
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
    
    # State variables
    Position          = convert(Vector{SVector{Dimensions,FloatType}},points.V)
    Velocity          = zeros(SVector{Dimensions,FloatType},NumberOfPoints)
    Density           = deepcopy([density_fluid;density_bound])
    Pressureᵢ         = @. EquationOfStateGamma7(Density,c₀,ρ₀)

    # Derivatives
    dρdtI             = zeros(FloatType, NumberOfPoints)
    dρdtIₙ⁺           = zeros(FloatType, NumberOfPoints)
    dvdtI              = zeros(SVector{Dimensions,FloatType},NumberOfPoints)
    dvdtIₙ⁺            = zeros(SVector{Dimensions,FloatType},NumberOfPoints)

    # Kernel values
    KernelGradient    = zeros(SVector{Dimensions,FloatType},NumberOfPoints)

    # Shifting correction
    ∇Cᵢ               = zeros(SVector{Dimensions,FloatType},NumberOfPoints)            
    ∇◌rᵢ              = zeros(FloatType,NumberOfPoints)            

    # Half point values for predictor-corrector algorithm
    Velocityₙ⁺ = zeros(SVector{Dimensions,FloatType},NumberOfPoints)
    Positionₙ⁺ = zeros(SVector{Dimensions,FloatType},NumberOfPoints)
    ρₙ⁺        = zeros(FloatType, NumberOfPoints)

    if !isdir(SimMetaData.SaveLocation)
        mkdir(SimMetaData.SaveLocation)
    end
    
    foreach(rm, filter(endswith(".vtp"), readdir(SimMetaData.SaveLocation,join=true)))
   

    SaveLocation_ = SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(0,6,"0") * ".vtp"
    PolyDataTemplate(SaveLocation_, to_3d(Position), ["ParticleConcentration", "ParticleDivergence", "KernelGradient", "Density", "Pressure","Velocity", "Acceleration"], ∇Cᵢ, ∇◌rᵢ, KernelGradient, Density, Pressureᵢ, Velocity, dvdtIₙ⁺)

    R = 2*h
    TheCLL = CLL(Points=Position,CutOff=R) #line is good idea at times

    # Assymmetric Stencil.
    LoopLayout = CartesianIndex.(CartesianIndices(TheCLL.Layout))[1:end-1, 2:end-1][:]
    Stencil    = neighbors(Val(getsvecD(eltype(Position))))

    generate_showvalues(Iteration, TotalTime) = () -> [(:(Iteration),format(FormatExpr("{1:d}"),  Iteration)), (:(TotalTime),format(FormatExpr("{1:3.3f}"), TotalTime))]
    OutputCounter = 0.0
    OutputIterationCounter = 0
    @time @inbounds while true
        
        @timeit HourGlass "0 Update particles in cells" updateCLL!(TheCLL, Position)
        # inline removes 96 bytes alloc..
        @timeit HourGlass "1 Main simulation loop" CustomCLL(TheCLL, LoopLayout, Stencil, SimConstants, SimMetaData, MotionLimiter, BoundaryBool, GravityFactor, ∇Cᵢ, ∇◌rᵢ, KernelGradient, Position, Density, Velocity, ρₙ⁺, Velocityₙ⁺, Positionₙ⁺, dρdtI,  dρdtIₙ⁺, dvdtI, dvdtIₙ⁺; 
        BoolDDT)
        
        OutputCounter += SimMetaData.CurrentTimeStep
        @timeit HourGlass "2 Output data" if OutputCounter >= SimMetaData.OutputEach
            OutputCounter = 0.0
            OutputIterationCounter += 1
            SaveLocation_= SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(OutputIterationCounter,6,"0") * ".vtp"
            Pressure!(Pressureᵢ,Density,SimConstants)
            PolyDataTemplate(SaveLocation_, to_3d(Position), ["ParticleConcentration", "ParticleDivergence", "KernelGradient", "Density", "Pressure", "Velocity", "Acceleration"], ∇Cᵢ, ∇◌rᵢ, KernelGradient, Density, Pressureᵢ, Velocity, dvdtIₙ⁺)
        end

        @timeit HourGlass "3 Next step" next!(SimMetaData.ProgressSpecification; showvalues = generate_showvalues(SimMetaData.Iteration , SimMetaData.TotalTime))

        if SimMetaData.TotalTime >= SimMetaData.SimulationTime + 1e-3
            break
        end
    end

    disable_timer!(HourGlass)
    show(HourGlass,sortby=:name)
    show(HourGlass)
    
    return nothing
end



# Initialize Simulation
begin
    D = 2
    T = Float64
    SimMetaData  = SimulationMetaData{D, T}(
                                    SimulationName="AllInOne", 
                                    SaveLocation=raw"E:\SecondApproach\Testing",
                                    SimulationTime=4,
                                    OutputEach=0.01
    )

    # Initialze the constants to use
    SimConstants = SimulationConstants{T}(
        dx = 0.02,
        h  = 1*sqrt(2)*0.02,
        c₀ = 88.14487860902641,
        α  = 0.02
    )

    # And here we run the function - enjoy!
    println(@code_warntype RunSimulation(
        FluidCSV     = "./input/DSPH_DamBreak_Fluid_Dp0.02.csv",
        BoundCSV     = "./input/DSPH_DamBreak_Boundary_Dp0.02.csv",
        SimMetaData  = SimMetaData,
        SimConstants = SimConstants
    )
    )

    println(@report_opt target_modules=(@__MODULE__,) RunSimulation(
        FluidCSV     = "./input/DSPH_DamBreak_Fluid_Dp0.02.csv",
        BoundCSV     = "./input/DSPH_DamBreak_Boundary_Dp0.02.csv",
        SimMetaData  = SimMetaData,
        SimConstants = SimConstants
    )
    )

    println(@report_call target_modules=(@__MODULE__,) RunSimulation(
        FluidCSV     = "./input/DSPH_DamBreak_Fluid_Dp0.02.csv",
        BoundCSV     = "./input/DSPH_DamBreak_Boundary_Dp0.02.csv",
        SimMetaData  = SimMetaData,
        SimConstants = SimConstants
    )
    )

    SimConstantsWedge = SimulationConstants{T}()
    @profview RunSimulation(
        FluidCSV     = "./input/StillWedge_Fluid_Dp0.02_LowResolution.csv",
        BoundCSV     = "./input/StillWedge_Bound_Dp0.02_LowResolution_5LAYERS.csv",
        SimMetaData  = SimMetaData,
        SimConstants = SimConstantsWedge
    )

    SimConstantsDamBreak = SimulationConstants{T}()
    # @profview RunSimulation(
    #     FluidCSV     = "./input/FluidPoints_Dp0.02_5LAYERS.csv",
    #     BoundCSV     = "./input/BoundaryPoints_Dp0.02_5LAYERS.csv",
    #     SimMetaData  = SimMetaData,
    #     SimConstants = SimConstantsDamBreak,
    #     BoolDDT      = true
    # )

    # RunSimulation(
    #     FluidCSV     = "./input/FluidPoints_Dp0.02_5LAYERS.csv",
    #     BoundCSV     = "./input/BoundaryPoints_Dp0.02_5LAYERS.csv",
    #     SimMetaData  = SimMetaData,
    #     SimConstants = SimConstantsDamBreak,
    #     BoolDDT      = true
    # )
end
