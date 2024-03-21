using SPHExample
using BenchmarkTools
using StaticArrays
using Parameters
using StructArrays
import LinearAlgebra: dot, norm, diagm, diag, cond, det
using LoopVectorization
using FastPow
import CellListMap: InPlaceNeighborList, update!, neighborlist!
import ProgressMeter: next!
using Formatting
using Bumper

import Base.Threads: nthreads, @threads
include("../src/ProduceVTP.jl")

function update_arr1_bumper!(arr1,indices)
    @no_escape begin
        temp  = @alloc(eltype(arr1),length(arr1))

        temp .= @view arr1[indices]
        arr1 .= temp
        
    end
end


function ConstructGravitySVector(_::SVector{N, T}, value) where {N, T}
    return SVector{N, T}(ntuple(i -> i == N ? value : 0, N))
end

function ConstructStencil(v::Val{d}) where d
    n_ = CartesianIndices(ntuple(_->-1:1,v))
    half_length = length(n_) ÷ 2
    n  = n_[1:half_length]

    return n
end

###=== Extract Cells
function ExtractCells!(Cells, Points, CutOff)
    for i ∈ eachindex(Cells)
        Cells[i] =  CartesianIndex(@. Int(fld(Points[i], CutOff)) ...) + CartesianIndex(1,1) + CartesianIndex(1,1) #+ ZeroOffset + HalfPad
    end
    return nothing
end

###===

###=== SimStep
@inline function dρᵢdt_dρⱼdt_(ρᵢ,ρⱼ,m₀,vᵢⱼ,∇ᵢWᵢⱼ)
    #original: - ρᵢ * dot((m₀/ρⱼ) *  -vᵢⱼ ,  ∇ᵢWᵢⱼ)
    symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ) # = dot(vᵢⱼ , -∇ᵢWᵢⱼ)
    dρdt⁺          = - ρᵢ * (m₀/ρⱼ) *  symmetric_term
    dρdt⁻          = - ρⱼ * (m₀/ρᵢ) *  symmetric_term

    return dρdt⁺, dρdt⁻
end

@fastpow function EquationOfStateGamma7(ρ,c₀,ρ₀)
    return ((c₀^2*ρ₀)/7) * ((ρ/ρ₀)^7 - 1)
end


function SimStepLocalCell(SimConstants, Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, StartIndex, EndIndex)
    @unpack ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η², H² = SimConstants

    @inbounds for i = StartIndex:EndIndex, j = (i+1):EndIndex

        xᵢⱼ  = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)

        if  xᵢⱼ² <= H²
            dᵢⱼ  = sqrt(xᵢⱼ²) #Using sqrt is what takes a lot of time?
            q    = clamp(dᵢⱼ * h⁻¹,0.0,2.0)
            Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)

            invd²η² = inv(dᵢⱼ*dᵢⱼ+η²)

            ∇ᵢWᵢⱼ = @fastpow (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 

            ρᵢ        = Density[i]
            ρⱼ        = Density[j]
        
            vᵢ        = Velocity[i]
            vⱼ        = Velocity[j]
            vᵢⱼ       = vᵢ - vⱼ

            dρdt⁺, dρdt⁻ = dρᵢdt_dρⱼdt_(ρᵢ,ρⱼ,m₀,vᵢⱼ,∇ᵢWᵢⱼ)

            Pᵢ        =  EquationOfStateGamma7(ρᵢ,c₀,ρ₀)
            Pⱼ        =  EquationOfStateGamma7(ρⱼ,c₀,ρ₀)
            Pfac      = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)

            ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
            cond      = dot(vᵢⱼ, xᵢⱼ)
            cond_bool = cond < 0.0
            μᵢⱼ       = h*cond * invd²η²
            Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
            Πⱼ        = - Πᵢ

            dvdt⁺ = - m₀ * (Pfac) *  ∇ᵢWᵢⱼ + Πᵢ
            dvdt⁻ = - dvdt⁺ + Πⱼ

            dρdtI[i] += dρdt⁺
            dρdtI[j] += dρdt⁻

            dvdtI[i] +=  dvdt⁺
            dvdtI[j] +=  dvdt⁻

            Kernel[i] += Wᵢⱼ
            Kernel[j] += Wᵢⱼ

            KernelGradient[i] +=  ∇ᵢWᵢⱼ
            KernelGradient[j] += -∇ᵢWᵢⱼ
        end
    end

    return nothing
end


function SimStepNeighborCell(SimConstants, Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, StartIndex, EndIndex, StartIndex_, EndIndex_)
    @unpack ρ₀, dx, h, h⁻¹, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η², H² = SimConstants

    @inbounds for i = StartIndex:EndIndex, j = StartIndex_:EndIndex_

        xᵢⱼ  = Position[i] - Position[j]
        xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)

        if  xᵢⱼ² <= H²
            dᵢⱼ  = sqrt(xᵢⱼ²) #Using sqrt is what takes a lot of time?
            q    = clamp(dᵢⱼ * h⁻¹,0.0,2.0)
            Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)

            invd²η² = inv(dᵢⱼ*dᵢⱼ+η²)

            ∇ᵢWᵢⱼ = @fastpow (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 

            ρᵢ        = Density[i]
            ρⱼ        = Density[j]
        
            vᵢ        = Velocity[i]
            vⱼ        = Velocity[j]
            vᵢⱼ       = vᵢ - vⱼ

            dρdt⁺, dρdt⁻ = dρᵢdt_dρⱼdt_(ρᵢ,ρⱼ,m₀,vᵢⱼ,∇ᵢWᵢⱼ)

            Pᵢ        =  EquationOfStateGamma7(ρᵢ,c₀,ρ₀)
            Pⱼ        =  EquationOfStateGamma7(ρⱼ,c₀,ρ₀)
            Pfac      = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)

            ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
            cond      = dot(vᵢⱼ, xᵢⱼ)
            cond_bool = cond < 0.0
            μᵢⱼ       = h*cond * invd²η²
            Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
            Πⱼ        = - Πᵢ

            dvdt⁺ = - m₀ * (Pfac) *  ∇ᵢWᵢⱼ + Πᵢ
            dvdt⁻ = - dvdt⁺ + Πⱼ

            dρdtI[i] += dρdt⁺
            dρdtI[j] += dρdt⁻

            dvdtI[i] +=  dvdt⁺
            dvdtI[j] +=  dvdt⁻

            Kernel[i] += Wᵢⱼ
            Kernel[j] += Wᵢⱼ

            KernelGradient[i] +=  ∇ᵢWᵢⱼ
            KernelGradient[j] += -∇ᵢWᵢⱼ
        end
    end

    return nothing
end

###===

###=== Function to update ordering
#https://cuda.juliagpu.org/stable/tutorials/performance/
function UpdateNeighbors!(Cells, CutOff, SortedIndices, Position, Density, Acceleration, Velocity, ParticleSplitter, ParticleSplitterLinearIndices)
    ExtractCells!(Cells,Position,CutOff)

    sortperm!(SortedIndices,Cells)

    # @. Cells           =  Cells[SortedIndices]
    # @. Position        =  Position[SortedIndices]
    # @. Density         =  Density[SortedIndices]
    # @. Acceleration    =  Acceleration[SortedIndices]
    # @. Velocity        =  Velocity[SortedIndices]

    update_arr1_bumper!(Cells, SortedIndices)
    update_arr1_bumper!(Position, SortedIndices)
    update_arr1_bumper!(Density, SortedIndices)
    update_arr1_bumper!(Acceleration, SortedIndices)
    update_arr1_bumper!(Velocity, SortedIndices)    

    ParticleSplitter[findall(.!iszero.(diff(Cells))) .+ 1] .= true

    # # Passing the view is fine, since it is not needed to actualize the vector
    @views ParticleRanges     = ParticleSplitterLinearIndices[ParticleSplitter]
    @views UniqueCells        = Cells[ParticleRanges[1:end-1]]

    return ParticleRanges, UniqueCells #Optimize out in shaa Allah!
end
###===

###=== Function to process each cell and its neighbors
#https://cuda.juliagpu.org/stable/tutorials/performance/
# 192 bytes and 4 allocs from launch config
# INLINE IS SO IMPORTANT 10X SPEED
function NeighborLoop!(SimConstants, UniqueCells, ParticleRanges, Stencil, Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI)
    for iter ∈ eachindex(UniqueCells)
        CellIndex = UniqueCells[iter]

        StartIndex = ParticleRanges[iter] 
        EndIndex   = ParticleRanges[iter+1] - 1

        @inline SimStepLocalCell(SimConstants, Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, StartIndex, EndIndex)

        @inbounds for S ∈ Stencil
            SCellIndex = CellIndex + S
    
            Needle            = isequal(SCellIndex)
            NeighborCellIndex = findfirst(Needle, UniqueCells)

            if !isnothing(NeighborCellIndex)
                StartIndex_       = ParticleRanges[NeighborCellIndex] 
                EndIndex_         = ParticleRanges[NeighborCellIndex+1] - 1

                @inline SimStepNeighborCell(SimConstants, Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, StartIndex, EndIndex, StartIndex_, EndIndex_)
            end
        end
    end

    return nothing
end

function SimulationLoop(SimConstants, Cells, Stencil, SortedIndices, ParticleSplitter, ParticleSplitterLinearIndices, Position, Kernel, KernelGradient, Density, Velocity, Acceleration, dρdtI, dvdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, dρdtIₙ⁺, GravityFactor, MotionLimiter, BoundaryBool)
    dt  = 1e-5
    dt₂ = dt * 0.5

    ParticleRanges,UniqueCells     = UpdateNeighbors!(Cells, SimConstants.H, SortedIndices, Position, Density, Acceleration, Velocity, ParticleSplitter, ParticleSplitterLinearIndices)
    
    ResetArrays!(Kernel, KernelGradient, dρdtI, dvdtI)

    NeighborLoop!(SimConstants, UniqueCells, ParticleRanges, Stencil, Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI)

    @inbounds for i in eachindex(Position)
        dvdtI[i]        +=  ConstructGravitySVector(dvdtI[i], g * GravityFactor[i])
        Velocityₙ⁺[i]    =  Velocity[i]   + dvdtI[i]  *  dt₂ * MotionLimiter[i]
        Positionₙ⁺[i]    =  Position[i]   + Velocityₙ⁺[i]   * dt₂  * MotionLimiter[i]
        ρₙ⁺[i]           =  Density[i]    + dρdtI[i]       *  dt₂
    end

    LimitDensityAtBoundary!(ρₙ⁺,BoundaryBool, SimConstants.ρ₀)

    ResetArrays!(Kernel, KernelGradient, dρdtI, dρdtIₙ⁺, dvdtI)

    NeighborLoop!(SimConstants, UniqueCells, ParticleRanges, Stencil, Positionₙ⁺, Kernel, KernelGradient, ρₙ⁺, Velocityₙ⁺, dρdtIₙ⁺, dvdtI)

    DensityEpsi!(Density,dρdtIₙ⁺,ρₙ⁺,dt)

    LimitDensityAtBoundary!(Density,BoundaryBool, SimConstants.ρ₀)

    @inbounds for i in eachindex(Position)
        dvdtI[i]       +=  ConstructGravitySVector(dvdtI[i], g * GravityFactor[i])
        Velocity[i]     +=  dvdtI[i] * dt * MotionLimiter[i]
        Position[i]     += (((Velocity[i] + (Velocity[i] - SVector(dvdtIX[i],dvdtIY[i]) * dt * MotionLimiter[i])) / 2) * dt) * MotionLimiter[i]
    end

    SimMetaData.Iteration      += 1
    SimMetaData.CurrentTimeStep = dt
    SimMetaData.TotalTime      += dt
end

###===

# Dimensions = 2
# FloatType  = Float64
# FluidCSV   = "./input/still_wedge_mdbc/StillWedge_Dp0.02_Fluid.csv"
# BoundCSV   = "./input/still_wedge_mdbc/StillWedge_Dp0.02_Bound.csv"

# KernelGradient = [KernelGradientX'; KernelGradientY']
# to_3d(vec_2d) = [SVector(v..., 0.0) for v in vec_2d]
# create_vtp_file(SimMetaData, SimConstantsWedge, to_3d(Position); Kernel, KernelGradient)

# display(@benchmark ParticleRanges,UniqueCells     = UpdateNeighbors!($Cells, $H, $SortedIndices, $Position, $Density, $Acceleration, $Velocity, $ParticleSplitter, $ParticleSplitterLinearIndices))
# display(@benchmark NeighborLoop!($SimConstantsWedge, $UniqueCells, $ParticleRanges, $Stencil, $Position, $Kernel, $KernelGradientX, $KernelGradientY, $Density, $Velocity, $dρdtI, $dvdtIX, $dvdtIY))

# @profview ParticleRanges,UniqueCells     = UpdateNeighbors!(Cells, H, SortedIndices, Position, Density, Acceleration, Velocity, ParticleSplitter, ParticleSplitterLinearIndices)
# @profview NeighborLoop!(SimConstantsWedge, UniqueCells, ParticleRanges, Stencil, Position, Kernel, KernelGradientX, KernelGradientY, Density, Velocity, dρdtI, dvdtIX, dvdtIY)

function RunSimulation(;FluidCSV::String,
    BoundCSV::String,
    SimMetaData::SimulationMetaData{Dimensions, FloatType},
    SimConstants::SimulationConstants,
    ViscosityTreatment = :LaminarSPS,
    BoolDDT = true,
    BoolShifting = true
    ) where {Dimensions,FloatType}

    if !isdir(SimMetaData.SaveLocation)
        mkdir(SimMetaData.SaveLocation)
    end
    
    foreach(rm, filter(endswith(".vtp"), readdir(SimMetaData.SaveLocation,join=true)))

    # Unpack the relevant simulation meta data
    @unpack HourGlass, SaveLocation, SimulationName, SilentOutput, ThreadsCPU = SimMetaData;

    # Unpack simulation constants
    @unpack ρ₀, dx, h, m₀, αD, α, g, c₀, γ, dt, δᵩ, CFL, η², H = SimConstants

    # Load in the fluid and boundary particles. Return these points and both data frames
    # @inline is a hack here to remove all allocations further down due to uncertainty of the points type at compile time
    @inline points, density_fluid, density_bound  = LoadParticlesFromCSV(Dimensions,FloatType, FluidCSV,BoundCSV)
    NumberOfPoints = length(points)
    Position = convert(Vector{SVector{Dimensions,FloatType}},points.V)
    Density  = deepcopy([density_bound; density_fluid])

    

    GravityFactor = [ zeros(size(density_bound,1)) ; -ones(size(density_fluid,1)) ]
    
    MotionLimiter = [ zeros(size(density_bound,1)) ;  ones(size(density_fluid,1)) ]

    BoundaryBool  = .!Bool.(MotionLimiter)

    Acceleration    = similar(Position)
    Velocity        = similar(Position)
    Kernel          = similar(Density)
    KernelGradientX = zeros(length(Density))
    KernelGradientY = zeros(length(Density))

    KernelGradient  = similar(Position)

    dρdtI           = similar(Density)


    dvdtIX = zeros(length(Density))
    dvdtIY = zeros(length(Density))
    dvdtI  = similar(Position)

    Velocityₙ⁺        = similar(Position)
    Positionₙ⁺        = similar(Position)
    ρₙ⁺               = zeros(FloatType, NumberOfPoints)
    dρdtIₙ⁺           = zeros(FloatType, NumberOfPoints)

    Pressureᵢ        = zeros(FloatType, NumberOfPoints)

    Cells                                   = similar(Position, CartesianIndex{Dimensions})
    ParticleSplitter                        = zeros(Bool, length(Cells) + 1)
    # ParticleRanges  = CUDA.zeros(Int,length(cuCells)+1) #+1 last cell to include as well, first cell is included in directly due to use of diff which reduces number of elements by 1!
    ParticleSplitter[1]   = true
    ParticleSplitter[end] = true #Have to add 1 even though it is wrong due to -1 at EndIndex, length + 1

    ParticleSplitterLinearIndices = LinearIndices(ParticleSplitter)

    SortedIndices   = similar(Cells, Int)

    Stencil         = ConstructStencil(Val(Dimensions))

    # Ensure zero, similar does not!
    ResetArrays!(Acceleration, Velocity, Kernel, KernelGradient, KernelGradientX, KernelGradientY, Cells, SortedIndices, dρdtI, dvdtIX, dvdtIY, dvdtI, Positionₙ⁺, Velocityₙ⁺)

    # Normal run and save data
    generate_showvalues(Iteration, TotalTime) = () -> [(:(Iteration),format(FormatExpr("{1:d}"),  Iteration)), (:(TotalTime),format(FormatExpr("{1:3.3f}"), TotalTime))]
    OutputCounter = 0.0
    OutputIterationCounter = 0
    @inbounds while true
        SimulationLoop(SimConstants, Cells, Stencil, SortedIndices, ParticleSplitter, ParticleSplitterLinearIndices, Position, Kernel, KernelGradient, Density, Velocity, Acceleration, dρdtI, dvdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, dρdtIₙ⁺, GravityFactor, MotionLimiter, BoundaryBool)
        
        OutputCounter += SimMetaData.CurrentTimeStep
        if OutputCounter >= SimMetaData.OutputEach
            OutputCounter = 0.0
            OutputIterationCounter += 1

            SaveLocation_ = SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(OutputIterationCounter,6,"0") * ".vtp"
            Pressure!(Pressureᵢ,Density,SimConstants)
            PolyDataTemplate(SaveLocation_, to_3d(Position), ["Kernel", "Density", "Pressure","Velocity", "Acceleration"], Kernel, Density, Pressureᵢ, Velocity, Acceleration)
        end

        next!(SimMetaData.ProgressSpecification; showvalues = generate_showvalues(SimMetaData.Iteration , SimMetaData.TotalTime))

        if SimMetaData.TotalTime >= SimMetaData.SimulationTime + 1e-3
            break
        end
    end
end

to_3d(vec_2d) = [SVector(v..., 0.0) for v in vec_2d]


Dimensions = 2
FloatType  = Float64

SimMetaData  = SimulationMetaData{Dimensions,FloatType}(
    SimulationName="Test", 
    SaveLocation="E:/SecondApproach/TESTING_CPU",
    SimulationTime=0.1,
    OutputEach=0.01,
)

SimConstantsWedge = SimulationConstants{FloatType}(c₀=42.48576250492629)

@time @profview RunSimulation(
    FluidCSV     = "./input/still_wedge_mdbc/StillWedge_Dp0.02_Fluid.csv",
    BoundCSV     = "./input/still_wedge_mdbc/StillWedge_Dp0.02_Bound.csv",
    SimMetaData  = SimMetaData,
    SimConstants = SimConstantsWedge
)