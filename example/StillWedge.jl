using SPHExample
using BenchmarkTools
using StaticArrays
using Parameters
using StructArrays
import LinearAlgebra: dot, norm, diagm, diag, cond, det
using LoopVectorization
using FastPow
import CellListMap: InPlaceNeighborList, update!, neighborlist!
import ProgressMeter: next!, finish!
using Formatting
using Bumper
using TimerOutputs
using Distances



# Really important to overload default function, gives 10x speed up?
# Overload the default function to do what you please
function SPHExample.ComputeInteractions!(SimConstants, Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI, i, j, MotionLimiter, ViscosityTreatment, BoolDDT, OutputKernelValues)
    @unpack ρ₀, h, h⁻¹, m₀, αD, α, g, c₀, δᵩ, η², H², Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants

    xᵢⱼ² = evaluate(SqEuclidean(), Position[i], Position[j])
    if  xᵢⱼ² <= H²
        xᵢⱼ  = Position[i] - Position[j]
        
        dᵢⱼ  = sqrt(xᵢⱼ²) #Using sqrt is what takes a lot of time?
        # Unsure what is faster, min should do less operations?
        q         = min(dᵢⱼ * h⁻¹, 2.0) #clamp(dᵢⱼ * h⁻¹,0.0,2.0)
        invd²η²   = inv(dᵢⱼ*dᵢⱼ+η²)
        ∇ᵢWᵢⱼ     = @fastpow (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 
        ρᵢ        = Density[i]
        ρⱼ        = Density[j]
    
        vᵢ        = Velocity[i]
        vⱼ        = Velocity[j]
        vᵢⱼ       = vᵢ - vⱼ
        density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ) # = dot(vᵢⱼ , -∇ᵢWᵢⱼ)
        dρdt⁺          = - ρᵢ * (m₀/ρⱼ) *  density_symmetric_term
        dρdt⁻          = - ρⱼ * (m₀/ρᵢ) *  density_symmetric_term

        # Density diffusion
        if BoolDDT
            Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
            ρᵢⱼᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pᵢⱼᴴ, Cb⁻¹)
            Pⱼᵢᴴ  = -Pᵢⱼᴴ
            ρⱼᵢᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pⱼᵢᴴ, Cb⁻¹)
        
            ρⱼᵢ   = ρⱼ - ρᵢ
            MLcond = MotionLimiter[i] * MotionLimiter[j]
            ddt_symmetric_term =  δᵩ * h * c₀ * 2 * invd²η² * dot(-xᵢⱼ,  ∇ᵢWᵢⱼ) * MLcond #  dot(-xᵢⱼ,  ∇ᵢWᵢⱼ) =  dot( xᵢⱼ, -∇ᵢWᵢⱼ)
            Dᵢ  = ddt_symmetric_term * (m₀/ρⱼ) * ( ρⱼᵢ - ρᵢⱼᴴ)
            Dⱼ  = ddt_symmetric_term * (m₀/ρᵢ) * (-ρⱼᵢ - ρⱼᵢᴴ)
        else
            Dᵢ  = 0.0
            Dⱼ  = 0.0
        end
        dρdtI[i] += dρdt⁺ + Dᵢ
        dρdtI[j] += dρdt⁻ + Dⱼ

        Pᵢ      =  EquationOfStateGamma7(ρᵢ,c₀,ρ₀)
        Pⱼ      =  EquationOfStateGamma7(ρⱼ,c₀,ρ₀)
        Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
        dvdt⁺   = - m₀ * Pfac *  ∇ᵢWᵢⱼ
        dvdt⁻   = - dvdt⁺

        if ViscosityTreatment == :ArtificialViscosity
            ρ̄ᵢⱼ       = (ρᵢ+ρⱼ)*0.5
            cond      = dot(vᵢⱼ, xᵢⱼ)
            cond_bool = cond < 0.0
            μᵢⱼ       = h*cond * invd²η²
            Πᵢ        = - m₀ * (cond_bool*(-α*c₀*μᵢⱼ)/ρ̄ᵢⱼ) * ∇ᵢWᵢⱼ
            Πⱼ        = - Πᵢ
        else
            Πᵢ        = zero(xᵢⱼ)
            Πⱼ        = Πᵢ
        end
    
        if ViscosityTreatment == :Laminar || ViscosityTreatment == :LaminarSPS
            # 4 comes from 2 divided by 0.5 from average density
            # should divide by ρᵢ eq 6 DPC
            ν₀∇²uᵢ = (1/ρᵢ) * ( (4 * m₀ * (ρᵢ * ν₀) * dot( xᵢⱼ, ∇ᵢWᵢⱼ)  ) / ( (ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²) ) ) *  vᵢⱼ
            ν₀∇²uⱼ = (1/ρⱼ) * ( (4 * m₀ * (ρⱼ * ν₀) * dot(-xᵢⱼ,-∇ᵢWᵢⱼ)  ) / ( (ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²) ) ) * -vᵢⱼ
        else
            ν₀∇²uᵢ = zero(xᵢⱼ)
            ν₀∇²uⱼ = ν₀∇²uᵢ
        end
    
        if ViscosityTreatment == :LaminarSPS 
            Iᴹ       = diagm(one.(xᵢⱼ))
            #julia> a .- a'
            # 3×3 SMatrix{3, 3, Float64, 9} with indices SOneTo(3)×SOneTo(3):
            # 0.0  0.0  0.0
            # 0.0  0.0  0.0
            # 0.0  0.0  0.0
            # Strain *rate* tensor is the gradient of velocity
            Sᵢ = ∇vᵢ =  (m₀/ρⱼ) * (vⱼ - vᵢ) * ∇ᵢWᵢⱼ'
            norm_Sᵢ  = sqrt(2 * sum(Sᵢ .^ 2))
            νtᵢ      = (SmagorinskyConstant * dx)^2 * norm_Sᵢ
            trace_Sᵢ = sum(diag(Sᵢ))
            τᶿᵢ      = 2*νtᵢ*ρᵢ * (Sᵢ - (1/3) * trace_Sᵢ * Iᴹ) - (2/3) * ρᵢ * BlinConstant * dx^2 * norm_Sᵢ^2 * Iᴹ
            Sⱼ = ∇vⱼ =  (m₀/ρᵢ) * (vᵢ - vⱼ) * -∇ᵢWᵢⱼ'
            norm_Sⱼ  = sqrt(2 * sum(Sⱼ .^ 2))
            νtⱼ      = (SmagorinskyConstant * dx)^2 * norm_Sⱼ
            trace_Sⱼ = sum(diag(Sⱼ))
            τᶿⱼ      = 2*νtⱼ*ρⱼ * (Sⱼ - (1/3) * trace_Sⱼ * Iᴹ) - (2/3) * ρⱼ * BlinConstant * dx^2 * norm_Sⱼ^2 * Iᴹ
    
            
            dτdtᵢ = (m₀/(ρⱼ * ρᵢ)) * (τᶿᵢ + τᶿⱼ) *  ∇ᵢWᵢⱼ # MATHEMATICALLY THIS IS DOT PRODUCT TO GO FROM TENSOR TO VECTOR, BUT USE * IN JULIA THIS TIME
            dτdtⱼ = (m₀/(ρᵢ * ρⱼ)) * (τᶿᵢ + τᶿⱼ) * -∇ᵢWᵢⱼ # MATHEMATICALLY THIS IS DOT PRODUCT TO GO FROM TENSOR TO VECTOR, BUT USE * IN JULIA THIS TIME
        else
            dτdtᵢ  = zero(xᵢⱼ)
            dτdtⱼ  = dτdtᵢ
        end
    
        dvdtI[i] += dvdt⁺ + Πᵢ + ν₀∇²uᵢ + dτdtᵢ
        dvdtI[j] += dvdt⁻ + Πⱼ + ν₀∇²uⱼ + dτdtⱼ

        if OutputKernelValues
            Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)
            Kernel[i]         += Wᵢⱼ
            Kernel[j]         += Wᵢⱼ
            KernelGradient[i] +=  ∇ᵢWᵢⱼ
            KernelGradient[j] += -∇ᵢWᵢⱼ
        end
    end

    return nothing
end

function SimulationLoop(SimMetaData, SimConstants, Cells, Stencil,  ParticleRanges, UniqueCells, SortedIndices, SortingScratchSpace, Position, Kernel, KernelGradient, Density, Velocity, Acceleration, dρdtI, dvdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, dρdtIₙ⁺, GravityFactor, MotionLimiter, ViscosityTreatment, BoolDDT, OutputKernelValues)
    @timeit SimMetaData.HourGlass "01 Update TimeStep"  dt  = Δt(Position, Velocity, Acceleration, SimConstants)
    dt₂ = dt * 0.5

    ResetArrays!(ParticleRanges)
    @timeit SimMetaData.HourGlass "02 Calculate IndexCounter" IndexCounter = UpdateNeighbors!(Cells, SimConstants.H*2, SortedIndices, SortingScratchSpace, Position, Density, Acceleration, Velocity, GravityFactor, MotionLimiter, ParticleRanges, UniqueCells)

    @timeit SimMetaData.HourGlass "03 ResetArrays"                           ResetArrays!(Kernel, KernelGradient, dρdtI, dvdtI)

    @timeit SimMetaData.HourGlass "04 First NeighborLoop"                    NeighborLoop!(SimConstants, ParticleRanges, Stencil, Position, Kernel, KernelGradient, Density, Velocity, dρdtI, dvdtI,  MotionLimiter, UniqueCells, IndexCounter, ViscosityTreatment, BoolDDT, OutputKernelValues)

    @timeit SimMetaData.HourGlass "05 Update To Half TimeStep" @inbounds for i in eachindex(Position)
        dvdtI[i]        +=  ConstructGravitySVector(dvdtI[i], SimConstants.g * GravityFactor[i])
        Velocityₙ⁺[i]    =  Velocity[i]   + dvdtI[i]  *  dt₂ * MotionLimiter[i]
        Positionₙ⁺[i]    =  Position[i]   + Velocityₙ⁺[i]   * dt₂  * MotionLimiter[i]
        ρₙ⁺[i]           =  Density[i]    + dρdtI[i]       *  dt₂
    end

    @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary"  LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, MotionLimiter)

    @timeit SimMetaData.HourGlass "07 ResetArrays"                  ResetArrays!(Kernel, KernelGradient, dρdtI, dρdtIₙ⁺, Acceleration)

    @timeit SimMetaData.HourGlass "08 Second NeighborLoop"          NeighborLoop!(SimConstants, ParticleRanges, Stencil, Positionₙ⁺, Kernel, KernelGradient, ρₙ⁺, Velocityₙ⁺, dρdtIₙ⁺, Acceleration, MotionLimiter, UniqueCells, IndexCounter, ViscosityTreatment, BoolDDT, OutputKernelValues)

    @timeit SimMetaData.HourGlass "09 Final Density"                DensityEpsi!(Density, dρdtIₙ⁺, ρₙ⁺, dt)

    @timeit SimMetaData.HourGlass "10 Final LimitDensityAtBoundary" LimitDensityAtBoundary!(Density, SimConstants.ρ₀, MotionLimiter)

    @timeit SimMetaData.HourGlass "11 Update To Final TimeStep"  @inbounds for i in eachindex(Position)
        Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
        Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
        Position[i]       +=  (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt) * MotionLimiter[i]
    end

    SimMetaData.Iteration      += 1
    SimMetaData.CurrentTimeStep = dt
    SimMetaData.TotalTime      += dt
end

###===

function RunSimulation(;FluidCSV::String,
    BoundCSV::String,
    SimMetaData::SimulationMetaData{Dimensions, FloatType},
    SimConstants::SimulationConstants,
    ViscosityTreatment = :LaminarSPS,
    BoolDDT = true,
    OutputKernelValues = false
    ) where {Dimensions,FloatType}

    if ViscosityTreatment ∉ Set((:None, :ArtificialViscosity, :Laminar, :LaminarSPS))
        error("ViscosityTreatment must be either :None, :ArtificialViscosity, :Laminar, :LaminarSPS")
    end

    if !isdir(SimMetaData.SaveLocation)
        mkdir(SimMetaData.SaveLocation)
    end
    
    foreach(rm, filter(endswith(".vtp"), readdir(SimMetaData.SaveLocation,join=true)))

    # Unpack the relevant simulation meta data
    @unpack HourGlass, SaveLocation, SimulationName, SilentOutput, ThreadsCPU = SimMetaData;

    # Load in the fluid and boundary particles. Return these points and both data frames
    # @inline is a hack here to remove all allocations further down due to uncertainty of the points type at compile time
    @inline Position, density_fluid, density_bound  = LoadParticlesFromCSV_StaticArrays(Dimensions,FloatType, FluidCSV,BoundCSV)
    NumberOfPoints = length(Position)
    Density  = deepcopy([density_bound; density_fluid])

    GravityFactor = [ zeros(size(density_bound,1)) ; -ones(size(density_fluid,1)) ]
    
    MotionLimiter = [ zeros(size(density_bound,1)) ;  ones(size(density_fluid,1)) ]

    Acceleration    = similar(Position)
    Velocity        = similar(Position)
    Kernel          = similar(Density)
    KernelGradient  = similar(Position)

    dρdtI           = similar(Density)

    dvdtI           = similar(Position)

    Velocityₙ⁺      = similar(Position)
    Positionₙ⁺      = similar(Position)
    ρₙ⁺             = zeros(FloatType, NumberOfPoints)
    dρdtIₙ⁺         = zeros(FloatType, NumberOfPoints)

    Pressureᵢ      = zeros(FloatType, NumberOfPoints)
    
    Cells          = similar(Position, CartesianIndex{Dimensions})

    ParticleRanges         = zeros(Int, length(Cells) + 1)
    UniqueCells            = zeros(CartesianIndex{Dimensions}, length(Cells))

    SortedIndices          = similar(Cells, Int)
    _, SortingScratchSpace = Base.Sort.make_scratch(nothing, eltype(SortedIndices), length(Cells))

    Stencil                = ConstructStencil(Val(Dimensions))

    # Ensure zero, similar does not!
    ResetArrays!(Acceleration, Velocity, Kernel, KernelGradient, Cells, SortedIndices, dρdtI, dvdtI, Positionₙ⁺, Velocityₙ⁺)
    Pressure!(Pressureᵢ,Density,SimConstants)

    SaveLocation_ = SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(0,6,"0") * ".vtp"
    SaveFile = (SaveLocation_) -> ExportVTP(SaveLocation_, to_3d(Position), ["Kernel", "KernelGradient", "Density", "Pressure","Velocity", "Acceleration"], Kernel, KernelGradient, Density, Pressureᵢ, Velocity, Acceleration)
    SaveFile(SaveLocation_)


    # Normal run and save data
    generate_showvalues(Iteration, TotalTime) = () -> [(:(Iteration),format(FormatExpr("{1:d}"),  Iteration)), (:(TotalTime),format(FormatExpr("{1:3.3f}"), TotalTime))]
    OutputCounter = 0.0
    OutputIterationCounter = 0
    @inbounds while true

        SimulationLoop(SimMetaData, SimConstants, Cells, Stencil, ParticleRanges, UniqueCells, SortedIndices, SortingScratchSpace, Position, Kernel, KernelGradient, Density, Velocity, Acceleration, dρdtI, dvdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, dρdtIₙ⁺, GravityFactor, MotionLimiter, ViscosityTreatment, BoolDDT, OutputKernelValues)

        OutputCounter += SimMetaData.CurrentTimeStep
        if OutputCounter >= SimMetaData.OutputEach
            OutputCounter = 0.0
            OutputIterationCounter += 1

            SaveLocation_ = SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(OutputIterationCounter,6,"0") * ".vtp"
            Pressure!(Pressureᵢ,Density,SimConstants)
            @timeit HourGlass "12 Output Data"  SaveFile(SaveLocation_)
        end
        @timeit HourGlass "13 Next TimeStep"    next!(SimMetaData.ProgressSpecification; showvalues = generate_showvalues(SimMetaData.Iteration , SimMetaData.TotalTime))

        if SimMetaData.TotalTime > SimMetaData.SimulationTime
            break
        end
    end
    finish!(SimMetaData.ProgressSpecification)
    show(HourGlass,sortby=:name)
    show(HourGlass)
end

let
    Dimensions = 2
    FloatType  = Float64

    SimMetaData  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="Test", 
        SaveLocation="E:/SecondApproach/TESTING_CPU",
        SimulationTime=0,
        OutputEach=0.01,
    )

    SimConstantsWedge = SimulationConstants{FloatType}(dx=0.01,c₀=42.48576250492629, δᵩ = 0.1, CFL=0.2)

    @profview RunSimulation(
        FluidCSV           = "./input/still_wedge_mdbc/StillWedge_Dp0.01_Fluid.csv",
        BoundCSV           = "./input/still_wedge_mdbc/StillWedge_Dp0.01_Bound.csv",
        SimMetaData        = SimMetaData,
        SimConstants       = SimConstantsWedge,
        ViscosityTreatment = :ArtificialViscosity,
        BoolDDT            = true,
        OutputKernelValues = false,
    )

    SimMetaData  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="Test", 
        SaveLocation="E:/SecondApproach/TESTING_CPU",
        SimulationTime=1,
        OutputEach=0.01,
    )

    SimConstantsWedge = SimulationConstants{FloatType}(dx=0.02,c₀=42.48576250492629, δᵩ = 0.1, CFL=0.2)

    @profview RunSimulation(
        FluidCSV           = "./input/still_wedge_mdbc/StillWedge_Dp0.02_Fluid.csv",
        BoundCSV           = "./input/still_wedge_mdbc/StillWedge_Dp0.02_Bound.csv",
        SimMetaData        = SimMetaData,
        SimConstants       = SimConstantsWedge,
        ViscosityTreatment = :ArtificialViscosity,
        BoolDDT            = true,
        OutputKernelValues = false,
    )

    # SimMetaData  = SimulationMetaData{Dimensions,FloatType}(
    #     SimulationName="Test", 
    #     SaveLocation="E:/SecondApproach/TESTING_CPU",
    #     SimulationTime=4,
    #     OutputEach=0.01,
    # )
    # SimConstantsDamBreak = SimulationConstants{FloatType}()
    # @profview RunSimulation(
    #     FluidCSV     = "./input/FluidPoints_Dp0.02_5LAYERS.csv",
    #     BoundCSV     = "./input/BoundaryPoints_Dp0.02_5LAYERS.csv",
    #     SimMetaData  = SimMetaData,
    #     SimConstants = SimConstantsDamBreak,
    #     BoolDDT      = true,
    #     ViscosityTreatment = :ArtificialViscosity
    # )
end