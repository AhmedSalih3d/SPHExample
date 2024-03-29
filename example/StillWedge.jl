using SPHExample
using StaticArrays
import StructArrays: StructArray
import LinearAlgebra: dot, norm, diagm, diag, cond, det
import Parameters: @unpack
import FastPow: @fastpow
import ProgressMeter: next!, finish!
using Format
using TimerOutputs

# Really important to overload default function, gives 10x speed up?
# Overload the default function to do what you please
function ComputeInteractions!(SimMetaData, SimConstants, Position, KernelThreaded, KernelGradientThreaded, Density, Pressure, Velocity, dρdtI, dvdtI, i, j, MotionLimiter, ichunk)
    @unpack FlagViscosityTreatment, FlagDensityDiffusion, FlagOutputKernelValues = SimMetaData
    @unpack ρ₀, h, h⁻¹, m₀, αD, α, g, c₀, δᵩ, η², H², Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants

    xᵢⱼ  = Position[i] - Position[j]
    xᵢⱼ² = dot(xᵢⱼ,xᵢⱼ)              
    if  xᵢⱼ² <= H²
        #https://discourse.julialang.org/t/sqrt-abs-x-is-even-faster-than-sqrt/58154/2
        dᵢⱼ  = sqrt(abs(xᵢⱼ²))

        q         = min(dᵢⱼ * h⁻¹, 2.0)
        invd²η²   = inv(dᵢⱼ*dᵢⱼ+η²)
        ∇ᵢWᵢⱼ     = @fastpow (αD*5*(q-2)^3*q / (8h*(q*h+η²)) ) * xᵢⱼ 
        ρᵢ        = Density[i]
        ρⱼ        = Density[j]
    
        vᵢ        = Velocity[i]
        vⱼ        = Velocity[j]
        vᵢⱼ       = vᵢ - vⱼ
        density_symmetric_term = dot(-vᵢⱼ, ∇ᵢWᵢⱼ)
        dρdt⁺          = - ρᵢ * (m₀/ρⱼ) *  density_symmetric_term
        dρdt⁻          = - ρⱼ * (m₀/ρᵢ) *  density_symmetric_term

        # Density diffusion
        if FlagDensityDiffusion
            Pᵢⱼᴴ  = ρ₀ * (-g) * -xᵢⱼ[end]
            ρᵢⱼᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pᵢⱼᴴ, Cb⁻¹)
            Pⱼᵢᴴ  = -Pᵢⱼᴴ
            ρⱼᵢᴴ  = InverseHydrostaticEquationOfState(ρ₀, Pⱼᵢᴴ, Cb⁻¹)
        
            ρⱼᵢ   = ρⱼ - ρᵢ
            MLcond = MotionLimiter[i] * MotionLimiter[j]
            ddt_symmetric_term =  δᵩ * h * c₀ * 2 * invd²η² * dot(-xᵢⱼ,  ∇ᵢWᵢⱼ) * MLcond
            Dᵢ  = ddt_symmetric_term * (m₀/ρⱼ) * ( ρⱼᵢ - ρᵢⱼᴴ)
            Dⱼ  = ddt_symmetric_term * (m₀/ρᵢ) * (-ρⱼᵢ - ρⱼᵢᴴ)
        else
            Dᵢ  = 0.0
            Dⱼ  = 0.0
        end
        dρdtI[i] += dρdt⁺ + Dᵢ
        dρdtI[j] += dρdt⁻ + Dⱼ


        Pᵢ      =  Pressure[i]
        Pⱼ      =  Pressure[j]
        Pfac    = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)
        dvdt⁺   = - m₀ * Pfac *  ∇ᵢWᵢⱼ
        dvdt⁻   = - dvdt⁺

        if FlagViscosityTreatment == :ArtificialViscosity
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
    
        if FlagViscosityTreatment == :Laminar || FlagViscosityTreatment == :LaminarSPS
            # 4 comes from 2 divided by 0.5 from average density
            # should divide by ρᵢ eq 6 DPC
            # ν₀∇²uᵢ = (1/ρᵢ) * ( (4 * m₀ * (ρᵢ * ν₀) * dot( xᵢⱼ, ∇ᵢWᵢⱼ)  ) / ( (ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²) ) ) *  vᵢⱼ
            # ν₀∇²uⱼ = (1/ρⱼ) * ( (4 * m₀ * (ρⱼ * ν₀) * dot(-xᵢⱼ,-∇ᵢWᵢⱼ)  ) / ( (ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²) ) ) * -vᵢⱼ
            visc_symmetric_term = (4 * m₀ * ν₀ * dot( xᵢⱼ, ∇ᵢWᵢⱼ)) / ((ρᵢ + ρⱼ) + (dᵢⱼ * dᵢⱼ + η²))
            # ν₀∇²uᵢ = (1/ρᵢ) * visc_symmetric_term *  vᵢⱼ * ρᵢ
            # ν₀∇²uⱼ = (1/ρⱼ) * visc_symmetric_term * -vᵢⱼ * ρⱼ
            ν₀∇²uᵢ =  visc_symmetric_term *  vᵢⱼ
            ν₀∇²uⱼ = -ν₀∇²uᵢ #visc_symmetric_term * -vᵢⱼ
        else
            ν₀∇²uᵢ = zero(xᵢⱼ)
            ν₀∇²uⱼ = ν₀∇²uᵢ
        end
    
        if FlagViscosityTreatment == :LaminarSPS 
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
    
            # MATHEMATICALLY THIS IS DOT PRODUCT TO GO FROM TENSOR TO VECTOR, BUT USE * IN JULIA TO REPRESENT IT
            dτdtᵢ = (m₀/(ρⱼ * ρᵢ)) * (τᶿᵢ + τᶿⱼ) *  ∇ᵢWᵢⱼ 
            dτdtⱼ = (m₀/(ρᵢ * ρⱼ)) * (τᶿᵢ + τᶿⱼ) * -∇ᵢWᵢⱼ 
        else
            dτdtᵢ  = zero(xᵢⱼ)
            dτdtⱼ  = dτdtᵢ
        end
    
        dvdtI[i] += dvdt⁺ + Πᵢ + ν₀∇²uᵢ + dτdtᵢ
        dvdtI[j] += dvdt⁻ + Πⱼ + ν₀∇²uⱼ + dτdtⱼ

        if FlagOutputKernelValues
            Wᵢⱼ  = @fastpow αD*(1-q/2)^4*(2*q + 1)
            KernelThreaded[ichunk][i]         += Wᵢⱼ
            KernelThreaded[ichunk][j]         += Wᵢⱼ
            KernelGradientThreaded[ichunk][i] +=  ∇ᵢWᵢⱼ
            KernelGradientThreaded[ichunk][j] += -∇ᵢWᵢⱼ
        end
    end

    return nothing
end

@inbounds function SimulationLoop(ComputeInteractions!, SimMetaData, SimConstants, SimParticles, Stencil,  ParticleRanges, UniqueCells, SortingScratchSpace, Kernel, KernelThreaded, KernelGradient, KernelGradientThreaded, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺)
    Position      = SimParticles.Position
    Density       = SimParticles.Density
    Pressure      = SimParticles.Pressure
    Velocity      = SimParticles.Velocity
    Acceleration  = SimParticles.Acceleration
    GravityFactor = SimParticles.GravityFactor
    MotionLimiter = SimParticles.MotionLimiter

    @timeit SimMetaData.HourGlass "01 Update TimeStep"  dt  = Δt(Position, Velocity, Acceleration, SimConstants)
    dt₂ = dt * 0.5

    @timeit SimMetaData.HourGlass "02 Calculate IndexCounter" IndexCounter = UpdateNeighbors!(SimParticles, SimConstants.H*2, SortingScratchSpace,  ParticleRanges, UniqueCells)

    @timeit SimMetaData.HourGlass "03 ResetArrays"                           ResetArrays!(Kernel, KernelGradient, dρdtI, Acceleration); ResetArrays!.(KernelThreaded, KernelGradientThreaded)

    Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)
    @timeit SimMetaData.HourGlass "04 First NeighborLoop"                    NeighborLoop!(ComputeInteractions!, SimMetaData, SimConstants, ParticleRanges, Stencil, Position, KernelThreaded, KernelGradientThreaded, Density, Pressure, Velocity, dρdtI, Acceleration,  MotionLimiter, UniqueCells, IndexCounter)

    @timeit SimMetaData.HourGlass "05 Update To Half TimeStep" @inbounds for i in eachindex(Position)
        Acceleration[i]  +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
        Velocityₙ⁺[i]     =  Velocity[i]   + Acceleration[i]  *  dt₂ * MotionLimiter[i]
        Positionₙ⁺[i]     =  Position[i]   + Velocityₙ⁺[i]   * dt₂  * MotionLimiter[i]
        ρₙ⁺[i]            =  Density[i]    + dρdtI[i]       *  dt₂
    end

    @timeit SimMetaData.HourGlass "06 Half LimitDensityAtBoundary"  LimitDensityAtBoundary!(ρₙ⁺, SimConstants.ρ₀, MotionLimiter)

    @timeit SimMetaData.HourGlass "07 ResetArrays"                  ResetArrays!(Kernel, KernelGradient, dρdtI, Acceleration); ResetArrays!.(KernelThreaded, KernelGradientThreaded)

    Pressure!(SimParticles.Pressure, ρₙ⁺,SimConstants)
    @timeit SimMetaData.HourGlass "08 Second NeighborLoop"          NeighborLoop!(ComputeInteractions!, SimMetaData, SimConstants, ParticleRanges, Stencil, Positionₙ⁺, KernelThreaded, KernelGradientThreaded, ρₙ⁺, Pressure, Velocityₙ⁺, dρdtI, Acceleration, MotionLimiter, UniqueCells, IndexCounter)

    @timeit SimMetaData.HourGlass "09 Final Density"                DensityEpsi!(Density, dρdtI, ρₙ⁺, dt)

    @timeit SimMetaData.HourGlass "10 Final LimitDensityAtBoundary" LimitDensityAtBoundary!(Density, SimConstants.ρ₀, MotionLimiter)

    @timeit SimMetaData.HourGlass "11 Update To Final TimeStep"  @inbounds for i in eachindex(Position)
        Acceleration[i]   +=  ConstructGravitySVector(Acceleration[i], SimConstants.g * GravityFactor[i])
        Velocity[i]       +=  Acceleration[i] * dt * MotionLimiter[i]
        Position[i]       +=  (((Velocity[i] + (Velocity[i] - Acceleration[i] * dt * MotionLimiter[i])) / 2) * dt) * MotionLimiter[i]
    end

    SimMetaData.Iteration      += 1
    SimMetaData.CurrentTimeStep = dt
    SimMetaData.TotalTime      += dt

    if SimMetaData.FlagOutputKernelValues
        Kernel         .= sum(KernelThreaded)
        KernelGradient .= sum(KernelGradientThreaded)
    end
end

###===

function RunSimulation(;FluidCSV::String,
    BoundCSV::String,
    SimMetaData::SimulationMetaData{Dimensions, FloatType},
    SimConstants::SimulationConstants
    ) where {Dimensions,FloatType}


    # If save directory is not already made, make it
    if !isdir(SimMetaData.SaveLocation)
        mkdir(SimMetaData.SaveLocation)
    end
    
    # Delete previous result files
    foreach(rm, filter(endswith(".vtp"), readdir(SimMetaData.SaveLocation,join=true)))

    # Unpack the relevant simulation meta data
    @unpack HourGlass, SaveLocation, SimulationName, SilentOutput, ThreadsCPU = SimMetaData;

    # Load in particles
    SimParticles, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, Kernel, KernelGradient = AllocateDataStructures(Dimensions,FloatType, FluidCSV,BoundCSV)
    Pressure!(SimParticles.Pressure,SimParticles.Density,SimConstants)

    KernelThreaded         = [copy(Kernel)         for _ in 1:Base.Threads.nthreads()]
    KernelGradientThreaded = [copy(KernelGradient) for _ in 1:Base.Threads.nthreads()]

    # Produce sorting related variables
    ParticleRanges = zeros(Int, length(SimParticles) + 1)
    UniqueCells    = zeros(CartesianIndex{Dimensions}, length(SimParticles))
    Stencil        = ConstructStencil(Val(Dimensions))
    _, SortingScratchSpace = Base.Sort.make_scratch(nothing, eltype(SimParticles), length(SimParticles))

    # Produce data saving functions
    SaveLocation_ = SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(0,6,"0") * ".vtp"
    SaveFile = (SaveLocation_) -> ExportVTP(SaveLocation_, to_3d(SimParticles.Position), ["Kernel", "KernelGradient", "Density", "Pressure","Velocity", "Acceleration", "BoundaryBool" , "ID"], Kernel, KernelGradient, SimParticles.Density, SimParticles.Pressure, SimParticles.Velocity, SimParticles.Acceleration, Int.(SimParticles.BoundaryBool), SimParticles.ID)
    SaveFile(SaveLocation_)


    # Normal run and save data
    generate_showvalues(Iteration, TotalTime) = () -> [(:(Iteration),format(FormatExpr("{1:d}"),  Iteration)), (:(TotalTime),format(FormatExpr("{1:3.3f}"), TotalTime))]
    OutputCounter = 0.0
    OutputIterationCounter = 0
    @inbounds while true

        SimulationLoop(ComputeInteractions!, SimMetaData, SimConstants, SimParticles, Stencil, ParticleRanges, UniqueCells, SortingScratchSpace, Kernel, KernelThreaded, KernelGradient, KernelGradientThreaded, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺)

        OutputCounter += SimMetaData.CurrentTimeStep
        if OutputCounter >= SimMetaData.OutputEach
            OutputCounter = 0.0
            OutputIterationCounter += 1

            SaveLocation_ = SimMetaData.SaveLocation * "/" * SimulationName * "_" * lpad(OutputIterationCounter,6,"0") * ".vtp"
            @timeit HourGlass "12 Output Data"  SaveFile(SaveLocation_)
        end

        if !SilentOutput
            @timeit HourGlass "13 Next TimeStep" next!(SimMetaData.ProgressSpecification; showvalues = generate_showvalues(SimMetaData.Iteration , SimMetaData.TotalTime))
        end

        if SimMetaData.TotalTime > SimMetaData.SimulationTime
            finish!(SimMetaData.ProgressSpecification)
            show(HourGlass,sortby=:name)
            show(HourGlass)
            break
        end
    end
end

let
    Dimensions = 2
    FloatType  = Float64

    SimMetaDataWedge  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="Test", 
        SaveLocation="E:/SecondApproach/TESTING_CPU",
        SimulationTime=1,
        OutputEach=0.01,
        FlagDensityDiffusion=true,
        FlagOutputKernelValues=true,
    )

    SimConstantsWedge = SimulationConstants{FloatType}(dx=0.02,c₀=42.48576250492629, δᵩ = 1, CFL=0.2)

    # Remove '@profview' if you do not want VS Code timers
    @profview RunSimulation(
        FluidCSV           = "./input/still_wedge/StillWedge_Dp0.02_Fluid.csv",
        BoundCSV           = "./input/still_wedge/StillWedge_Dp0.02_Bound.csv",
        SimMetaData        = SimMetaDataWedge,
        SimConstants       = SimConstantsWedge
    )
end