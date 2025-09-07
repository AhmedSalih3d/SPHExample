using SPHExample

let
    Dimensions = 2
    FloatType  = Float64

    SimConstantsDambreak = SimulationConstants{FloatType}(dx=0.01,c₀=88.14487860902641, δᵩ = 0.1, CFL=0.5, α = 0.01)

    # Create Geometry instances
    FixedBoundary = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/dam_break_2d/DamBreak2d_Dp0.02_MDBC_Bound_ThreeLayers.csv",
        GroupMarker = 1,
        Type        = Fixed,   # Using the enum value Fixed
        Motion      = nothing
    )

    Water = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/dam_break_2d/DamBreak2d_Dp0.02_MDBC_Fluid_ThreeLayers.csv",
        GroupMarker = 2,
        Type        = Fluid,   # Using the enum value Fluid
        Motion      = nothing
    )

    # Collect the Geometry instances into a vector
    SimulationGeometry = [FixedBoundary; Water]

    # Load in particles
    SimParticles = AllocateDataStructures(SimulationGeometry)

    SimMetaDataDambreak  = SimulationMetaData{Dimensions,FloatType,NoShifting,NoKernelOutput,SimpleMDBC,StoreLog}(
        SimulationName="DamBreak2D",
        SaveLocation="E:/SecondApproach/DamBreak2D_MDBC/",
        SimulationTime=2,
        OutputTimes=collect(0.01:0.01:2),
        VisualizeInParaview=true,
        ExportSingleVTKHDF=true,
        ExportGridCells=true,
        OpenLogFile=true
    )

    # If save directory is not already made, make it
    if !isdir(SimMetaDataDambreak.SaveLocation)
        mkdir(SimMetaDataDambreak.SaveLocation)
    end

    # How to overload and define your own viscosity model:
    # Artificial viscosity formulation.
    # using Parameters
    # using LinearAlgebra
    # struct MyTurbulenceModel <: SPHViscosity end
    # @inline function SPHExample.compute_viscosity(::MyTurbulenceModel, SimKernel, SimConstants, SimParticles, xᵢⱼ, vᵢⱼ, ∇ᵢWᵢⱼ, i, j)
    #     @unpack ρ₀, m₀, α, γ, g, c₀, δᵩ, Cb, Cb⁻¹, ν₀, dx, SmagorinskyConstant, BlinConstant = SimConstants
    #     @unpack h, η² = SimKernel

    #     dᵢⱼ =  sqrt(abs(dot(xᵢⱼ,xᵢⱼ)))
    #     ρᵢ  = SimParticles.Density[i]
    #     ρⱼ  = SimParticles.Density[j]

    #     invd²η²   =  1.0 / (dᵢⱼ*dᵢⱼ+η²)
    #     ρ̄ = (ρᵢ + ρⱼ) * 0.5
    #     cond = dot(vᵢⱼ, xᵢⱼ)
    #     flag = cond < 0 ? one(eltype(cond)) : zero(eltype(cond))
    #     μ = h * cond * invd²η²
    #     Π = -m₀ * (flag * (-α * c₀ * μ) / ρ̄) * ∇ᵢWᵢⱼ
    #     return 0*Π, -Π*0
    # end

    SimLogger = SimulationLogger(SimMetaDataDambreak.SaveLocation; to_console=true)

    CleanUpSimulationFolder(SimMetaDataDambreak.SaveLocation)

    @profview RunSimulation(
        SimGeometry          = SimulationGeometry,
        SimMetaData          = SimMetaDataDambreak,
        SimConstants         = SimConstantsDambreak,
        SimKernel            = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); dx = SimConstantsDambreak.dx),
        SimLogger            = SimLogger,
        SimParticles         = SimParticles,
        SimViscosity         = ArtificialViscosity(),
        SimDensityDiffusion  = LinearDensityDiffusion(),
        ParticleNormalsPath  = "./input/dam_break_2d/DamBreak2d_Dp0.02_MDBC_GhostNodes_ThreeLayers.csv"
    )
end
