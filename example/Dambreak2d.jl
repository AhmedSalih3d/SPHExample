using SPHExample

let
    Dimensions = 2
    FloatType  = Float64
    SimMetaDataDambreak  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="Test", 
        SaveLocation="E:/SecondApproach/TESTING_CPU",
        SimulationTime=1,
        OutputEach=0.01,
        FlagDensityDiffusion=true,
        FlagOutputKernelValues=false,
        FlagLog=true
    )

    SimConstantsDambreak = SimulationConstants{FloatType}(dx=0.02,c₀=88.14487860902641, δᵩ = 0.1, CFL=0.2, α = 0.02)

    SimLogger = SimulationLogger(SimMetaDataDambreak.SaveLocation)

    println(@report_opt target_modules=(@__MODULE__,) RunSimulation(
        FluidCSV           = "./input/dam_break_2d/DamBreak2d_Dp0.02_Fluid_OneLayer.csv",
        FixedCSV           = "./input/dam_break_2d/DamBreak2d_Dp0.02_Bound_ThreeLayers.csv",
        MovingCSV           = nothing,
        SimMetaData        = SimMetaDataDambreak,
        SimConstants       = SimConstantsDambreak,
        SimLogger          = SimLogger
    )
    )

    RunSimulation(
        FluidCSV           = "./input/dam_break_2d/DamBreak2d_Dp0.02_Fluid_OneLayer.csv",
        FixedCSV           = "./input/dam_break_2d/DamBreak2d_Dp0.02_Bound_ThreeLayers.csv",
        MovingCSV           = nothing,
        SimMetaData        = SimMetaDataDambreak,
        SimConstants       = SimConstantsDambreak,
        SimLogger          = SimLogger
    )
end
