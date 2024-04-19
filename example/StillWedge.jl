using SPHExample

let
    Dimensions = 2
    FloatType  = Float64

    # Let us define the 

    SimMetaDataWedge  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="StillWedge", 
        SaveLocation="E:/SecondApproach/TESTING_CPU_StillWedge",
        SimulationTime=1,
        OutputEach=0.01,
        FlagDensityDiffusion=false,
        FlagOutputKernelValues=false,
        FlagLog=true,
        FlagShifting=false,
    )

    SimConstantsWedge = SimulationConstants{FloatType}(dx=0.02,c₀=42.48576250492629, δᵩ = 0.1, CFL=0.2)

    SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation)

    println(@report_opt target_modules=(@__MODULE__,) RunSimulation(
        FluidCSV           = "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Fluid.csv",
        FixedCSV           = "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Bound.csv",
        MovingCSV           = nothing,
        SimMetaData        = SimMetaDataWedge,
        SimConstants       = SimConstantsWedge,
        SimLogger          = SimLogger
    )
    )

    RunSimulation(
        FluidCSV           = "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Fluid.csv",
        FixedCSV           = "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Bound.csv",
        MovingCSV           = nothing,
        SimMetaData        = SimMetaDataWedge,
        SimConstants       = SimConstantsWedge,
        SimLogger          = SimLogger
    )
end
