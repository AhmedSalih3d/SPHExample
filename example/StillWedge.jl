using SPHExample

let
    Dimensions = 2
    FloatType  = Float64

    
    SimConstantsWedge = SimulationConstants{FloatType}(dx=0.02,c₀=42.48576250492629, δᵩ = 0.1, CFL=0.2)

    # Assuming SimConstantsWedge is defined somewhere else with the field `dx`
    FixedBoundary = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Bound.csv",
        GroupMarker = 1,
        Type        = Fixed,   # Using the enum value Fixed
        Motion      = nothing
    )

    Water = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Fluid.csv",
        GroupMarker = 2,
        Type        = Fluid,   # Using the enum value Fluid
        Motion      = nothing
    )

    SimulationGeometry = [FixedBoundary;Water]

    SimMetaDataWedge  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="StillWedge", 
        SaveLocation="E:/SecondApproach/TESTING_CPU_StillWedge",
        SimulationTime=4,
        OutputEach=0.01,
        VisualizeInParaview=true,
        ExportSingleVTKHDF=true,
        ExportGridCells=true,
        OpenLogFile=true,
        FlagDensityDiffusion=true,
        FlagLinearizedDDT=true,
        FlagOutputKernelValues=false,
        FlagLog=true,
        FlagShifting=false,
    )

    SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation)

    CleanUpSimulationFolder(SimMetaDataWedge.SaveLocation)

    @profview RunSimulation(
        SimGeometry        = SimulationGeometry,
        SimMetaData        = SimMetaDataWedge,
        SimConstants       = SimConstantsWedge,
        SimLogger          = SimLogger
    )
end

