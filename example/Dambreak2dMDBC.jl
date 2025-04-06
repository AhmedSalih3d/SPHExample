using SPHExample

let
    Dimensions = 2
    FloatType  = Float64

    SimConstantsDambreak = SimulationConstants{FloatType}(dx=0.02,c₀=88.14487860902641, δᵩ = 0.1, CFL=0.2, α = 0.01)

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

    SimMetaDataDambreak  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="Test", 
        SaveLocation="E:/SecondApproach/TESTING_CPU",
        SimulationTime=0.5,
        OutputEach=0.01,
        VisualizeInParaview=true,
        ExportSingleVTKHDF=true,
        OpenLogFile=true,
        FlagDensityDiffusion=true,
        FlagLinearizedDDT=true,
        FlagOutputKernelValues=false,
        FlagMDBCSimple=true,
        FlagLog=true
    )


    SimLogger = SimulationLogger(SimMetaDataDambreak.SaveLocation)

    CleanUpSimulationFolder(SimMetaDataDambreak.SaveLocation)

    @profview RunSimulation(
        SimGeometry        = SimulationGeometry,
        SimMetaData        = SimMetaDataDambreak,
        SimConstants       = SimConstantsDambreak,
        SimLogger          = SimLogger,
        SimParticles       = SimParticles,
        # path_mdbc          = "./input/dam_break_2d/DamBreak2d_Dp0.02_MDBC_GhostNodes_ThreeLayers.csv" 
        path_mdbc          = "./input/dam_break_2d/test.csv"
    )
end
