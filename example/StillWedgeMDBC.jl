using SPHExample

let
    Dimensions = 2
    FloatType  = Float64

    # Kernel = WendlandC2Kernel{Dimensions, FloatType}(0.02)
    SimConstantsWedge = SimulationConstants{FloatType}(dx=0.02,c₀=42.48576250492629, δᵩ = 0.1, CFL=0.5)
    # SimConstantsWedge = SimulationConstants{FloatType}(dx=0.01,c₀=43.4, δᵩ = 0.1, CFL=0.2)

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
# 
    SimulationGeometry = [FixedBoundary;Water]
    
    # Load in particles
    SimParticles = AllocateDataStructures(SimulationGeometry)

    SimMetaDataWedge  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="StillWedge", 
        SaveLocation="E:/SecondApproach/StillWedge2D_MDBC",
        SimulationTime=4,
        OutputEach=0.01,
        VisualizeInParaview=true,
        ExportSingleVTKHDF=true,
        ExportGridCells=false,
        OpenLogFile=true,
        FlagDensityDiffusion=true,
        FlagLinearizedDDT=true,
        FlagOutputKernelValues=false,
        FlagLog=true,
        FlagMDBCSimple=true,
    )

    SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation)

    CleanUpSimulationFolder(SimMetaDataWedge.SaveLocation)

    SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(), SimConstantsWedge.dx)

    @profview RunSimulation(
        SimGeometry         = SimulationGeometry,
        SimMetaData         = SimMetaDataWedge,
        SimConstants        = SimConstantsWedge,
        SimKernel           = SimKernel,
        SimLogger           = SimLogger,
        SimParticles        = SimParticles,
        SimViscosity        = ArtificialViscosity(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        ParticleNormalsPath = "./input/still_wedge_mdbc/StillWedge_Dp$(SimConstantsWedge.dx)_GhostNodes_Correct.csv"
    )

    # return SimParticles
end




