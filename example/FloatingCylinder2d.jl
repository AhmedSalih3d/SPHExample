using SPHExample
using StaticArrays

let
    Dimensions = 2
    FloatType  = Float64

    SimConstantsCylinder = SimulationConstants{FloatType}(
        dx=0.05,
        c₀=84.04284584365287,
        δᵩ = 0.1,
        CFL=0.2,
    )

    FixedBoundary = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/floating_cylinder_2d/Wall_WithWalls_Dp$(SimConstantsCylinder.dx).csv",
        GroupMarker = 1,
        Type        = Fixed,
        Motion      = nothing,
    )

    Cylinder = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/floating_cylinder_2d/Cylinder_Dp$(SimConstantsCylinder.dx).csv",
        GroupMarker = 3,
        Type        = Floating,
        Motion      = nothing,
        Mass        = 3882.75,
        COG         = SVector{Dimensions, FloatType}(0.0, 14.0),
        Inertia     = SMatrix{3, 3, FloatType}((999.714, 0, 0, 0, 1999.43, 0, 0, 0, 999.714)),
    )

    SimulationGeometry = [FixedBoundary; Cylinder]

    SimParticles = AllocateDataStructures(SimulationGeometry)

    SimMetaDataCylinder = SimulationMetaData{Dimensions, FloatType}(
        SimulationName="FloatingCylinder",
        SaveLocation="E:/SecondApproach/FloatingCylinder2d",
        SimulationTime=2.0,
        OutputTimes=0.02,
        VisualizeInParaview=true,
        ExportSingleVTKHDF=true,
        ExportGridCells=true,
        OpenLogFile=true,
        FlagOutputKernelValues=true,
        FlagLog=true,
        FlagMDBCSimple=false,
    )

    SimLogger = SimulationLogger(SimMetaDataCylinder.SaveLocation; to_console=true)
    CleanUpSimulationFolder(SimMetaDataCylinder.SaveLocation)

    SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); dx = SimConstantsCylinder.dx)

    RunSimulation(
        SimGeometry         = SimulationGeometry,
        SimMetaData         = SimMetaDataCylinder,
        SimConstants        = SimConstantsCylinder,
        SimKernel           = SimKernel,
        SimLogger           = SimLogger,
        SimParticles        = SimParticles,
        SimViscosity        = ArtificialViscosity(),
        SimDensityDiffusion = LinearDensityDiffusion()
    )

end
