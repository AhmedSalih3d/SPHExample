using SPHExample
using StaticArrays

let
    Dimensions = 2
    FloatType  = Float64

    SimConstantsFloatingCylinder = SimulationConstants{FloatType}(
        dx=0.05,
        c₀=84.04284584365287,
        δᵩ = 0.1,
        CFL=0.2,
        k = 1.69706,
    )

    FixedBoundary = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/floating_cylinder_2d/Wall_WithWalls_Dp$(SimConstantsFloatingCylinder.dx).csv",
        GroupMarker = 1,
        Type        = Fixed,
        Motion      = nothing,
    )

    Cylinder = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/floating_cylinder_2d/Cylinder_Dp$(SimConstantsFloatingCylinder.dx).csv",
        GroupMarker = 3,
        Type        = Floating,
        Motion      = nothing,
        Mass        = 3882.75,
        COG         = SVector{Dimensions, FloatType}(0.0, 14.0),
        Inertia     = SMatrix{3, 3, FloatType}((999.714, 0, 0, 0, 1999.43, 0, 0, 0, 999.714)),
    )

    SimulationGeometry = [FixedBoundary; Cylinder]

    SimParticles = AllocateDataStructures(SimulationGeometry)

    SimMetaDataFloatingCylinder = SimulationMetaData{Dimensions, FloatType}(
        SimulationName="FloatingCylinder",
        SaveLocation="./output/FloatingCylinder2d",
        SimulationTime=2.0,
        OutputEach=0.02,
        FlagDensityDiffusion=true,
        FlagOutputKernelValues=true,
        FlagLog=true,
        FlagShifting=false,
    )

    SimLogger = SimulationLogger(SimMetaDataFloatingCylinder.SaveLocation)
    CleanUpSimulationFolder(SimMetaDataFloatingCylinder.SaveLocation)

    SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); dx = SimConstantsFloatingCylinder.dx)

    RunSimulation(
        SimGeometry         = SimulationGeometry,
        SimMetaData         = SimMetaDataFloatingCylinder,
        SimConstants        = SimConstantsFloatingCylinder,
        SimKernel           = SimKernel,
        SimLogger           = SimLogger,
        SimParticles        = SimParticles,
        SimDensityDiffusion = LinearDensityDiffusion(),
    )
end
