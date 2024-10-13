import StaticArrays: SVector
using SPHExample

let
    Dimensions = 2
    FloatType  = Float64

    
    SimConstantsPeriodicity = SimulationConstants{FloatType}(dx=0.002,c₀=15.53467127810402, δᵩ = 0.1, CFL=0.2)

    # Assuming SimConstantsPeriodicity is defined somewhere else with the field `dx`
    FixedBoundary = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/periodicity_2d/Periodicity_Dp$(SimConstantsPeriodicity.dx)_Bound.csv",
        GroupMarker = 1,
        Type        = Fixed,   # Using the enum value Fixed
        Motion      = nothing
    )

    Water = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/periodicity_2d/Periodicity_Dp$(SimConstantsPeriodicity.dx)_Fluid.csv",
        GroupMarker = 2,
        Type        = Fluid,   # Using the enum value Fluid
        Motion      = nothing
    )

    Perio = PeriodicityConditions{Dimensions, FloatType}(
        IsPeriodic     = SVector( true,  false),  # Only x is periodic
        MinBounds      = SVector(-0.01,  0.0),    # x_min = -0.01, y_min = 0.0, z_min = 0.0
        MaxBounds      = SVector( 0.8,   0.0),    # x_max = 0.8,   y_max = 0.0, z_max = 0.0
        HeightIncrease = 0.3                      # Scalar value for increase in height (z) when reentering
    )

    println(Perio)

    SimulationGeometry = [FixedBoundary;Water]

    SimMetaDataPeriodicity  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="Periodicity2d", 
        SaveLocation="E:/SecondApproach/Periodicity2d",
        SimulationTime=6,
        OutputEach=0.06 ,
        VisualizeInParaview=true,
        OpenLogFile=true,
        FlagDensityDiffusion=true, 
        FlagOutputKernelValues=false,
        FlagLog=true,
        FlagShifting=false,
        FlagPeriodicity=true,
    )

    SimLogger = SimulationLogger(SimMetaDataPeriodicity.SaveLocation)

    CleanUpSimulationFolder(SimMetaDataPeriodicity.SaveLocation)

    @profview RunSimulation(
        SimGeometry        = SimulationGeometry,
        SimMetaData        = SimMetaDataPeriodicity,
        SimConstants       = SimConstantsPeriodicity,
        SimLogger          = SimLogger
    )
end

