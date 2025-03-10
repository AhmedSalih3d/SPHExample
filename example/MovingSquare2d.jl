import StaticArrays: SVector
using SPHExample

let
    Dimensions = 2
    FloatType  = Float64

    # ViscoBoundFactor should be 1, but need to understand how to implement it
    SimConstantsMovingSquare = SimulationConstants{FloatType}(dx=0.02,
        c₀=28, 
        δᵩ = 0.1,
        g  = 0,
        Cb = 112000,
        α  = 1e-6,
        k  = sqrt(2),
        CFL=0.2
    )

    SimMetaDataMovingSquare  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="MovingSquare2D", 
        SaveLocation="E:/SecondApproach/MovingSquare2D",
        SimulationTime=2.5,
        OutputEach=0.01,
        VisualizeInParaview=true,
        ExportSingleVTKHDF=true,
        OpenLogFile=true,
        FlagDensityDiffusion=true,
        FlagOutputKernelValues=false,
        FlagLog=true,
        FlagShifting=true,
        FlagViscosityTreatment=:LaminarSPS
    )
    FixedBoundary = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/moving_square_2d/MovingSquare_Dp$(SimConstantsMovingSquare.dx)_Fixed.csv",
        GroupMarker = 1,
        Type        = Fixed,
        Motion      = nothing
    )
    
    Water = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/moving_square_2d/MovingSquare_Dp$(SimConstantsMovingSquare.dx)_Fluid.csv",
        GroupMarker = 2,
        Type        = Fluid,
        Motion      = nothing
    )
    
    MovingSquare = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/moving_square_2d/MovingSquare_Dp$(SimConstantsMovingSquare.dx)_Square.csv",
        GroupMarker = 3,
        Type        = Moving,
        Motion      = MotionDetails{Dimensions, FloatType}(
            Velocity  = 2.8,
            StartTime = 0.0,
            Duration  = 3.0,
            Direction = SVector{Dimensions, FloatType}(1.0, 0.0)  # 2D direction vector with Float64 type
        )
    )
    
    # Collect Geometry instances into a vector
    SimulationGeometry = [FixedBoundary, Water, MovingSquare]
    # If save directory is not already made, make it
    if !isdir(SimMetaDataMovingSquare.SaveLocation)
        mkdir(SimMetaDataMovingSquare.SaveLocation)
    end

    SimLogger = SimulationLogger(SimMetaDataMovingSquare.SaveLocation)

    CleanUpSimulationFolder(SimMetaDataMovingSquare.SaveLocation)

    RunSimulation(
        SimGeometry        = SimulationGeometry,
        SimMetaData        = SimMetaDataMovingSquare,
        SimConstants       = SimConstantsMovingSquare,
        SimLogger          = SimLogger
    )
end
