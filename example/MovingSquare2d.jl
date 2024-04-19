using SPHExample

let
    Dimensions = 2
    FloatType  = Float64

    SimMetaDataMovingSquare  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="MovingSquare2D", 
        SaveLocation="E:/SecondApproach/MovingSquare2D",
        SimulationTime=0.5,
        OutputEach=0.01,
        FlagDensityDiffusion=true,
        FlagOutputKernelValues=false,
        FlagLog=true,
        FlagShifting=true,
        FlagViscosityTreatment=:LaminarSPS
    )

    # If save directory is not already made, make it
    if !isdir(SimMetaDataMovingSquare.SaveLocation)
        mkdir(SimMetaDataMovingSquare.SaveLocation)
    end

    # ViscoBoundFactor should be 1, but need to understand how to implement it
    SimConstantsMovingSquare = SimulationConstants{FloatType}(dx=0.02,
    c₀=28, 
    δᵩ = 0.1,
    g  = 0,
    Cb = 112000,
    α  = 1e-6,
    k  = sqrt(2),
    CFL=0.2)

    SimLogger = SimulationLogger(SimMetaDataMovingSquare.SaveLocation)

    # println(
    # @report_opt 
    RunSimulation(
        FluidCSV           = "./input/moving_square_2d/MovingSquare_Dp$(SimConstantsMovingSquare.dx)_Fluid.csv",
        FixedCSV           = "./input/moving_square_2d/MovingSquare_Dp$(SimConstantsMovingSquare.dx)_Fixed.csv",
        MovingCSV          = "./input/moving_square_2d/MovingSquare_Dp$(SimConstantsMovingSquare.dx)_Square.csv",
        SimMetaData        = SimMetaDataMovingSquare,
        SimConstants       = SimConstantsMovingSquare,
        SimLogger          = SimLogger
    )
    # )
end
