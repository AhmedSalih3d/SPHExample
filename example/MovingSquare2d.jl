using SPHExample

let
    Dimensions = 2
    FloatType  = Float64

    SimMetaDataWedge  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="MovingSquare2D", 
        SaveLocation="E:/SecondApproach/MovingSquare2D",
        SimulationTime=2.5,
        OutputEach=0.01,
        FlagDensityDiffusion=true,
        FlagOutputKernelValues=false,
        FlagLog=true,
        FlagShifting=true,
        FlagViscosityTreatment=:LaminarSPS
    )

    # If save directory is not already made, make it
    if !isdir(SimMetaDataWedge.SaveLocation)
        mkdir(SimMetaDataWedge.SaveLocation)
    end

    # ViscoBoundFactor should be 1, but need to understand how to implement it
    SimConstantsWedge = SimulationConstants{FloatType}(dx=0.04,
    c₀=28, 
    δᵩ = 0.1,
    g  = 0,
    Cb = 112000,
    α  = 1e-6,
    k  = sqrt(2),
    CFL=0.2)

    SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation)

    @profview RunSimulation(
        FluidCSV           = "./input/moving_square_2d/MovingSquare_Dp$(SimConstantsWedge.dx)_Fluid.csv",
        FixedCSV           = "./input/moving_square_2d/MovingSquare_Dp$(SimConstantsWedge.dx)_Fixed.csv",
        MovingCSV          = "./input/moving_square_2d/MovingSquare_Dp$(SimConstantsWedge.dx)_Square.csv",
        SimMetaData        = SimMetaDataWedge,
        SimConstants       = SimConstantsWedge,
        SimLogger          = SimLogger
    )
end
