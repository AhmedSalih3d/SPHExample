using SPHExample

let
    Dimensions = 2
    FloatType  = Float64

    
    SimConstantsWedge = SimulationConstants{FloatType}(dx=0.02,c₀=42.48576250492629, δᵩ = 0.1, CFL=0.2)

    # Define the dictionary with specific types for keys and values to avoid any type ambiguity
    SimulationGeometry = Dict{Symbol, Dict{String, Union{String, Int, ParticleType, Nothing}}}()
        
    # Create a Geometry instance using the @enum ParticleType
    FixedBoundary = Geometry(
        "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Bound.csv",
        1,
        Fixed,   # Using the enum value Fixed
        nothing
    )

    Water = Geometry(
        "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Fluid.csv",
        2,
        Fluid,   # Using the enum value Fluid
        nothing
    )

    SimulationGeometry = [FixedBoundary;Water]


    SimMetaDataWedge  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="StillWedge", 
        SaveLocation="E:/SecondApproach/TESTING_CPU_StillWedge",
        SimulationTime=4,
        OutputEach=0.01,
        VisualizeInParaview=true,
        OpenLogFile=true,
        FlagDensityDiffusion=true,
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

