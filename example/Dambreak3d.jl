using SPHExample

let
    Dimensions = 3
    FloatType  = Float64

    dx = 0.02
    SimConstantsDambreak3D = SimulationConstants{FloatType}(
        dx  = dx,
        c₀  = 33.14,
        h   = 1 * sqrt(3 * dx^2),
        α   = 0.1,
        αD  = 21/(16*π*(1.2 * sqrt(3) * dx)^3),
        m₀  = 1000 * dx^3,
        CFL = 0.2)
    
    # Define the dictionary with specific types for keys and values to avoid any type ambiguity
    SimulationGeometry = Dict{Symbol, Dict{String, Union{String, Int, ParticleType, Nothing}}}()

    # Populate the dictionary
    SimulationGeometry[:FixedBoundary] = Dict(
        "CSVFile"     => "./input/dam_break_3d/DamBreak3d_Dp$(dx)_Bound.csv",
        "GroupMarker" => 1,
        "Type"        => Fixed,
        "Motion"      => nothing
    )

    SimulationGeometry[:Water] = Dict(
        "CSVFile"     => "./input/dam_break_3d/DamBreak3d_Dp$(dx)_Fluid.csv",
        "GroupMarker" => 2,
        "Type"        => Fluid,
        "Motion"      => nothing
    )


    SimMetaDataDambreak3D  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="Test", 
        SaveLocation="E:/SecondApproach/TESTING_CPU",
        SimulationTime=1.6,
        OutputEach=0.01,
        VisualizeInParaview=true,
        OpenLogFile=true,
        FlagDensityDiffusion=true,
        FlagOutputKernelValues=false,
        FlagLog=true
    )

    SimLogger = SimulationLogger(SimMetaDataDambreak3D.SaveLocation)

    @warn("3D is included, but has not been extensively tested.
    With only one particle layer for boundary, it seems that fluid particles
    can wiggle through when the initial water column is dissolved")


    RunSimulation(
        SimGeometry        = SimulationGeometry,
        SimMetaData        = SimMetaDataDambreak3D,
        SimConstants       = SimConstantsDambreak3D,
        SimLogger          = SimLogger
    )
end
