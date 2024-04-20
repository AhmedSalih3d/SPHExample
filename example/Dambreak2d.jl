using SPHExample

let
    Dimensions = 2
    FloatType  = Float64

    SimConstantsDambreak = SimulationConstants{FloatType}(dx=0.02,c₀=88.14487860902641, δᵩ = 0.1, CFL=0.2, α = 0.02)

    # Define the dictionary with specific types for keys and values to avoid any type ambiguity
    SimulationGeometry = Dict{Symbol, Dict{String, Union{String, Int, ParticleType, Nothing}}}()

    # Populate the dictionary
    SimulationGeometry[:FixedBoundary] = Dict(
        "CSVFile"     => "./input/dam_break_2d/DamBreak2d_Dp0.02_Bound_ThreeLayers.csv",
        "GroupMarker" => 1,
        "Type"        => Fixed,
        "Motion"      => nothing
    )

    SimulationGeometry[:Water] = Dict(
        "CSVFile"     => "./input/dam_break_2d/DamBreak2d_Dp0.02_Fluid_OneLayer.csv",
        "GroupMarker" => 2,
        "Type"        => Fluid,
        "Motion"      => nothing
    )


    SimMetaDataDambreak  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="Test", 
        SaveLocation="E:/SecondApproach/TESTING_CPU",
        SimulationTime=2,
        OutputEach=0.01,
        FlagDensityDiffusion=true,
        FlagOutputKernelValues=false,
        FlagLog=true
    )


    SimLogger = SimulationLogger(SimMetaDataDambreak.SaveLocation)

    println(@report_opt target_modules=(@__MODULE__,) RunSimulation(
        SimGeometry        = SimulationGeometry,
        SimMetaData        = SimMetaDataDambreak,
        SimConstants       = SimConstantsDambreak,
        SimLogger          = SimLogger
    )
    )

    RunSimulation(
        SimGeometry        = SimulationGeometry,
        SimMetaData        = SimMetaDataDambreak,
        SimConstants       = SimConstantsDambreak,
        SimLogger          = SimLogger
    )
end
