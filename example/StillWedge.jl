using SPHExample

let
    Dimensions = 2
    FloatType  = Float64

    
    SimConstantsWedge = SimulationConstants{FloatType}(dx=0.01,c₀=43.48576250492629, δᵩ = 0.1, CFL=0.2)

    # Define the dictionary with specific types for keys and values to avoid any type ambiguity
    SimulationGeometry = Dict{Symbol, Dict{String, Union{String, Int, ParticleType, Nothing}}}()

    # Populate the dictionary
    SimulationGeometry[:FixedBoundary] = Dict(
        "CSVFile"     => "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Bound.csv",
        "GroupMarker" => 1,
        "Type"        => Fixed,
        "Motion"      => nothing
    )

    SimulationGeometry[:Water] = Dict(
        "CSVFile"     => "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Fluid.csv",
        "GroupMarker" => 2,
        "Type"        => Fluid,
        "Motion"      => nothing
    )


    SimMetaDataWedge  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="StillWedge", 
        SaveLocation="E:/SecondApproach/TESTING_CPU_StillWedge",
        SimulationTime=4,
        OutputEach=0.01,
        FlagDensityDiffusion=true,
        FlagOutputKernelValues=false,
        FlagLog=true,
        FlagShifting=false,
    )

    SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation)

    println(@report_opt target_modules=(@__MODULE__,) RunSimulation(
        SimGeometry        = SimulationGeometry,
        SimMetaData        = SimMetaDataWedge,
        SimConstants       = SimConstantsWedge,
        SimLogger          = SimLogger
    )
    )

    @profview RunSimulation(
        SimGeometry        = SimulationGeometry,
        SimMetaData        = SimMetaDataWedge,
        SimConstants       = SimConstantsWedge,
        SimLogger          = SimLogger
    )
end

