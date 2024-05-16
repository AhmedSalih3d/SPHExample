using SPHExample
using StaticArrays

let
    Dimensions = 2
    FloatType  = Float64

    
    # SimConstantsWedge = SimulationConstants{FloatType}(dx=0.025,c₀=84.04284584365287, δᵩ = 0.1, CFL=0.2, k = 1.69706)
    SimConstantsWedge = SimulationConstants{FloatType}(dx=0.05,c₀=84.04284584365287, δᵩ = 0.1, CFL=0.2, k = 1.69706)

    # Define the dictionary with specific types for keys and values to avoid any type ambiguity
    SimulationGeometry = Dict{Symbol, Dict{String, Union{String, Int, FloatType, ParticleType, SVector, SMatrix, Nothing}}}()

    # Populate the dictionary
    SimulationGeometry[:FixedBoundary] = Dict(
        "CSVFile"     => "./input/floating_cylinder_2d/Wall_WithWalls_Dp$(SimConstantsWedge.dx).csv",
        "GroupMarker" => 1,
        "Type"        => Fixed,
        "Motion"      => nothing
    )

    # SimulationGeometry[:Water] = Dict(
    #     "CSVFile"     => "./input/floating_cylinder_2d/Fluid_WithWalls_Dp$(SimConstantsWedge.dx).csv",
    #     "GroupMarker" => 2,
    #     "Type"        => Fluid,
    #     "Motion"      => nothing
    # )

    SimulationGeometry[:Cylinder] = Dict(
        "CSVFile"     => "./input/floating_cylinder_2d/Cylinder_Dp$(SimConstantsWedge.dx).csv",
        "GroupMarker" => 3,
        "Type"        => Floating,
        "Motion"      => nothing,
        "Mass"        => 3882.75, 
        "COG"         => SVector{Dimensions, FloatType}(0.0, 14.0),
        "Inertia"     => SMatrix{3, 3, FloatType}((999.714, 0, 0, 0, 1999.43, 0, 0, 0, 999.714))
    )


    SimMetaDataWedge  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="FloatingCylinder", 
        SaveLocation="E:/SecondApproach/FloatingCylinder2d",
        SimulationTime=2,
        OutputEach=0.02,
        FlagDensityDiffusion=true,
        FlagOutputKernelValues=true,
        FlagLog=true,
        FlagShifting=false,
    )

    SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation)

    @profview RunSimulation(
        SimGeometry        = SimulationGeometry,
        SimMetaData        = SimMetaDataWedge,
        SimConstants       = SimConstantsWedge,
        SimLogger          = SimLogger
    )
end

