using SPHExample

let
    Dimensions = 2
    FloatType  = Float64

    SimConstantsWedge = SimulationConstants{FloatType}(dx=0.02,c₀=42.48576250492629, δᵩ = 0.1, CFL=0.5)
    # SimConstantsWedge = SimulationConstants{FloatType}(dx=0.01,c₀=43.4, δᵩ = 0.1, CFL=0.2)
# 
    # Assuming SimConstantsWedge is defined somewhere else with the field `dx`
    FixedBoundary = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Bound.csv",
        GroupMarker = 1,
        Type        = Fixed,   # Using the enum value Fixed
        Motion      = nothing
    )

    Water = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/still_wedge/StillWedge_Dp$(SimConstantsWedge.dx)_Fluid.csv",
        GroupMarker = 2,
        Type        = Fluid,   # Using the enum value Fluid
        Motion      = nothing
    )

    SimulationGeometry = [FixedBoundary;Water]
    
    # Load in particles
    SimParticles = AllocateDataStructures(SimulationGeometry)

    SimMetaDataWedge  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="StillWedge", 
        SaveLocation="E:/SecondApproach/StillWedge2D_MDBC",
        SimulationTime=4,
        OutputEach=0.01,
        VisualizeInParaview=true,
        ExportSingleVTKHDF=true,
        ExportGridCells=true,
        OpenLogFile=true,
        FlagOutputKernelValues=false,
        FlagLog=true,
        FlagMDBCSimple=true,
        OutputVariables = [
            # "ChunkID",
            # "Kernel",
            # "KernelGradient",
            "Density",
            "Pressure",
            "Velocity",
            "Acceleration",
            # "BoundaryBool",
            # "ID",
            # "Type",
            # "GroupMarker",
            # "GhostPoints",
            # "GhostNormals",
        ]
    )

    SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation; to_console=true)

    CleanUpSimulationFolder(SimMetaDataWedge.SaveLocation)

    SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); dx = SimConstantsWedge.dx)

    @profview RunSimulation(
        SimGeometry         = SimulationGeometry,
        SimMetaData         = SimMetaDataWedge,
        SimConstants        = SimConstantsWedge,
        SimKernel           = SimKernel,
        SimLogger           = SimLogger,
        SimParticles        = SimParticles,
        SimViscosity        = ArtificialViscosity(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        ParticleNormalsPath = "./input/still_wedge_mdbc/StillWedge_Dp$(SimConstantsWedge.dx)_GhostNodes_Correct.csv"
    )

    # This can be used to plot pressure profile results after simulation
    # using Plots
    # using StaticArrays
    
    # # Assuming 'data' is a vector of the named tuples containing the data
    
    # # Constants
    # max_height = 0.5             # maximum height in meters
    # rho = SimConstantsWedge.ρ₀   # density in kg/m^3 (adjust if needed)
    # g   = SimConstantsWedge.g    # gravitational acceleration in m/s^2
    
    # # Filter only fluid particles
    # fluid_data = filter(d -> d.Type == Fluid, SimParticles)
    
    # # Extract positions and pressures for fluid particles
    # positions = [d.Position[2] for d in fluid_data]  # Extract the height (y-component)
    # pressures = [d.Pressure    for d in fluid_data]  # Extract the pressure
    
    # # Normalize positions and pressures
    # normalized_positions = [p / max_height for p in positions]  # Normalize height
    # hydrostatic_pressure = [rho * g * (max_height - h) for h in positions]     # Theoretical hydrostatic pressure
    # normalized_pressures = [p / maximum(hydrostatic_pressure) for p in pressures]  # Normalize pressure
    
    # # Create the plot
    # plt = scatter(normalized_pressures, normalized_positions, label="Fluid Pressure", xlabel="Normalized Height", ylabel="Normalized Pressure", linestyle=:auto, marker=:circle, legend=:topright)
    
    # # Plot the theoretical hydrostatic pressure line (with correct flipped axes)
    # plot!(hydrostatic_pressure ./ maximum(hydrostatic_pressure), normalized_positions, label="Theoretical Hydrostatic Pressure", linestyle=:dash)
    
    # # Set fixed axis limits for better comparison
    # xlims!((0, 1))
    # ylims!((0, 1))
    
    # display(plt)
end




