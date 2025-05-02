using SPHExample

let
    Dimensions = 2
    FloatType  = Float64

    SimConstantsWedge = SimulationConstants{FloatType}(dx=0.02,c₀=63.26515727475907, δᵩ = 0.1, CFL=0.5, ν₀=1e-3)
    # SimConstantsWedge = SimulationConstants{FloatType}(dx=0.01,c₀=62.95426812377379,  g=0.0, δᵩ = 0.1, CFL=0.5, ν₀=1e-3)

    # Assuming SimConstantsWedge is defined somewhere else with the field `dx`
    FixedBoundary = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/lid_driven_cavity/LidDrivenCavity_Dp$(SimConstantsWedge.dx)_Bound.csv",
        GroupMarker = 1,
        Type        = Fixed,   # Using the enum value Fixed
        Motion      = nothing
    )

    Water = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/lid_driven_cavity/LidDrivenCavity_Dp$(SimConstantsWedge.dx)_Fluid.csv",
        GroupMarker = 2,
        Type        = Fluid,   # Using the enum value Fluid
        Motion      = nothing
    )
# 
    SimulationGeometry = [FixedBoundary;Water]
    
    # Load in particles
    SimParticles = AllocateDataStructures(SimulationGeometry)

    SimMetaDataWedge  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="LidDrivenCavity", 
        SaveLocation="E:/SecondApproach/LidDrivenCavity_MDBC",
        SimulationTime=7.0,
        OutputEach=0.1,
        VisualizeInParaview=true,
        ExportSingleVTKHDF=true,
        ExportGridCells=true,
        OpenLogFile=true,
        FlagOutputKernelValues=false,
        FlagLog=true,
        FlagMDBCSimple=true
    )

    # If save directory is not already made, make it
    if !isdir(SimMetaDataWedge.SaveLocation)
        mkdir(SimMetaDataWedge.SaveLocation)
    end

    SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation)

    CleanUpSimulationFolder(SimMetaDataWedge.SaveLocation)

    SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); h = 1.2 * sqrt(2 * SimConstantsWedge.dx^2))

    @profview RunSimulation(
        SimGeometry         = SimulationGeometry,
        SimMetaData         = SimMetaDataWedge,
        SimConstants        = SimConstantsWedge,
        SimKernel           = SimKernel,
        SimLogger           = SimLogger,
        SimParticles        = SimParticles,
        SimViscosity        = LaminarSPS(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        ParticleNormalsPath = "./input/lid_driven_cavity/LidDrivenCavity_Dp$(SimConstantsWedge.dx)_GhostNodes.csv"
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





