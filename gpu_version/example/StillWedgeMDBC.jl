# Still-wedge particles are generated in this file, then the GPU solver runs.
# Including or running this file starts the simulation. Requires an NVIDIA GPU.
#
#     julia --project=gpu_version gpu_version/example/StillWedgeMDBC.jl
#
# FloatType = Float32 is usually 2-4x faster than Float64 on consumer and
# laptop GPUs (their double precision throughput is low); Float64 reproduces
# the CPU results to round-off.
using SPHExampleGPU
using StaticArrays: SVector
using Meshes

# Set to `false` to skip the optional PDF report.
GeneratePDFReport = true

include(joinpath(@__DIR__, "StillWedgeMDBCReport.jl"))
using .StillWedgeMDBCReport: write_still_wedge_report

function run_still_wedge(;
    time_stepping::TimeSteppingMode = SymplecticTimeStepping(),
    simulation_time = 4.00,
    save_location = "C:/TestSimulations/StillWedge2D_MDBC_GPU",
    visualize::Bool = true,
    open_log_file::Bool = true,
    generate_pdf_report::Bool = GeneratePDFReport,
)
    Dimensions = 2
    FloatType = Float32

    SimConstantsWedge = SimulationConstants{FloatType}(
        dx = 0.02,
        c₀ = 42.48576250492629,
        δᵩ = 0.1,
        CFL = 0.5,
    )
    # SimConstantsWedge = SimulationConstants{FloatType}(dx=0.01,c₀=43.4, δᵩ = 0.1, CFL=0.2)
    #
    polygons = still_wedge_2d_polygons()
    regions = [
        ParticleRegion("Bound", polygons.tank, Fixed),
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    sampled = sample_particles(regions, SimConstantsWedge.dx)
    boundary, fluid = sampled
    FixedBoundary = SPHGeometry{Dimensions, FloatType}(
        boundary.positions;
        Density = SimConstantsWedge.ρ₀,
        # CSVFile = "input/still_wedge_generated/StillWedge2D_Dp$(SimConstantsWedge.dx)_Bound.csv",
        GroupMarker = 1,
        Type = Fixed,
    )
    Water = SPHGeometry{Dimensions, FloatType}(
        fluid.positions;
        Density = hydrostatic_density(fluid.positions, SimConstantsWedge),
        # CSVFile = "input/still_wedge_generated/StillWedge2D_Dp$(SimConstantsWedge.dx)_Fluid.csv",
        GroupMarker = 2,
        Type = Fluid,
    )
    SimulationGeometry = [FixedBoundary, Water]

    SimMetaDataWedge = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        NoMDBC,
        StoreLog,
    }(
        SimulationName = "StillWedge",
        SaveLocation = save_location,
        SimulationTime = simulation_time,
        OutputTimes = 0.01,
        VisualizeInParaview = visualize,
        ExportSingleVTKHDF = true,
        ExportGridCells = true,
        OpenLogFile = open_log_file,
        GPUDoublePosition = true,
        GPURadialInteractions = true,
        GPUCacheInteractionFields = true,
        # This small case needs only a few queued frames to overlap disk output.
        GPUOutputQueueBytes = 8 * 1024^2,
        # OutputVariables = [
        #     # "ChunkID",
        #     # "Kernel",
        #     # "KernelGradient",
        #     "Density",
        #     "Pressure",
        #     "Velocity",
        #     "Acceleration",
        #     # "BoundaryBool",
        #     # "ID",
        #     # "Type",
        #     # "GroupMarker",
        #     # "GhostPoints",
        #     # "GhostNormals",
        # ]
    )

    mkpath(SimMetaDataWedge.SaveLocation)

    SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaDataWedge)

    CleanUpSimulationFolder(SimMetaDataWedge.SaveLocation)

    SimKernel =
        SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); dx = SimConstantsWedge.dx)
    SimViscosity = ArtificialViscosity()
    SimDensityDiffusion = LinearDensityDiffusion()

    result = RunSimulation(
        SimGeometry = SimulationGeometry,
        SimMetaData = SimMetaDataWedge,
        SimConstants = SimConstantsWedge,
        SimKernel = SimKernel,
        SimLogger = SimLogger,
        SimParticles = SimParticles,
        SimViscosity = SimViscosity,
        SimDensityDiffusion = SimDensityDiffusion,
        SimTimeStepping = time_stepping,
        ParticleNormalsPath = "./input/still_wedge_mdbc/StillWedge_Dp$(SimConstantsWedge.dx)_GhostNodes_Correct.csv",
    )

    if generate_pdf_report
        report_files = write_still_wedge_report(
            result,
            SimConstantsWedge;
            output_dir = SimMetaDataWedge.SaveLocation,
            simulation_time,
            simulation_name = SimMetaDataWedge.SimulationName,
            metadata = SimMetaDataWedge,
            geometry = SimulationGeometry,
            kernel = SimKernel,
            viscosity = SimViscosity,
            density_diffusion = SimDensityDiffusion,
        )
        @info "Wrote StillWedge report" typst_source = report_files.typ_path pdf = report_files.pdf_path
    end

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

    # # Plot the theoretical hydrostatic pressure line (with correct flipped axes)'
    # plot!(hydrostatic_pressure ./ maximum(hydrostatic_pressure), normalized_positions, label="Theoretical Hydrostatic Pressure", linestyle=:dash)

    # # Set fixed axis limits for better comparison
    # xlims!((0, 1))
    # ylims!((0, 1))

    # display(plt)
    return result
end

"""
    still_wedge_2d_polygons(; tank_width = 2.2, tank_height = 0.7, water_height = 0.5,
                             wall_thickness = 0.04, wedge_apex = (1.1, 0.26),
                             wedge_shell = 0.06)

Tank outline and water of the still wedge case as 2D polygons, in metres.

The tank is one polygon: a floor and two side walls of `wall_thickness`
carrying a hollow wedge with 45° slopes whose tip is `wedge_apex`. The wedge is
a shell of horizontal thickness `wedge_shell`; its inner V continues through
the floor, so there is no boundary below the wedge (as in the reference
`input/still_wedge` particles). The water fills the tank around the wedge up
to `water_height`.
"""
function still_wedge_2d_polygons(;
    tank_width = 2.2,
    tank_height = 0.7,
    water_height = 0.5,
    wall_thickness = 0.04,
    wedge_apex = (1.1, 0.26),
    wedge_shell = 0.06,
)
    apex_x, apex_y = wedge_apex
    t = wall_thickness
    outer_half_width = apex_y                   # 45° slopes: half width = height
    inner_apex_y = apex_y - wedge_shell
    inner_half_width = inner_apex_y + t         # inner V where it meets the floor bottom

    wedge_left = (apex_x - outer_half_width, 0.0)
    wedge_right = (apex_x + outer_half_width, 0.0)

    # Counter clockwise: along the floor bottom, up the right wall, back along
    # the wetted surface (floor, wedge, floor) and up the left wall.
    tank = PolyArea([
        (-t, -t),
        (apex_x - inner_half_width, -t),
        (apex_x, inner_apex_y),                 # inner tip of the hollow wedge
        (apex_x + inner_half_width, -t),
        (tank_width + t, -t),
        (tank_width + t, tank_height),
        (tank_width, tank_height),
        (tank_width, 0.0),
        wedge_right,
        wedge_apex,
        wedge_left,
        (0.0, 0.0),
        (0.0, tank_height),
        (-t, tank_height),
    ])

    water = PolyArea([
        (0.0, 0.0),
        wedge_left,
        wedge_apex,
        wedge_right,
        (tank_width, 0.0),
        (tank_width, water_height),
        (0.0, water_height),
    ])

    return (; tank, water)
end

"""
    generate_still_wedge_2d_example(output_dir; dx = 0.02,
        SimConstants = SimulationConstants{Float64}(; dx, c₀ = 42.48576250492629),
        water_level = nothing)

Write the polygons (`StillWedge2D_Geometry.vtkhdf`), the particles sampled from
them (`StillWedge2D_Dp<dx>_Particles.vtkhdf`) and the CSV input files
(`StillWedge2D_Dp<dx>_Bound.csv`, `..._Fluid.csv`) to `output_dir`.

Boundary particles start at `ρ₀`. Fluid particles start in hydrostatic
equilibrium: their density is the inverse of the solver's equation of state for
`P = ρ₀ g (water_level - y)`. `water_level = nothing` uses the highest fluid
particle. Pass the `SimConstants` of the simulation so that `ρ₀`, `g` and `c₀`
match; the default uses those of `StillWedgeMDBC.jl`.
Related cases can reuse the sampling and export pipeline by supplying
`polygons`, the fixed `boundary` geometry and a `case_name` for the filenames.
Returns the sampled particles per region, with their densities.
"""
function generate_still_wedge_2d_example(
    output_dir;
    dx = 0.02,
    SimConstants = SimulationConstants{Float64}(; dx, c₀ = 42.48576250492629),
    water_level = nothing,
    polygons = still_wedge_2d_polygons(),
    boundary = polygons.tank,
    case_name = "StillWedge2D",
)
    isfinite(dx) && dx > 0 ||
        throw(ArgumentError("particle spacing dx must be finite and positive"))
    regions = [
        ParticleRegion("Bound", boundary, Fixed),  # listed first: owns the shared surfaces
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    sampled = sample_particles(regions, dx)

    fluid_positions = SVector{2, Float64}[]
    for r in sampled
        r.type == Fluid && append!(fluid_positions, r.positions)
    end
    isempty(fluid_positions) &&
        throw(ArgumentError("particle spacing dx leaves the water region empty"))
    level = something(water_level, maximum(last, fluid_positions))
    particles = map(sampled) do region
        density =
            region.type == Fluid ?
            hydrostatic_density(region.positions, SimConstants; water_level = level) :
            fill(SimConstants.ρ₀, length(region.positions))
        (; region..., density)
    end

    mkpath(output_dir)
    prefix = joinpath(output_dir, "$(case_name)_Dp$(dx)")
    SavePolygonVTKHDF(joinpath(output_dir, "$(case_name)_Geometry.vtkhdf"), polygons)

    next_id = 0
    for region in particles
        next_id = write_particle_csv(
            "$(prefix)_$(region.name).csv",
            region.positions;
            density = region.density,
            first_id = next_id,
        )
    end

    (; ρ₀, c₀) = SimConstants
    positions = SVector{3, Float64}[]
    density   = Float64[]
    types     = Int8[]
    markers   = Int[]
    for (id, region) in enumerate(particles)
        append!(positions, to_3d(region.positions))
        append!(density, region.density)
        append!(types, fill(Int8(region.type), length(region.positions)))
        append!(markers, fill(id, length(region.positions)))
    end
    pressure = EquationOfStateGamma7.(density, c₀, ρ₀)
    SaveVTKHDF(
        "$(prefix)_Particles.vtkhdf",
        positions,
        ["Density", "Pressure", "Type", "GroupMarker"],
        density,
        pressure,
        types,
        markers,
    )

    return particles
end

"""
    still_wedge_2d_geometry(constants; water_level = nothing)

Generate boundary and fluid `SPHGeometry` groups directly in memory, with
hydrostatic fluid density. No CSV or visualization files are written.
"""
function still_wedge_2d_geometry(
    constants::SimulationConstants{T};
    water_level = nothing,
) where {T}
    polygons = still_wedge_2d_polygons()
    regions = [
        ParticleRegion("Bound", polygons.tank, Fixed),
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    sampled = sample_particles(regions, constants.dx)
    level = water_level === nothing ? maximum(last, sampled[2].positions) : water_level
    return [
        SPHGeometry{2, T}(
            region.positions;
            Density = region.type == Fluid ?
                      hydrostatic_density(
                region.positions,
                constants;
                water_level = level,
            ) : constants.ρ₀,
            GroupMarker = marker,
            Type = region.type,
        ) for (marker, region) in enumerate(sampled)
    ]
end

# Including this file starts the simulation.
run_still_wedge(generate_pdf_report = GeneratePDFReport)
