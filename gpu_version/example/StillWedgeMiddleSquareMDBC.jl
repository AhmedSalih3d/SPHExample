# Middle-square still-wedge particles are generated in this file, then the GPU solver runs.
# Including or running this file starts the simulation. Requires an NVIDIA GPU.
#
#     julia --project=gpu_version gpu_version/example/StillWedgeMiddleSquareMDBC.jl
#
# FloatType = Float32 is usually 2-4x faster than Float64 on consumer and
# laptop GPUs (their double precision throughput is low); Float64 reproduces
# the CPU results to round-off.
using SPHExampleGPU
let path = joinpath(@__DIR__, "StillWedgeMDBC.jl")
    # Load the shared wedge helpers without also starting that simulation.
    source = read(path, String)
    source = replace(source, r"\r?\n# Including this file starts the simulation\.[\s\S]*\z" => "\n")
    include_string(@__MODULE__, source, path)
end
using StructArrays
using Meshes

function run_still_wedge_middle_square()
    Dimensions = 2
    FloatType = Float32

    SimConstantsWedge = SimulationConstants{FloatType}(
        dx = 0.02,
        c₀ = 42.48576250492629,
        δᵩ = 0.1,
        CFL = 0.5,
    )
    # SimConstantsWedge = SimulationConstants{FloatType}(dx=0.01,c₀=43.4, δᵩ = 0.1, CFL=0.2)

    # Assuming SimConstantsWedge is defined somewhere else with the field `dx`
    # Sample particles here; the generator script supplies shape definitions.
    polygons = still_wedge_middle_square_polygons()
    regions = [
        ParticleRegion("Bound", Multi([polygons.tank, polygons.square]), Fixed),
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    sampled = sample_particles(regions, SimConstantsWedge.dx)
    positions(name) = only(r.positions for r in sampled if r.name == name)
    water_level = maximum(last, positions("Fluid"))

    FixedBoundary = SPHGeometry{Dimensions, FloatType}(
        Particles = particle_struct_array(positions("Bound"), SimConstantsWedge.ρ₀),
        # CSVFile     = "./input/still_wedge_middle_square_mdbc/StillWedge_MiddleSquare_Dp$(SimConstantsWedge.dx)_Bound.csv",
        GroupMarker = 1,
        Type = Fixed,   # Using the enum value Fixed
        Motion = nothing,
    )

    Water = SPHGeometry{Dimensions, FloatType}(
        Particles = StructArray((
            Position = positions("Fluid"),
            Density = hydrostatic_density(
                positions("Fluid"),
                SimConstantsWedge;
                water_level,
            ),
        )),
        # CSVFile     = "./input/still_wedge_middle_square_mdbc/StillWedge_MiddleSquare_Dp$(SimConstantsWedge.dx)_Fluid.csv",
        GroupMarker = 2,
        Type = Fluid,   # Using the enum value Fluid
        Motion = nothing,
    )
    #
    SimulationGeometry = [FixedBoundary; Water]

    SimMetaDataWedge = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        SimpleMDBC,
        StoreLog,
    }(
        SimulationName = "StillWedge",
        SaveLocation = "C:/TestSimulations/StillWedgeMiddleSquare2D_MDBC_GPU",
        SimulationTime = 4,
        OutputTimes = 0.01,
        VisualizeInParaview = true,
        ExportSingleVTKHDF = true,
        ExportGridCells = true,
        OpenLogFile = true,
    )

    # If save directory is not already made, make it
    if !isdir(SimMetaDataWedge.SaveLocation)
        mkpath(SimMetaDataWedge.SaveLocation)
    end

    SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaDataWedge)

    CleanUpSimulationFolder(SimMetaDataWedge.SaveLocation)

    SimKernel =
        SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); dx = SimConstantsWedge.dx)

    RunSimulation(
        SimGeometry = SimulationGeometry,
        SimMetaData = SimMetaDataWedge,
        SimConstants = SimConstantsWedge,
        SimKernel = SimKernel,
        SimLogger = SimLogger,
        SimParticles = SimParticles,
        SimViscosity = ArtificialViscosity(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        SimTimeStepping = SymplecticTimeStepping(),
        ParticleNormalsPath = "./input/still_wedge_middle_square_mdbc/StillWedge_MiddleSquare_Dp$(SimConstantsWedge.dx)_GhostNodes.csv",
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

const MIDDLE_SQUARE_OUTPUT_DIR =
    normpath(joinpath(@__DIR__, "..", "input", "still_wedge_middle_square_generated"))

"""
    still_wedge_middle_square_polygons(; tank_width = 2.2, tank_height = 0.7,
        water_height = 0.5, wall_thickness = 0.04, wedge_apex = (1.1, 0.26),
        wedge_shell = 0.06, square_lower_left = (0.9, 0.36),
        square_width = 0.4, square_height = 0.5)

Return the tank with its hollow 45° wedge, water and fixed central block as
`PolyArea`s, in metres. The block starts above the wedge tip and extends
through the free surface by default. The water excludes the block, using a
notch when it reaches the surface or a hole when it is fully submerged.
"""
function still_wedge_middle_square_polygons(;
    tank_width = 2.2,
    tank_height = 0.7,
    water_height = 0.5,
    wall_thickness = 0.04,
    wedge_apex = (1.1, 0.26),
    wedge_shell = 0.06,
    square_lower_left = (0.9, 0.36),
    square_width = 0.4,
    square_height = 0.5,
)
    apex_x, apex_y = wedge_apex
    square_x, square_y = square_lower_left
    dimensions = (
        tank_width,
        tank_height,
        water_height,
        wall_thickness,
        apex_x,
        apex_y,
        wedge_shell,
        square_x,
        square_y,
        square_width,
        square_height,
    )
    if !all(isfinite, dimensions) || !(
        0 < water_height <= tank_height &&
        wall_thickness > 0 &&
        0 < wedge_shell < apex_y < water_height &&
        apex_y < apex_x < tank_width - apex_y
    )
        throw(
            ArgumentError(
                "tank, water and hollow wedge dimensions must be " *
                "finite, positive and fit inside the tank",
            ),
        )
    end
    if !(
        square_width > 0 &&
        square_height > 0 &&
        0 < square_x < square_x + square_width < tank_width &&
        apex_y < square_y < water_height
    )
        throw(
            ArgumentError(
                "the block must fit horizontally inside the tank " *
                "and start above the wedge and below the water level",
            ),
        )
    end

    base = still_wedge_2d_polygons(;
        tank_width,
        tank_height,
        water_height,
        wall_thickness,
        wedge_apex,
        wedge_shell,
    )
    square_vertices = [
        (square_x, square_y),
        (square_x + square_width, square_y),
        (square_x + square_width, square_y + square_height),
        (square_x, square_y + square_height),
    ]
    square = PolyArea(square_vertices)
    water_vertices = [Tuple(Meshes.ustrip.(to(point))) for point in vertices(base.water)]
    water = if square_y + square_height < water_height
        PolyArea([water_vertices, reverse(square_vertices)])
    else
        PolyArea(
            vcat(
                water_vertices[1:(end-1)],
                [
                    (square_x + square_width, water_height),
                    (square_x + square_width, square_y),
                    (square_x, square_y),
                    (square_x, water_height),
                    last(water_vertices),
                ],
            ),
        )
    end

    return (; tank = base.tank, water, square)
end

"""
    generate_still_wedge_middle_square_geometry(output_dir;
        dx = 0.02,
        SimConstants = SimulationConstants{Float64}(; dx, c₀ = 42.48576250492629),
        water_level = nothing)

Build and sample the middle-square still-wedge polygons without reading input
CSVs. Write `StillWedge_MiddleSquare_Geometry.vtkhdf`, sampled particle VTKHDF
and `Bound`/`Fluid` CSV files to `output_dir`.

The tank, wedge and block share fixed boundary marker 1; fluid uses marker 2.
Boundary density starts at `ρ₀` and fluid density is hydrostatic, using the
same package tools as the StillWedge generator. Returns the sampled boundary
and fluid regions with their densities. No simulation is run or GPU needed.
"""
function generate_still_wedge_middle_square_geometry(
    output_dir;
    dx = 0.02,
    SimConstants = SimulationConstants{Float64}(; dx, c₀ = 42.48576250492629),
    water_level = nothing,
)
    polygons = still_wedge_middle_square_polygons()
    boundary = Multi([polygons.tank, polygons.square])
    return generate_still_wedge_2d_example(
        output_dir;
        dx,
        SimConstants,
        water_level,
        polygons,
        boundary,
        case_name = "StillWedge_MiddleSquare",
    )
end

# Including this file starts the simulation.
run_still_wedge_middle_square()
