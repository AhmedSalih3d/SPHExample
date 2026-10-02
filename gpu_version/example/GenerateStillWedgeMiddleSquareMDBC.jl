# Generate the middle-square still-wedge case from polygons.
#
# Run from the repository root with:
#     julia --project=gpu_version \
#         gpu_version/example/GenerateStillWedgeMiddleSquareMDBC.jl [output_dir] [dx]
include(joinpath(@__DIR__, "GenerateStillWedgeMDBC.jl"))

const MIDDLE_SQUARE_OUTPUT_DIR = normpath(joinpath(
    @__DIR__, "..", "input", "still_wedge_middle_square_generated"))

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
function still_wedge_middle_square_polygons(; tank_width = 2.2, tank_height = 0.7,
        water_height = 0.5, wall_thickness = 0.04, wedge_apex = (1.1, 0.26),
        wedge_shell = 0.06, square_lower_left = (0.9, 0.36),
        square_width = 0.4, square_height = 0.5)
    apex_x, apex_y = wedge_apex
    square_x, square_y = square_lower_left
    dimensions = (tank_width, tank_height, water_height, wall_thickness,
                  apex_x, apex_y, wedge_shell, square_x, square_y,
                  square_width, square_height)
    if !all(isfinite, dimensions) ||
       !(0 < water_height <= tank_height && wall_thickness > 0 &&
         0 < wedge_shell < apex_y < water_height &&
         apex_y < apex_x < tank_width - apex_y)
        throw(ArgumentError("tank, water and hollow wedge dimensions must be " *
                            "finite, positive and fit inside the tank"))
    end
    if !(square_width > 0 && square_height > 0 &&
         0 < square_x < square_x + square_width < tank_width &&
         apex_y < square_y < water_height)
        throw(ArgumentError("the block must fit horizontally inside the tank " *
                            "and start above the wedge and below the water level"))
    end

    base = still_wedge_2d_polygons(; tank_width, tank_height, water_height,
                                   wall_thickness, wedge_apex, wedge_shell)
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
        PolyArea(vcat(water_vertices[1:end-1], [
            (square_x + square_width, water_height),
            (square_x + square_width, square_y),
            (square_x, square_y),
            (square_x, water_height),
            last(water_vertices),
        ]))
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
function generate_still_wedge_middle_square_geometry(output_dir;
        dx = 0.02,
        SimConstants = SimulationConstants{Float64}(; dx, c₀ = 42.48576250492629),
        water_level = nothing)
    polygons = still_wedge_middle_square_polygons()
    boundary = Multi([polygons.tank, polygons.square])
    return generate_still_wedge_2d_example(output_dir;
        dx, SimConstants, water_level, polygons, boundary,
        case_name = "StillWedge_MiddleSquare")
end


output_dir = normpath(joinpath(@__DIR__, "..", "input", "still_wedge_middle_square_generated"))
dx = 0.02
particles = generate_still_wedge_middle_square_geometry(output_dir; dx)
for region in particles
    @info "$(region.name): $(length(region.positions)) particles"
end
@info "Saved middle-square StillWedge geometry and particles" output_dir
