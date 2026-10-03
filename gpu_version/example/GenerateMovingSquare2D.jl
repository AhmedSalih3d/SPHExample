# Generate the 2D moving-square case from polygons instead of hand-made CSV files.
#
# The tank, water and moving square are described as Meshes.jl `PolyArea`s.
# They are sampled on one particle lattice, then written as CSV input files for
# `SPHGeometry` and as VTKHDF for ParaView. No simulation is run and no GPU is
# needed.
#
# Run from the repository root with:
#     julia --project=gpu_version gpu_version/example/GenerateMovingSquare2D.jl \
#         [output_dir] [dx]
using SPHExampleGPU
using Meshes

"""
    moving_square_2d_polygons(; tank_width = 10.0, tank_height = 5.0,
                               wall_thickness = 0.04,
                               square_lower_left = (1.0, 2.0), square_size = 1.0)

Return the closed tank, fluid region and initial moving square as `PolyArea`s,
in metres. The tank wall is a rectangular shell drawn outside the fluid domain
with `outline`; the fluid polygon has the square as a hole so fluid particles
do not overlap the moving body.
"""
function moving_square_2d_polygons(;
    tank_width = 10.0,
    tank_height = 5.0,
    wall_thickness = 0.04,
    square_lower_left = (1.0, 2.0),
    square_size = 1.0,
)
    if !(tank_width > 0 && tank_height > 0 && wall_thickness > 0 && square_size > 0)
        throw(
            ArgumentError(
                "tank dimensions, wall thickness, and square size " * "must be positive",
            ),
        )
    end

    square_x, square_y = square_lower_left
    if !(
        0 < square_x &&
        square_x + square_size < tank_width &&
        0 < square_y &&
        square_y + square_size < tank_height
    )
        throw(ArgumentError("the moving square must be strictly inside the tank"))
    end

    domain = rectangle((0.0, 0.0), tank_width, tank_height)
    tank = outline(domain; thickness = wall_thickness, side = :outward)
    body = square(square_lower_left, square_size)
    water = polygon(domain; holes = [body])

    return (; tank, water, square = body)
end

"""
    generate_moving_square_2d_example(output_dir; dx = 0.02,
        SimConstants = SimulationConstants{Float64}(; dx, c₀ = 28, g = 0))

Write the polygons (`MovingSquare2D_Geometry.vtkhdf`), sampled particles
(`MovingSquare2D_Dp<dx>_Particles.vtkhdf`) and CSV input files
(`MovingSquare2D_Dp<dx>_<Fixed|Fluid|Square>.csv`) to `output_dir`.

All particles start at `ρ₀`, matching the zero-gravity moving-square case.
Pass the simulation's `SimConstants` to use its reference density.
Returns the sampled particles per region, with their densities.
"""
function generate_moving_square_2d_example(
    output_dir;
    dx = 0.02,
    SimConstants = SimulationConstants{Float64}(; dx, c₀ = 28, g = 0),
)
    dx > 0 || throw(ArgumentError("particle spacing dx must be positive"))

    polygons = moving_square_2d_polygons()
    regions = [
        ParticleRegion("Fixed", polygons.tank, Fixed),
        ParticleRegion("Square", polygons.square, Moving),
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    sampled = sample_particles(regions, dx)
    particles = map(sampled) do region
        density = fill(SimConstants.ρ₀, length(region.positions))
        (; region..., density)
    end

    mkpath(output_dir)
    prefix = joinpath(output_dir, "MovingSquare2D_Dp$(dx)")
    SavePolygonVTKHDF(joinpath(output_dir, "MovingSquare2D_Geometry.vtkhdf"), polygons)

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
    positions = to_3d(reduce(vcat, region.positions for region in particles))
    density = reduce(vcat, region.density for region in particles)
    pressure = EquationOfStateGamma7.(density, c₀, ρ₀)
    types = reduce(
        vcat,
        (fill(Int8(region.type), length(region.positions)) for region in particles),
    )
    group_markers = Dict("Fixed" => 1, "Fluid" => 2, "Square" => 3)
    markers = reduce(
        vcat,
        (
            fill(group_markers[region.name], length(region.positions)) for
            region in particles
        ),
    )
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

output_dir = normpath(joinpath(@__DIR__, "..", "input", "moving_square_2d_generated"))
dx = 0.02
particles = generate_moving_square_2d_example(output_dir; dx)
for region in particles
    @info "$(region.name): $(length(region.positions)) particles"
end
@info "Saved MovingSquare2D geometry and particles" output_dir

