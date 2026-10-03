# Showcase of the polygon drawing helpers (`src/PolygonDrawing.jl`).
#
# A 2D tank with a cylinder, a triangular wedge, a ramp, an arc shaped baffle
# and a tilted floating square, and a 3D tank built from prisms with a pillar
# and a wedge. Each scene is sampled on one particle lattice and written as
# polygon and particle VTKHDF files for ParaView plus CSV input files for
# `SPHGeometry`. No simulation is run and no GPU is needed.
#
# Run from the repository root with:
#     julia --project=gpu_version gpu_version/example/GenerateShapesShowcase.jl \
#         [output_dir] [dx]
using SPHExampleGPU
using Meshes

"""
    showcase_2d_shapes(; dx = 0.02)

The 2D scene as named shapes. Walls are three particle layers thick past their
surface (`3dx`), so the wetted surfaces lie exactly on the drawn paths.
"""
function showcase_2d_shapes(; dx = 0.02)
    t = 3dx
    width, height, depth = 3.0, 1.2, 0.6

    # Open tank: walking down the left wall, along the floor and up the right
    # wall, the outside is on the right.
    tank = polyline([(0, height), (0, 0), (width, 0), (width, height)];
                    thickness = t, side = :right)
    cylinder = circle((0.5, 0.3), 0.15)
    wedge    = triangle((1.0, 0.0), (1.4, 0.0), (1.2, 0.25))
    # The ramp surface is the drawn line; its wall lies below it (right side).
    ramp     = line((2.2, 0.0), (width, 0.35); thickness = t, side = :right)
    # A U shaped baffle hanging into the water, centred on its arc.
    baffle   = polyline(arc((1.85, 0.75), 0.3, π, 2π); thickness = 2dx)
    body     = square((2.55, 0.45), 0.15; angle = π / 4, centered = true)
    water    = rectangle((0, 0), width, depth)

    return (; boundary = Multi([tank, cylinder, wedge, ramp, baffle]), body, water)
end

"""
    showcase_3d_shapes(; dx = 0.02)

The 3D scene: a closed bottom tank made of wall and floor prisms, a
cylindrical pillar, a triangular wedge on the floor and a block of water.
"""
function showcase_3d_shapes(; dx = 0.02)
    t = 3dx
    height = 0.6
    interior = rectangle((0, 0), 1.0, 0.6)
    walls  = prism(outline(interior; thickness = t), -t, height)
    floor_ = prism(offset_polygon(interior, t), -t, 0.0)
    pillar = prism(circle((0.3, 0.3), 0.08), 0.0, height)
    wedge  = prism(triangle((0.6, 0.15), (0.9, 0.15), (0.75, 0.45)), 0.0, 0.15)
    water  = prism(interior, 0.0, 0.3)

    return (; boundary = (walls, floor_, pillar, wedge), water)
end

"""
    sample_and_save(output_dir, case_name, shapes, regions, dx, SimConstants)

Sample `regions`, give fluid particles hydrostatic densities and the others
`ρ₀`, and write `<case_name>_Geometry.vtkhdf`, `<case_name>_Dp<dx>_Particles.vtkhdf`
and one CSV per region to `output_dir`. Returns the sampled regions.
"""
function sample_and_save(output_dir, case_name, shapes, regions, dx, SimConstants)
    particles = map(sample_particles(regions, dx)) do region
        density = region.type == Fluid ?
            hydrostatic_density(region.positions, SimConstants) :
            fill(SimConstants.ρ₀, length(region.positions))
        (; region..., density)
    end

    mkpath(output_dir)
    SavePolygonVTKHDF(joinpath(output_dir, "$(case_name)_Geometry.vtkhdf"), shapes)
    prefix = joinpath(output_dir, "$(case_name)_Dp$(dx)")
    next_id = 0
    for region in particles
        next_id = write_particle_csv("$(prefix)_$(region.name).csv", region.positions;
                                     density = region.density, first_id = next_id)
    end

    (; ρ₀, c₀) = SimConstants
    positions = reduce(vcat, region.positions for region in particles)
    positions = length(first(positions)) == 2 ? to_3d(positions) : positions
    density   = reduce(vcat, region.density for region in particles)
    pressure  = EquationOfStateGamma7.(density, c₀, ρ₀)
    types     = reduce(vcat, (fill(Int8(region.type), length(region.positions))
                              for region in particles))
    markers   = reduce(vcat, (fill(id, length(region.positions))
                              for (id, region) in enumerate(particles)))
    SaveVTKHDF("$(prefix)_Particles.vtkhdf", positions,
               ["Density", "Pressure", "Type", "GroupMarker"],
               density, pressure, types, markers)
    return particles
end

"""
    generate_shapes_showcase(output_dir; dx = 0.02,
        SimConstants = SimulationConstants{Float64}(; dx, c₀ = 25.0))

Build, sample and write both showcase scenes (`Showcase2D_*` and
`Showcase3D_*` files). Returns `(; two_d, three_d)` with the sampled regions.
"""
function generate_shapes_showcase(output_dir; dx = 0.02,
        SimConstants = SimulationConstants{Float64}(; dx, c₀ = 25.0))
    shapes_2d = showcase_2d_shapes(; dx)
    two_d = sample_and_save(output_dir, "Showcase2D", shapes_2d, [
            ParticleRegion("Bound", shapes_2d.boundary, Fixed),
            ParticleRegion("Body", shapes_2d.body, Moving),
            ParticleRegion("Fluid", shapes_2d.water, Fluid),
        ], dx, SimConstants)

    shapes_3d = showcase_3d_shapes(; dx)
    three_d = sample_and_save(output_dir, "Showcase3D", shapes_3d, [
            ParticleRegion("Bound", shapes_3d.boundary, Fixed),
            ParticleRegion("Fluid", shapes_3d.water, Fluid),
        ], dx, SimConstants)

    return (; two_d, three_d)
end


output_dir = normpath(joinpath(@__DIR__, "..", "input", "shapes_showcase_generated"))
dx = 0.02
cases = generate_shapes_showcase(output_dir; dx)
for (scene, particles) in pairs(cases), region in particles
    @info "$(scene) $(region.name): $(length(region.positions)) particles"
end
@info "Saved the shapes showcase" output_dir
