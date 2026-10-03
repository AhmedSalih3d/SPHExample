# Generate the 2D dam-break case from polygons instead of hand-made CSV files.
#
# The tank and the initial water column are described as Meshes.jl `PolyArea`s.
# Both regions are sampled on one particle lattice, then written as CSV input
# files for `SPHGeometry` and as VTKHDF for ParaView. No simulation is run and
# no GPU is needed.
#
# Run from the repository root with:
#     julia --project=gpu_version gpu_version/example/GenerateDamBreak2DMDBC.jl \
#         [output_dir] [dx]
using SPHExampleGPU
using Meshes

"""
    dam_break_2d_polygons(; tank_width = 4.0, tank_height = 3.0,
                           water_width = 1.0, water_height = 2.0,
                           wall_thickness = 0.06)

Return the open tank and initial water column of the 2D dam-break case as
`PolyArea`s, in metres. The tank is a three-sided shell drawn along the wetted
surface with `polyline`: its floor and side walls extend `wall_thickness`
outside the fluid domain. The water starts in the left corner and is open at
the top.
"""
function dam_break_2d_polygons(; tank_width = 4.0, tank_height = 3.0,
                                water_width = 1.0, water_height = 2.0,
                                wall_thickness = 0.06)
    if !(0 < water_width <= tank_width) ||
       !(0 < water_height <= tank_height) ||
       !(wall_thickness > 0)
        throw(ArgumentError("water dimensions must fit in a positive tank " *
                            "and wall thickness"))
    end

    # Walking down the left wall, along the floor and up the right wall, the
    # outside of the tank is on the right.
    wetted_surface = [(0.0, tank_height), (0.0, 0.0),
                      (tank_width, 0.0), (tank_width, tank_height)]
    tank  = polyline(wetted_surface; thickness = wall_thickness, side = :right)
    water = rectangle((0.0, 0.0), water_width, water_height)

    return (; tank, water)
end

"""
    generate_dam_break_2d_example(output_dir; dx = 0.02,
        SimConstants = SimulationConstants{Float64}(; dx, c₀ = 88.14487860902641),
        water_level = nothing)

Write the tank and water polygons (`DamBreak2D_Geometry.vtkhdf`), the sampled
particles (`DamBreak2D_Dp<dx>_Particles.vtkhdf`) and the CSV input files
(`DamBreak2D_Dp<dx>_Bound.csv`, `..._Fluid.csv`) to `output_dir`.

Boundary particles start at `ρ₀`. Fluid particles start in hydrostatic
equilibrium: their density is the inverse of the solver's equation of state
for `P = ρ₀ g (water_level - y)`. `water_level = nothing` uses the highest
fluid particle. Pass the `SimConstants` of the simulation so that `ρ₀`, `g`
and `c₀` match the case.
Returns the sampled particles per region, with their densities.
"""
function generate_dam_break_2d_example(output_dir; dx = 0.02,
        SimConstants = SimulationConstants{Float64}(; dx, c₀ = 88.14487860902641),
        water_level = nothing)
    polygons = dam_break_2d_polygons()
    regions  = [
        ParticleRegion("Bound", polygons.tank, Fixed),
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    sampled = sample_particles(regions, dx)

    fluid_positions = reduce(vcat, (r.positions for r in sampled if r.type == Fluid))
    level = something(water_level, maximum(last, fluid_positions))
    particles = map(sampled) do region
        density = region.type == Fluid ?
            hydrostatic_density(region.positions, SimConstants; water_level = level) :
            fill(SimConstants.ρ₀, length(region.positions))
        (; region..., density)
    end

    mkpath(output_dir)
    prefix = joinpath(output_dir, "DamBreak2D_Dp$(dx)")
    SavePolygonVTKHDF(joinpath(output_dir, "DamBreak2D_Geometry.vtkhdf"), polygons)

    next_id = 0
    for region in particles
        next_id = write_particle_csv("$(prefix)_$(region.name).csv", region.positions;
                                     density = region.density, first_id = next_id)
    end

    (; ρ₀, c₀) = SimConstants
    positions = to_3d(reduce(vcat, region.positions for region in particles))
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


output_dir = normpath(joinpath(@__DIR__, "..", "input", "dam_break_2d_generated"))
dx = 0.02
particles = generate_dam_break_2d_example(output_dir; dx)
for region in particles
    @info "$(region.name): $(length(region.positions)) particles"
end
@info "Saved DamBreak2D geometry and particles" output_dir
