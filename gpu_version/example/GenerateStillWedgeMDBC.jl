# Generate the 2D still wedge case from polygons instead of hand made CSV files.
#
# The tank (walls, floor and the hollow wedge) and the water are described as
# Meshes.jl `PolyArea`s. The polygons are written as VTKHDF for ParaView, then
# sampled on one particle lattice with `RegularSampling`: lattice points on or
# inside the tank outline become boundary particles, points strictly inside the
# water become fluid particles, so the two never overlap. Fluid particles get
# the density whose pressure under the solver's equation of state is
# hydrostatic, so the case starts at rest. The particles are
# written as CSV input files for `SPHGeometry` and as VTKHDF for ParaView.
# No simulation is run and no GPU is needed.
#
# Run from the repository root with:
#     julia --project=gpu_version gpu_version/example/GenerateStillWedgeMDBC.jl [output_dir] [dx]
using SPHExampleGPU
using Meshes

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
function still_wedge_2d_polygons(; tank_width = 2.2, tank_height = 0.7, water_height = 0.5,
                                 wall_thickness = 0.04, wedge_apex = (1.1, 0.26),
                                 wedge_shell = 0.06)
    apex_x, apex_y = wedge_apex
    t = wall_thickness
    outer_half_width = apex_y                   # 45° slopes: half width = height
    inner_apex_y     = apex_y - wedge_shell
    inner_half_width = inner_apex_y + t         # inner V where it meets the floor bottom

    wedge_left  = (apex_x - outer_half_width, 0.0)
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
function generate_still_wedge_2d_example(output_dir; dx = 0.02,
        SimConstants = SimulationConstants{Float64}(; dx, c₀ = 42.48576250492629),
        water_level = nothing, polygons = still_wedge_2d_polygons(),
        boundary = polygons.tank, case_name = "StillWedge2D")
    isfinite(dx) && dx > 0 ||
        throw(ArgumentError("particle spacing dx must be finite and positive"))
    regions  = [
        ParticleRegion("Bound", boundary, Fixed),  # listed first: owns the shared surfaces
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    sampled = sample_particles(regions, dx)

    fluid_positions = reduce(vcat, (r.positions for r in sampled if r.type == Fluid))
    isempty(fluid_positions) &&
        throw(ArgumentError("particle spacing dx leaves the water region empty"))
    level = something(water_level, maximum(last, fluid_positions))
    particles = map(sampled) do region
        density = region.type == Fluid ?
            hydrostatic_density(region.positions, SimConstants; water_level = level) :
            fill(SimConstants.ρ₀, length(region.positions))
        (; region..., density)
    end

    mkpath(output_dir)
    prefix = joinpath(output_dir, "$(case_name)_Dp$(dx)")
    SavePolygonVTKHDF(joinpath(output_dir, "$(case_name)_Geometry.vtkhdf"), polygons)

    next_id = 0
    for region in particles
        next_id = write_particle_csv("$(prefix)_$(region.name).csv", region.positions;
                                     density = region.density, first_id = next_id)
    end

    (; ρ₀, c₀) = SimConstants
    positions = to_3d(reduce(vcat, region.positions for region in particles))
    density   = reduce(vcat, region.density for region in particles)
    pressure  = EquationOfStateGamma7.(density, c₀, ρ₀)
    types     = reduce(vcat, fill(Int8(region.type), length(region.positions)) for region in particles)
    markers   = reduce(vcat, fill(id, length(region.positions)) for (id, region) in enumerate(particles))
    SaveVTKHDF("$(prefix)_Particles.vtkhdf", positions,
               ["Density", "Pressure", "Type", "GroupMarker"], density, pressure, types, markers)

    return particles
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    output_dir = isempty(ARGS) ?
        normpath(joinpath(@__DIR__, "..", "input", "still_wedge_generated")) : ARGS[1]
    dx = length(ARGS) < 2 ? 0.02 : parse(Float64, ARGS[2])
    particles = generate_still_wedge_2d_example(output_dir; dx)
    for region in particles
        @info "$(region.name): $(length(region.positions)) particles"
    end
    @info "Saved StillWedge2D geometry and particles" output_dir
end