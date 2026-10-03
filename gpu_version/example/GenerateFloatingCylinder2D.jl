# Generate the 2D falling cylinder of DualSPHysics (examples/main/11_Floating,
# CaseFloatingSphereVal2D): a cylinder of radius 1 m, 1.2 times as dense as
# water, starts half submerged at the surface of a 10 m wide, 14 m deep tank
# and sinks under gravity. DualSPHysics uses periodic side boundaries; here
# they are fixed walls 4 m from the cylinder.
#
# The cylinder is sampled along its outline (`sampling = :conforming`), as
# concentric rings, so its surface is a true circle instead of a staircase;
# the tank and the water use the common lattice. Densities are hydrostatic
# below the free surface. No simulation is run and no GPU is needed; run the
# case with `example/FloatingCylinder2d.jl`.
#
# Run from the repository root with:
#     julia --project=gpu_version gpu_version/example/GenerateFloatingCylinder2D.jl \
#         [output_dir] [dx]
using SPHExampleGPU
using Meshes

"""
    floating_cylinder_2d_shapes(; dx = 0.025, width = 10.0, depth = 14.0,
                                tank_height = 16.0, radius = 1.0)

The tank (walls three spacings thick outside the wetted surface), the water
and the cylinder, centred on the free surface in the middle of the tank. The
water reaches half a spacing above `depth`, so its top particle layer lies at
`depth` like the DualSPHysics case.
"""
function floating_cylinder_2d_shapes(; dx = 0.025, width = 10.0, depth = 14.0,
                                     tank_height = 16.0, radius = 1.0)
    half = width / 2
    tank = polyline([(-half, tank_height), (-half, 0.0), (half, 0.0), (half, tank_height)];
                    thickness = 3dx, side = :right)
    water    = rectangle((-half, 0.0), width, depth + dx / 2)
    cylinder = circle((0.0, depth), radius; segments = 256)
    return (; tank, water, cylinder)
end

"""
    floating_cylinder_simulation_constants(FloatType = Float64; dx = 0.025)

Constants of the DualSPHysics case: `c₀ = coefsound * sqrt(g * hswl)` with
`coefsound = 30` and `hswl = 0.8`, `ρ₀ = 1000`, `g = 9.81`, `CFL = 0.2`.
"""
floating_cylinder_simulation_constants(FloatType = Float64; dx = 0.025) =
    SimulationConstants{FloatType}(; dx, ρ₀ = 1000, g = 9.81, c₀ = 30 * sqrt(9.81 * 0.8),
                                   δᵩ = 0.1, CFL = 0.2, ν₀ = 1e-6)

"""
    generate_floating_cylinder_2d(output_dir; dx = 0.025,
        SimConstants = floating_cylinder_simulation_constants(; dx))

Write `FloatingCylinder2D_Geometry.vtkhdf`, `FloatingCylinder2D_Dp<dx>_Particles.vtkhdf`
and the CSV files `FloatingCylinder2D_Dp<dx>_<Cylinder|Bound|Fluid>.csv` to
`output_dir`. Every particle below the top fluid layer gets the hydrostatic
density of its depth, the others `ρ₀`. Returns the sampled regions with their
densities.
"""
function generate_floating_cylinder_2d(output_dir; dx = 0.025,
        SimConstants = floating_cylinder_simulation_constants(; dx))
    dx > 0 || throw(ArgumentError("particle spacing dx must be positive"))
    shapes = floating_cylinder_2d_shapes(; dx)
    # The conforming cylinder comes first: the lattice keeps half a spacing from it.
    regions = [
        ParticleRegion("Cylinder", shapes.cylinder, Floating; sampling = :conforming),
        ParticleRegion("Bound", shapes.tank, Fixed),
        ParticleRegion("Fluid", shapes.water, Fluid),
    ]
    sampled = sample_particles(regions, dx)
    water_level = maximum(last, only(r for r in sampled if r.type == Fluid).positions)
    particles = map(sampled) do region
        density = hydrostatic_density(region.positions, SimConstants; water_level)
        (; region..., density)
    end

    mkpath(output_dir)
    SavePolygonVTKHDF(joinpath(output_dir, "FloatingCylinder2D_Geometry.vtkhdf"), shapes)
    prefix = joinpath(output_dir, "FloatingCylinder2D_Dp$(dx)")
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
    markers   = reduce(vcat, (fill(k, length(region.positions))
                              for (k, region) in enumerate(particles)))
    SaveVTKHDF("$(prefix)_Particles.vtkhdf", positions,
               ["Density", "Pressure", "Type", "GroupMarker"],
               density, pressure, types, markers)
    return particles
end


output_dir = length(ARGS) >= 1 ? ARGS[1] :
             normpath(joinpath(@__DIR__, "..", "input", "floating_cylinder_2d_generated"))
dx = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.025
particles = generate_floating_cylinder_2d(output_dir; dx)
for region in particles
    @info "$(region.name): $(length(region.positions)) particles"
end
@info "Saved FloatingCylinder2D geometry and particles" output_dir
