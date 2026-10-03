# Generate the 2D Re = 100 lid-driven cavity from the Rocky 2025 R1
# verification manual. A 1 m square is filled with fluid; the bottom and side
# walls are stationary, and the top lid has a prescribed +x velocity.
#
# Run from the repository root with:
#     julia --project=gpu_version gpu_version/example/GenerateLidDrivenCavity2D.jl \
#         [output_dir] [dx]
using SPHExampleGPU
using Meshes

const LID_CAVITY_DENSITY = 10.0
const LID_CAVITY_DYNAMIC_VISCOSITY = 0.1
const LID_CAVITY_SOUND_SPEED = 10.0
const LID_CAVITY_LENGTH = 1.0
const LID_CAVITY_WALL_THICKNESS = 0.05
const LID_CAVITY_LID_SPEED = 1.0

"""
    lid_driven_cavity_2d_shapes(; side_length = 1.0, wall_thickness = 0.05)

Stationary bottom and side walls, a separate top lid, and the fluid-filled
square cavity. Both the stationary walls and the lid extend outwards from the
cavity, leaving the full 1 m × 1 m square for the fluid.
"""
function lid_driven_cavity_2d_shapes(; side_length = LID_CAVITY_LENGTH,
                                     wall_thickness = LID_CAVITY_WALL_THICKNESS)
    (isfinite(side_length) && side_length > 0) ||
        throw(ArgumentError("side_length must be finite and positive"))
    (isfinite(wall_thickness) && 0 < wall_thickness < side_length) ||
        throw(ArgumentError("wall_thickness must be finite, positive and less than side_length"))

    walls = polyline([(0.0, side_length), (0.0, 0.0), (side_length, 0.0),
                      (side_length, side_length)];
                     thickness = wall_thickness, side = :right)
    lid = line((0.0, side_length), (side_length, side_length);
               thickness = wall_thickness, side = :left)
    fluid = rectangle((0.0, 0.0), side_length, side_length)
    return (; walls, lid, fluid)
end

"""
    lid_driven_cavity_2d_constants(::Type{T} = Float64; dx = 0.01)

Rocky verification inputs in SPH form: `ρ₀ = 10 kg/m³`, dynamic viscosity
`μ = 0.1 Pa·s`, hence kinematic viscosity `ν₀ = μ / ρ₀ = 0.01 m²/s`;
`c₀ = 10 m/s`, zero gravity and `CFL = 0.2`. For `L = 1 m` and lid speed
`U = 1 m/s` these give `Re = ρ₀ U L / μ = 100`.
"""
function lid_driven_cavity_2d_constants(::Type{T} = Float64; dx = 0.01) where {T <: AbstractFloat}
    (isfinite(dx) && dx > 0) || throw(ArgumentError("dx must be finite and positive"))
    ρ₀ = T(LID_CAVITY_DENSITY)
    μ  = T(LID_CAVITY_DYNAMIC_VISCOSITY)
    return SimulationConstants{T}(; dx = T(dx), ρ₀, g = zero(T),
                                   c₀ = T(LID_CAVITY_SOUND_SPEED),
                                   ν₀ = μ / ρ₀, CFL = T(0.2))
end

"""
    generate_lid_driven_cavity_2d(output_dir; dx = 0.01,
        SimConstants = lid_driven_cavity_2d_constants(; dx))

Write `LidDrivenCavity2D_Geometry.vtkhdf`,
`LidDrivenCavity2D_Dp<dx>_Particles.vtkhdf` and `Fixed`, `Lid`, and `Fluid`
CSV particle files to `output_dir`. All particles start at the reference
density. Returns the sampled regions.
"""
function generate_lid_driven_cavity_2d(output_dir; dx = 0.01,
        SimConstants = lid_driven_cavity_2d_constants(; dx))
    (isfinite(dx) && 0 < dx <= LID_CAVITY_LENGTH) ||
        throw(ArgumentError("dx must be finite, positive and no larger than the cavity"))

    shapes = lid_driven_cavity_2d_shapes()
    regions = [
        ParticleRegion("Fixed", shapes.walls, Fixed),
        ParticleRegion("Lid", shapes.lid, Moving),
        ParticleRegion("Fluid", shapes.fluid, Fluid),
    ]
    sampled = sample_particles(regions, dx)
    particles = map(sampled) do region
        density = fill(SimConstants.ρ₀, length(region.positions))
        (; region..., density)
    end

    mkpath(output_dir)
    SavePolygonVTKHDF(joinpath(output_dir, "LidDrivenCavity2D_Geometry.vtkhdf"), shapes)
    prefix = joinpath(output_dir, "LidDrivenCavity2D_Dp$(dx)")
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

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    output_dir = length(ARGS) >= 1 ? ARGS[1] :
                 normpath(joinpath(@__DIR__, "..", "input", "lid_driven_cavity_2d_generated"))
    dx = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.01
    particles = generate_lid_driven_cavity_2d(output_dir; dx)
    for region in particles
        @info "$(region.name): $(length(region.positions)) particles"
    end
    @info "Saved LidDrivenCavity2D geometry and particles" output_dir
end
