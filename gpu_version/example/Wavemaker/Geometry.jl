module WavemakerGeometry

using SPHExampleGPU
using StaticArrays: SVector
using Meshes: vertices, to, ustrip

export wavemaker_geometry, save_wavemaker_geometry

"""
    wavemaker_geometry(; dx = 0.02, boundary_layers = 3)

Build the tank, piston and water for DualSPHysics `main/06_Wavemaker`.
Coordinates are `(x, y, z)` in metres; z is vertical and y is periodic.
This function only builds host geometry and does not require a CUDA device.
"""
function wavemaker_geometry(; dx = 0.02, boundary_layers = 3)
    isfinite(dx) && 0 < dx <= 0.1 || throw(ArgumentError("dx must be finite and in (0, 0.1]"))
    boundary_layers isa Integer && boundary_layers >= 2 ||
        throw(ArgumentError("boundary_layers must be an integer of at least 2"))
    dx = SPHExampleGPU.ParticleGenerator.lattice_spacing(dx)

    tank_length = 5.0
    tank_width = 2.0
    tank_height = 1.5
    water_depth = 0.75
    beach_start = 1.0
    beach_height = 1.1
    piston_thickness = max(0.06, (boundary_layers - 1) * dx)
    wall_thickness = (boundary_layers - 1) * dx

    # Draw the x/z section first. Fixed walls grow below/outside the water.
    # There is no fixed left wall: the sliding piston closes that end.
    bed_profile = [(0.0, 0.0), (beach_start, 0.0),
        (tank_length, beach_height), (tank_length, tank_height)]
    fixed_section = polyline(bed_profile; thickness = wall_thickness, side = :right)
    piston_section = rectangle((0.0, 0.0), piston_thickness, tank_height)

    beach_slope = beach_height / (tank_length - beach_start)
    shoreline = beach_start + water_depth / beach_slope
    water_section = polygon([(0.0, 0.0), (beach_start, 0.0),
        (shoreline, water_depth), (0.0, water_depth)])

    # One shared lattice gives each point to one region. The fixed bed owns
    # shared nodes, then the piston, then the water, avoiding duplicate particles.
    sections = sample_particles([
        ParticleRegion("Fixed", fixed_section, Fixed),
        ParticleRegion("Piston", piston_section, Moving),
        ParticleRegion("Water", water_section, Fluid),
    ], dx)

    # Keep both endpoint rows, as in GenCase. Periodic faces lie half a spacing
    # beyond them, so the seam has exactly the same spacing as the interior.
    y_rows = collect(0:dx:tank_width)
    extrude_positions(section) = [SVector(x, y, z) for (x, z) in section.positions for y in y_rows]
    positions = (
        fixed = extrude_positions(sections[1]),
        piston = extrude_positions(sections[2]),
        water = extrude_positions(sections[3]),
    )
    periodicity = PeriodicBoundary{3}(2; min = -dx / 2, max = last(y_rows) + dx / 2)
    water_level = maximum(last, positions.water)

    surfaces = (
        fixed = extrude_surface(fixed_section, first(y_rows), last(y_rows)),
        piston = extrude_surface(piston_section, first(y_rows), last(y_rows)),
        water = extrude_surface(water_section, first(y_rows), last(y_rows)),
    )
    return (; positions, periodicity, surfaces, water_level, water_depth,
        tank_length, tank_width, tank_height, beach_start, beach_height, piston_thickness)
end

# The package's prism helper extrudes along z. This case instead spans y,
# keeping the beach section in x/z and gravity along -z.
function extrude_surface(section, y_min, y_max)
    outline = [SVector{2,Float64}(ustrip.(to(point))) for point in vertices(section)]
    n = length(outline)
    points = [SVector(x, y, z) for y in (y_min, y_max) for (x, z) in outline]
    faces = [collect(1:n), collect(2n:-1:(n + 1))]
    for i in 1:n
        j = mod1(i + 1, n)
        push!(faces, [i, i + n, j + n, j])
    end
    return PolygonSurface(points, faces)
end

"""
    save_wavemaker_geometry(directory, case, geometry)

Write readable Fixed/Piston/Water particle CSVs and a static polygon VTKHDF
preview. The simulation uses the generated in-memory particles directly.
"""
function save_wavemaker_geometry(directory, case, geometry)
    mkpath(directory)
    next_id = 0
    for (name, group) in zip(("Fixed", "Piston", "Water"), geometry)
        next_id = write_particle_csv(joinpath(directory, "Wavemaker_$(name).csv"),
            group.Particles.Position; density = group.Particles.Density, first_id = next_id)
    end
    SavePolygonVTKHDF(joinpath(directory, "Wavemaker_Geometry.vtkhdf"), case.surfaces)
    return nothing
end

end # module WavemakerGeometry
