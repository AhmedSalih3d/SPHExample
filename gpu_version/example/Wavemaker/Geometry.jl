module WavemakerGeometry

using SPHExampleGPU
using StaticArrays: SVector
using CSV
using Meshes: vertices, to, ustrip

export wavemaker_geometry, save_wavemaker_geometry

"""
    wavemaker_geometry()

Reproduce the initial particle layout of DualSPHysics 5.4 main/06_Wavemaker.
The small CSV stores GenCase's rasterized fixed-wall cross section as integer
lattice coordinates. Fluid and piston particles are generated from that skin;
all sections are repeated over the reference's 100 periodic y rows.
"""
function wavemaker_geometry(; dx = 0.02)
    dx == 0.02 || throw(ArgumentError("The matched GenCase layout uses dx = 0.02; regenerate BoundarySection.csv before changing resolution"))
    tank_length, tank_width, tank_height = 5.0, 2.0, 1.5
    water_depth, beach_start, beach_height = 0.75, 1.0, 1.1
    piston_front, piston_thickness = 0.06, 0.06

    # GenCase's lattice: x = ix*dx, y = (iy+1/2)*dx, z = (iz+1/2)*dx.
    # Preserve its rasterized skin, including the slope's corner cells.
    rows = CSV.File(joinpath(@__DIR__, "BoundarySection.csv"))
    fixed_indices = [(Int(row.x_index), Int(row.z_index)) for row in rows]
    bed_top = Dict(ix => maximum(iz for (x, iz) in fixed_indices if x == ix) for ix in 0:250)
    piston_indices = [(ix, iz) for ix in 0:3 for iz in 1:75]
    water_indices = [(ix, iz) for ix in 4:250 for iz in (bed_top[ix] + 1):36]
    extrude(indices) = [SVector(ix * dx, (iy + 0.5) * dx, (iz + 0.5) * dx)
        for (ix, iz) in indices for iy in 0:99]
    positions = (; fixed = extrude(fixed_indices), piston = extrude(piston_indices), water = extrude(water_indices))
    @assert length.(values(positions)) == (31600, 30000, 401600)

    bed_profile = [(0.0, 0.0), (beach_start, 0.0), (tank_length, beach_height), (tank_length, tank_height)]
    fixed_section = polyline(bed_profile; thickness = dx, side = :right)
    piston_section = rectangle((0.0, 0.0), piston_thickness, tank_height)
    shoreline = beach_start + water_depth * (tank_length - beach_start) / beach_height
    water_section = polygon([(0.0, 0.0), (beach_start, 0.0), (shoreline, water_depth), (0.0, water_depth)])
    surfaces = (
        fixed = extrude_surface(fixed_section, 0.0, tank_width),
        piston = extrude_surface(piston_section, 0.0, tank_width),
        water = extrude_surface(water_section, 0.0, tank_width),
    )
    periodicity = PeriodicBoundary{3}(2; min = 0.0, max = tank_width)
    water_level = maximum(last, positions.water)
    return (; positions, periodicity, surfaces, water_level, water_depth, dx,
        tank_length, tank_width, tank_height, beach_start, beach_height, piston_front, piston_thickness)
end

bed_clearance(x, z) = x <= 1.0 ? z : (z - 0.275 * (x - 1.0)) / sqrt(1 + 0.275^2)

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

end # module