using SPHExampleGPU
using Meshes

"""
    still_wedge_polygons()

Return the fixed boundary and water as 2D polygon regions, in metres.
Separate boundary components leave the particle-free region below the wedge open.
"""
function still_wedge_polygons()
    tank_width = 2.2
    tank_height = 0.7
    wall_thickness = 0.04
    water_height = 0.5
    wedge_base_left = (0.84, 0.0)
    wedge_peak = (1.1, 0.26)
    wedge_base_right = (1.36, 0.0)

    left_wall = PolyArea([
        (-wall_thickness, -wall_thickness),
        (0.86, -wall_thickness),
        (0.90, 0.0),
        (0.0, 0.0),
        (0.0, tank_height),
        (-wall_thickness, tank_height),
    ])
    wedge = PolyArea([wedge_base_left, wedge_base_right, wedge_peak])
    right_wall = PolyArea([
        (1.34, -wall_thickness),
        (tank_width + wall_thickness, -wall_thickness),
        (tank_width + wall_thickness, tank_height),
        (tank_width, tank_height),
        (tank_width, 0.0),
        (1.30, 0.0),
    ])
    fixed_boundary = Multi([left_wall, wedge, right_wall])

    water = PolyArea([
        (0.0, 0.0),
        wedge_base_left,
        wedge_peak,
        wedge_base_right,
        (tank_width, 0.0),
        (tank_width, water_height),
        (0.0, water_height),
    ])

    return (; fixed_boundary, water)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) <= 1 || throw(ArgumentError("Expected at most one output file path"))
    output_path = isempty(ARGS) ?
        joinpath(@__DIR__, "StillWedgeMDBC_Geometry.vtkhdf") : abspath(only(ARGS))
    polygons = still_wedge_polygons()
    SavePolygonVTKHDF(output_path, polygons)
    @info "Saved StillWedge polygon geometry" output_path
end