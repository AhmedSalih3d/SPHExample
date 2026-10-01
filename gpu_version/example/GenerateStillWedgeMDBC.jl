using SPHExampleGPU
using Meshes
using HDF5

"""
    still_wedge_polygons()

Return the fixed boundary and water as 2D `PolyArea`s, in metres.
The boundary follows the particle envelope of the dx = 0.02 StillWedge example,
including its grid-rounded wedge peak at (1.1, 0.26). The water surface is at 0.5 m.
"""
function still_wedge_polygons()
    tank_width = 2.2
    tank_height = 0.7
    wall_thickness = 0.04
    water_height = 0.5
    wedge_left = (0.84, 0.0)
    wedge_peak = (1.1, 0.26)
    wedge_right = (1.36, 0.0)

    fixed_boundary = PolyArea([
        (-wall_thickness, -wall_thickness),
        (tank_width + wall_thickness, -wall_thickness),
        (tank_width + wall_thickness, tank_height),
        (tank_width, tank_height),
        (tank_width, 0.0),
        wedge_right,
        wedge_peak,
        wedge_left,
        (0.0, 0.0),
        (0.0, tank_height),
        (-wall_thickness, tank_height),
    ])

    water = PolyArea([
        (0.0, 0.0),
        wedge_left,
        wedge_peak,
        wedge_right,
        (tank_width, 0.0),
        (tank_width, water_height),
        (0.0, water_height),
    ])

    return (; fixed_boundary, water)
end

"""
    save_polygon_vtkhdf(filepath, polygons)

Save 2D Meshes polygons as static VTKHDF PolyData in the solver's XY plane (z = 0).
Triangulate the concave regions for reliable visualization; `Region` cell data
identifies each input polygon by its one-based index. No particles are generated.
"""
function save_polygon_vtkhdf(filepath::AbstractString, polygons)
    points = NTuple{3, Float64}[]
    connectivity = Int64[]
    offsets = Int64[0]
    regions = Int32[]

    for (region, polygon) in enumerate(polygons)
        mesh = discretize(polygon)
        point_offset = length(points)
        for vertex in vertices(mesh)
            x, y = Meshes.ustrip.(to(vertex))
            push!(points, (x, y, 0.0))
        end
        for cell in elements(topology(mesh))
            append!(connectivity, point_offset .+ indices(cell) .- 1)
            push!(offsets, length(connectivity))
            push!(regions, region)
        end
    end

    mkpath(dirname(abspath(filepath)))
    h5open(filepath, "w") do file
        root = create_group(file, "VTKHDF")
        attrs(root)["Version"] = Int32[2, 3]
        SPHExampleGPU.ProduceHDFVTK.write_ascii_attribute(root, "Type", "PolyData")
        root["NumberOfPoints"] = Int64[length(points)]
        root["Points"] = stack(points)

        for cell_type in ("Vertices", "Lines", "Polygons", "Strips")
            group = create_group(root, cell_type)
            is_polygon = cell_type == "Polygons"
            group["NumberOfCells"] = Int64[is_polygon ? length(regions) : 0]
            group["NumberOfConnectivityIds"] =
                Int64[is_polygon ? length(connectivity) : 0]
            group["Connectivity"] = is_polygon ? connectivity : Int64[]
            group["Offsets"] = is_polygon ? offsets : Int64[0]
        end
        cell_data = create_group(root, "CellData")
        cell_data["Region"] = regions
    end

    return filepath
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    length(ARGS) <= 1 || throw(ArgumentError("Expected at most one output file path"))
    output_path = isempty(ARGS) ?
        joinpath(@__DIR__, "StillWedgeMDBC_Geometry.vtkhdf") : abspath(only(ARGS))
    polygons = still_wedge_polygons()
    save_polygon_vtkhdf(output_path, values(polygons))
    @info "Saved StillWedge polygon geometry" output_path
end