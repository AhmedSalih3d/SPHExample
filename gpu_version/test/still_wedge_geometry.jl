using Test
using CSV
using HDF5

include(joinpath(@__DIR__, "..", "example", "GenerateStillWedgeMDBC.jl"))

function contains_particle(point, polygon::PolyArea)
    point in polygon || any(segments(boundary(polygon))) do segment
        distance = Meshes.ustrip(
            Meshes.evaluate(Meshes.Euclidean(), point, segment))
        distance <= 1e-12
    end
end

contains_particle(point, polygons::Multi) =
    any(polygon -> contains_particle(point, polygon), parent(polygons))

@testset "StillWedge polygon geometry" begin
    polygons = still_wedge_polygons()
    @test polygons.fixed_boundary isa Multi
    @test length(parent(polygons.fixed_boundary)) == 3
    @test polygons.water isa PolyArea
    @test isapprox(Meshes.ustrip(measure(polygons.fixed_boundary)), 0.1972)
    @test isapprox(Meshes.ustrip(measure(polygons.water)), 1.0324)
    @test contains_particle(Point(1.1, 0.2), polygons.fixed_boundary)
    @test !(Point(1.1, 0.2) in polygons.water)
    @test Point(1.1, 0.4) in polygons.water
    @test !contains_particle(Point(1.1, 0.4), polygons.fixed_boundary)
    @test !contains_particle(Point(1.1, -0.02), polygons.fixed_boundary)
    @test contains_particle(Point(0.3, -0.02), polygons.fixed_boundary)
    @test contains_particle(Point(1.8, -0.02), polygons.fixed_boundary)
    @test !contains_particle(Point(1.1, 0.6), polygons.fixed_boundary)
    @test !(Point(1.1, 0.6) in polygons.water)

    input_dir = joinpath(@__DIR__, "..", "input", "still_wedge")
    for (suffix, polygon) in (("Bound", polygons.fixed_boundary),
                              ("Fluid", polygons.water))
        reference = CSV.File(joinpath(input_dir, "StillWedge_Dp0.02_$suffix.csv"))
        @test all(reference) do row
            point = Point(row[Symbol("Points:0")], row[Symbol("Points:2")])
            contains_particle(point, polygon)
        end
    end

    mktempdir() do directory
        path = joinpath(directory, "geometry", "StillWedge.vtkhdf")
        @test SavePolygonVTKHDF(path, polygons) == path
        h5open(path, "r") do file
            root = file["VTKHDF"]
            @test attrs(root)["Type"] == "PolyData"
            @test attrs(root)["Version"] == Int32[2, 3]
            @test read(root["NumberOfPoints"]) == [22]
            points = read(root["Points"])
            @test size(points) == (3, 22)
            @test all(iszero, points[3, :])
            @test extrema(points[1, :]) == (-0.04, 2.24)
            @test extrema(points[2, :]) == (-0.04, 0.7)

            cells = root["Polygons"]
            @test read(cells["NumberOfCells"]) == [14]
            @test read(cells["NumberOfConnectivityIds"]) == [42]
            connectivity = read(cells["Connectivity"])
            offsets = read(cells["Offsets"])
            cell_regions = read(root["CellData/Region"])
            @test offsets == collect(0:3:42)
            @test all(id -> 0 <= id < 22, connectivity)
            @test count(==(1), cell_regions) == 9
            @test count(==(2), cell_regions) == 5

            areas = zeros(2)
            for cell in eachindex(cell_regions)
                ids = connectivity[offsets[cell] + 1:offsets[cell + 1]] .+ 1
                a, b, c = eachcol(points[1:2, ids])
                signed_area = ((b[1] - a[1]) * (c[2] - a[2]) -
                               (b[2] - a[2]) * (c[1] - a[1])) / 2
                @test signed_area > 0
                areas[cell_regions[cell]] += signed_area
            end
            @test isapprox(areas, [0.1972, 1.0324])

            for cell_type in ("Vertices", "Lines", "Strips")
                group = root[cell_type]
                @test read(group["NumberOfCells"]) == [0]
                @test read(group["NumberOfConnectivityIds"]) == [0]
                @test isempty(read(group["Connectivity"]))
                @test read(group["Offsets"]) == [0]
            end
        end
    end
end
