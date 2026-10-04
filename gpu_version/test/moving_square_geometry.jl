using Test
using SPHExampleGPU
using CSV
using HDF5
using StaticArrays
using Meshes

include(joinpath(@__DIR__, "..", "example", "GenerateMovingSquare2D.jl"))

moving_square_lattice_nodes(positions, dx) =
    Set(Tuple(round.(Int, collect(position) ./ dx)) for position in positions)

function moving_square_reference_nodes(name, dx)
    path = joinpath(@__DIR__, "..", "input", "moving_square_2d",
                    "MovingSquare_Dp$(dx)_$(name).csv")
    rows = CSV.File(path)
    positions = ((row[Symbol("Points:0")], row[Symbol("Points:2")]) for row in rows)
    return moving_square_lattice_nodes(positions, dx)
end

function polygon_motion_region_bounds(path, region_id, axis)
    return h5open(path, "r") do file
        root = file["VTKHDF"]
        regions = read(root["CellData/Region"])
        offsets = read(root["Polygons/Offsets"])
        connectivity = read(root["Polygons/Connectivity"])
        point_ids = Int[]
        for cell in eachindex(regions)
            regions[cell] == region_id || continue
            append!(point_ids, connectivity[offsets[cell] + 1:offsets[cell + 1]] .+ 1)
        end
        return extrema(read(root["Points"])[axis, unique(point_ids)])
    end
end

@testset "Polygon motion sequence export" begin
    mktempdir() do directory
        stationary = rectangle((0.0, 0.0), 0.5, 0.5)
        square = rectangle((1.0, 2.0), 1.0, 1.0)
        regions = (; stationary, square)
        motions = (
            stationary = MotionDetails{2, Float64}(
                Velocity = 9.0,
                StartTime = 0.0,
                Duration = 1.0,
                Direction = SVector{2, Float64}(1.0, 0.0),
                MoveParticles = false,
            ),
            square = MotionDetails{2, Float64}(
                Velocity = 2.0,
                StartTime = 0.5,
                Duration = 1.0,
                Direction = SVector{2, Float64}(1.0, 0.0),
            ),
        )
        times = [0.0, 0.5, 1.0, 1.5, 2.0]
        path = joinpath(directory, "preview.pvd")

        @test SavePolygonMotionSequence(path, regions; motions, times) == path
        @test isfile(path)
        collection = replace(read(path, String), "\\" => "/")
        @test count("<DataSet ", collection) == length(times)
        @test occursin("timestep=\"0.0\"", collection)
        @test occursin("preview_frames/frame_000001.vtkhdf", collection)
        @test all(isfile(joinpath(directory, "preview_frames",
                                 "frame_$(lpad(string(i), 6, '0')).vtkhdf"))
                  for i in eachindex(times))

        expected_x = [(1.0, 2.0), (1.0, 2.0), (2.0, 3.0),
                      (3.0, 4.0), (3.0, 4.0)]
        for (i, bounds) in enumerate(expected_x)
            frame = joinpath(directory, "preview_frames",
                             "frame_$(lpad(string(i), 6, '0')).vtkhdf")
            @test polygon_motion_region_bounds(frame, 1, 1) == (0.0, 0.5)
            @test collect(polygon_motion_region_bounds(frame, 2, 1)) ≈ collect(bounds)
            @test polygon_motion_region_bounds(frame, 2, 2) == (2.0, 3.0)
        end

        @test_throws ArgumentError SavePolygonMotionSequence(
            path, regions; motions, times = [0.0, 0.0])
        @test_throws ArgumentError SavePolygonMotionSequence(
            path, regions; motions = (; missing = only(values(motions))), times)
    end

    @testset "3D polygon motion" begin
        mktempdir() do directory
            block = prism(rectangle((0.0, 0.0), 1.0, 1.0), 0.0, 1.0)
            motion = MotionDetails{3, Float64}(
                Velocity = 1.0,
                StartTime = 0.0,
                Duration = 1.0,
                Direction = SVector{3, Float64}(0.0, 0.0, 1.0),
            )
            path = joinpath(directory, "block.pvd")
            SavePolygonMotionSequence(path, (; block); motions = (; block = motion),
                                      times = [0.0, 1.0])

            @test polygon_motion_region_bounds(
                joinpath(directory, "block_frames", "frame_000002.vtkhdf"), 1, 3) ==
                  (1.0, 2.0)
        end
    end
end

@testset "MovingSquare polygon geometry" begin
    dx = 0.02
    polygons = moving_square_2d_polygons()

    @test polygons.tank isa PolyArea
    @test polygons.water isa PolyArea
    @test polygons.square isa PolyArea
    @test nvertices(polygons.tank) == 8
    @test nvertices(polygons.water) == 8
    @test nvertices(polygons.square) == 4
    @test isapprox(Meshes.ustrip(measure(polygons.tank)),
                   (10.0 + 2 * 0.04) * (5.0 + 2 * 0.04) - 10.0 * 5.0)
    @test isapprox(Meshes.ustrip(measure(polygons.water)), 10.0 * 5.0 - 1.0)
    @test isapprox(Meshes.ustrip(measure(polygons.square)), 1.0)
    @test Point(1.5, 2.5) in polygons.square
    @test !(Point(1.5, 2.5) in polygons.water)
    @test Point(3.0, 2.5) in polygons.water
    @test_throws ArgumentError moving_square_2d_polygons(
        tank_width = 2.0, square_lower_left = (1.5, 2.0))

    regions = [
        ParticleRegion("Fixed", polygons.tank, Fixed),
        ParticleRegion("Square", polygons.square, Moving),
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    particles = sample_particles(regions, dx)
    fixed, square, fluid = particles

    @test [region.name for region in particles] == ["Fixed", "Square", "Fluid"]
    @test moving_square_lattice_nodes(fixed.positions, dx) ==
          moving_square_reference_nodes("Fixed", dx)
    @test moving_square_lattice_nodes(square.positions, dx) ==
          moving_square_reference_nodes("Square", dx)
    @test moving_square_lattice_nodes(fluid.positions, dx) ==
          moving_square_reference_nodes("Fluid", dx)
    @test isempty(intersect(Set(fixed.positions), Set(square.positions)))
    @test isempty(intersect(Set(fixed.positions), Set(fluid.positions)))
    @test isempty(intersect(Set(square.positions), Set(fluid.positions)))
    @test length(fixed.positions) == 4524
    @test length(square.positions) == 2601
    @test length(fluid.positions) == 121650

    mktempdir() do directory
        generated = generate_moving_square_2d_example(directory; dx)
        prefix = joinpath(directory, "MovingSquare2D_Dp0.02")
        @test isfile(joinpath(directory, "MovingSquare2D_Geometry.vtkhdf"))
        @test isfile("$(prefix)_Particles.vtkhdf")
        @test all(name -> isfile("$(prefix)_$(name).csv"), ("Fixed", "Fluid", "Square"))
        @test all(region -> all(==(1000.0), region.density), generated)

        simulation_geometry = [
            SPHGeometry{2, Float64}(
                CSVFile = "$(prefix)_Fixed.csv",
                GroupMarker = 1,
                Type = Fixed,
                Motion = nothing
            ),
            SPHGeometry{2, Float64}(
                CSVFile = "$(prefix)_Fluid.csv",
                GroupMarker = 2,
                Type = Fluid,
                Motion = nothing
            ),
            SPHGeometry{2, Float64}(
                CSVFile = "$(prefix)_Square.csv",
                GroupMarker = 3,
                Type = Moving,
                Motion = MotionDetails{2, Float64}(
                    Velocity = 2.8,
                    StartTime = 0.0,
                    Duration = 3.0,
                    Direction = SVector{2, Float64}(1.0, 0.0)
                )
            ),
        ]
        loaded = AllocateDataStructures(simulation_geometry)
        @test length(loaded) == 4524 + 121650 + 2601
        @test loaded.ID == 1:length(loaded)
        @test moving_square_lattice_nodes(loaded.Position[loaded.Type .== Fixed], dx) ==
              moving_square_reference_nodes("Fixed", dx)
        @test moving_square_lattice_nodes(loaded.Position[loaded.Type .== Fluid], dx) ==
              moving_square_reference_nodes("Fluid", dx)
        @test moving_square_lattice_nodes(loaded.Position[loaded.Type .== Moving], dx) ==
              moving_square_reference_nodes("Square", dx)
        @test all(==(1000.0), loaded.Density)

        h5open(joinpath(directory, "MovingSquare2D_Geometry.vtkhdf"), "r") do file
            root = file["VTKHDF"]
            @test attrs(root)["Type"] == "PolyData"
            @test sort(unique(read(root["CellData/Region"]))) == [1, 2, 3]
        end

        h5open("$(prefix)_Particles.vtkhdf", "r") do file
            point_data = file["VTKHDF/PointData"]
            @test sort(unique(read(point_data["GroupMarker"]))) == [1, 2, 3]
            @test all(iszero, read(point_data["Pressure"]))
        end
    end
end
