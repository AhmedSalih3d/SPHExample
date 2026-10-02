module StillWedgeMiddleSquareGeometryTests

using Test
using SPHExampleGPU
using CSV
using HDF5

include(joinpath(@__DIR__, "..", "example",
                 "GenerateStillWedgeMiddleSquareMDBC.jl"))

middle_square_lattice_nodes(positions, dx) =
    Set(Tuple(round.(Int, collect(position) ./ dx)) for position in positions)

function middle_square_reference_rows(name, dx)
    path = joinpath(@__DIR__, "..", "input", "still_wedge_middle_square_mdbc",
                    "StillWedge_MiddleSquare_Dp$(dx)_$(name).csv")
    return CSV.File(path)
end

@testset "StillWedge middle-square polygon geometry" begin
    dx = 0.02
    constants = SimulationConstants{Float64}(; dx, c₀ = 42.48576250492629)
    polygons = still_wedge_middle_square_polygons()

    @test polygons.tank isa PolyArea
    @test polygons.water isa PolyArea
    @test polygons.square isa PolyArea
    @test nvertices(polygons.tank) == 14
    @test nvertices(polygons.water) == 11
    @test nvertices(polygons.square) == 4
    @test isapprox(Meshes.ustrip(measure(polygons.water)),
                   2.2 * 0.5 - 0.26^2 - 0.4 * (0.5 - 0.36))
    @test isapprox(Meshes.ustrip(measure(polygons.square)), 0.4 * 0.5)
    @test Point(1.1, 0.23) in polygons.tank
    @test !(Point(1.1, 0.15) in polygons.tank)
    @test !(Point(1.1, -0.02) in polygons.tank)
    @test Point(1.1, 0.32) in polygons.water
    @test !(Point(1.1, 0.46) in polygons.water)
    @test Point(1.1, 0.46) in polygons.square
    @test Point(1.1, 0.8) in polygons.square

    submerged = still_wedge_middle_square_polygons(square_height = 0.1)
    @test length(rings(submerged.water)) == 2
    @test isapprox(Meshes.ustrip(measure(submerged.water)),
                   2.2 * 0.5 - 0.26^2 - 0.4 * 0.1)
    @test Point(1.1, 0.48) in submerged.water
    @test !(Point(1.1, 0.4) in submerged.water)
    @test_throws ArgumentError still_wedge_middle_square_polygons(
        square_lower_left = (-0.1, 0.36))
    @test_throws ArgumentError still_wedge_middle_square_polygons(
        square_lower_left = (0.9, 0.24))
    @test_throws ArgumentError still_wedge_middle_square_polygons(square_width = NaN)
    @test_throws ArgumentError still_wedge_middle_square_polygons(water_height = 0.8)

    regions = [
        ParticleRegion("Bound", Multi([polygons.tank, polygons.square]), Fixed),
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    sampled = sample_particles(regions, dx)
    bound, fluid = sampled
    @test [region.name for region in sampled] == ["Bound", "Fluid"]
    @test [region.type for region in sampled] == [Fixed, Fluid]
    @test length(bound.positions) == 1126
    @test length(fluid.positions) == 2300
    @test length(Set(bound.positions)) == length(bound.positions)
    @test length(Set(fluid.positions)) == length(fluid.positions)
    @test isempty(intersect(Set(bound.positions), Set(fluid.positions)))
    @test count(p -> Point(p...) in polygons.square, bound.positions) == 546
    @test extrema(last.(fluid.positions)) == (0.02, 0.48)
    for region in sampled
        rows = middle_square_reference_rows(region.name, dx)
        reference_positions = [(row[Symbol("Points:0")], row[Symbol("Points:2")])
                               for row in rows]
        @test middle_square_lattice_nodes(region.positions, dx) ==
              middle_square_lattice_nodes(reference_positions, dx)
    end

    mktempdir() do directory
        @testset "generated files without input CSVs" begin
            particles = generate_still_wedge_middle_square_geometry(directory; dx)
            prefix = joinpath(directory, "StillWedge_MiddleSquare_Dp$(dx)")
            polygon_path = joinpath(directory, "StillWedge_MiddleSquare_Geometry.vtkhdf")
            @test isfile(polygon_path)
            @test isfile("$(prefix)_Particles.vtkhdf")
            @test isfile("$(prefix)_Bound.csv")
            @test isfile("$(prefix)_Fluid.csv")

            geometry = [
                SPHGeometry{2, Float64}(CSVFile = "$(prefix)_Bound.csv",
                                       GroupMarker = 1, Type = Fixed),
                SPHGeometry{2, Float64}(CSVFile = "$(prefix)_Fluid.csv",
                                       GroupMarker = 2, Type = Fluid),
            ]
            loaded = AllocateDataStructures(geometry)
            @test length(loaded) == 3426
            @test loaded.ID == 1:length(loaded)
            @test loaded.Position == vcat(bound.positions, fluid.positions)
            @test count(==(Fixed), loaded.Type) == 1126
            @test count(==(Fluid), loaded.Type) == 2300
            @test all(==(constants.ρ₀), particles[1].density)
            @test particles[2].density ==
                  hydrostatic_density(fluid.positions, constants)
            @test loaded.Density == vcat(particles[1].density, particles[2].density)

            reference_density = Dict(
                Tuple(round.(Int, (row[Symbol("Points:0")], row[Symbol("Points:2")]) ./ dx))
                => row.Rhop for row in middle_square_reference_rows("Fluid", dx))
            @test all(zip(particles[2].positions, particles[2].density)) do (p, density)
                round(density; digits = 1) ==
                    reference_density[Tuple(round.(Int, p ./ dx))]
            end

            h5open(polygon_path, "r") do file
                root = file["VTKHDF"]
                @test attrs(root)["Type"] == "PolyData"
                points = read(root["Points"])
                @test size(points) == (3, 29)
                @test read(root["NumberOfPoints"]) == [29]
                @test all(iszero, points[3, :])
                @test extrema(points[1, :]) == (-0.04, 2.24)
                @test extrema(points[2, :]) == (-0.04, 0.86)
                cells = root["Polygons"]
                connectivity = read(cells["Connectivity"])
                offsets = read(cells["Offsets"])
                region_ids = read(root["CellData/Region"])
                @test read(cells["NumberOfCells"]) == [23]
                @test offsets == collect(0:3:69)
                @test sort(unique(region_ids)) == [1, 2, 3]
                areas = zeros(3)
                for cell in eachindex(region_ids)
                    ids = connectivity[offsets[cell] + 1:offsets[cell + 1]] .+ 1
                    a, b, c = eachcol(points[1:2, ids])
                    area = ((b[1] - a[1]) * (c[2] - a[2]) -
                            (b[2] - a[2]) * (c[1] - a[1])) / 2
                    @test area > 0
                    areas[region_ids[cell]] += area
                end
                @test isapprox(areas,
                    Meshes.ustrip.(measure.([polygons.tank, polygons.water, polygons.square])))
            end

            h5open("$(prefix)_Particles.vtkhdf", "r") do file
                root = file["VTKHDF"]
                data = root["PointData"]
                @test read(root["NumberOfPoints"]) == [length(loaded)]
                @test read(root["Points"]) == stack(to_3d(loaded.Position))
                @test read(data["Density"]) == loaded.Density
                @test read(data["Type"]) == Int8.(loaded.Type)
                @test read(data["GroupMarker"]) == loaded.GroupMarker
                pressure = read(data["Pressure"])
                @test pressure ==
                      EquationOfStateGamma7.(loaded.Density, constants.c₀, constants.ρ₀)
                @test maximum(pressure) ≈ constants.ρ₀ * constants.g * 0.46 rtol = 1e-6
                @test minimum(pressure) ≈ 0 atol = 1e-6
            end
        end

        @testset "spacing and hydrostatic overrides" begin
            finer = generate_still_wedge_middle_square_geometry(
                joinpath(directory, "finer"); dx = 0.01)
            @test length(finer[1].positions) > 1126
            @test length(finer[2].positions) > 2300
            @test maximum(last, finer[2].positions) == 0.49
            @test isfile(joinpath(directory, "finer",
                                 "StillWedge_MiddleSquare_Dp0.01_Fluid.csv"))

            custom_constants = SimulationConstants{Float64}(; dx, ρ₀ = 1050, c₀ = 60, g = 4)
            custom = generate_still_wedge_middle_square_geometry(
                joinpath(directory, "custom"); dx, SimConstants = custom_constants,
                water_level = 0.5)
            @test all(==(1050.0), custom[1].density)
            @test custom[2].density == hydrostatic_density(
                custom[2].positions, custom_constants; water_level = 0.5)
            @test isapprox(
                EquationOfStateGamma7.(custom[2].density, 60, 1050),
                1050 * 4 .* (0.5 .- last.(custom[2].positions)); atol = 1e-6)

            for invalid_dx in (0.0, -0.02, NaN, Inf, 0.5)
                @test_throws ArgumentError generate_still_wedge_middle_square_geometry(
                    joinpath(directory, "invalid"); dx = invalid_dx)
            end
            @test !isdir(joinpath(directory, "invalid"))
        end
    end
end

end # module StillWedgeMiddleSquareGeometryTests
