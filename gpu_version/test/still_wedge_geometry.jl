using Test
using CSV
using HDF5
using StaticArrays

include(joinpath(@__DIR__, "..", "example", "GenerateStillWedgeMDBC.jl"))

# Lattice indices of 2D positions, for exact set comparisons.
lattice_nodes(positions, dx) = Set(Tuple(round.(Int, collect(p) ./ dx)) for p in positions)

function reference_nodes(suffix, dx)
    file = joinpath(@__DIR__, "..", "input", "still_wedge", "StillWedge_Dp$(dx)_$suffix.csv")
    rows = CSV.File(file)
    lattice_nodes([(row[Symbol("Points:0")], row[Symbol("Points:2")]) for row in rows], dx)
end

function reference_densities(suffix, dx)
    file = joinpath(@__DIR__, "..", "input", "still_wedge", "StillWedge_Dp$(dx)_$suffix.csv")
    Dict(Tuple(round.(Int, (row[Symbol("Points:0")], row[Symbol("Points:2")]) ./ dx)) => row.Rhop
         for row in CSV.File(file))
end

@testset "hydrostatic density" begin
    constants = SimulationConstants{Float64}(dx = 0.02, c₀ = 42.48576250492629)
    (; ρ₀, g, c₀) = constants
    positions = [SVector(0.3, y) for y in 0.02:0.02:0.48]
    ρ = hydrostatic_density(positions, constants)
    @test ρ[end] == ρ₀                        # default level: highest particle
    @test issorted(ρ; rev = true)             # denser with depth
    # inverse of the solver's equation of state
    @test isapprox(EquationOfStateGamma7.(ρ, c₀, ρ₀), ρ₀ * g .* (0.48 .- last.(positions)); atol = 1e-4)
    ρ₅ = hydrostatic_density(positions, constants; water_level = 0.5)
    @test isapprox(EquationOfStateGamma7(ρ₅[end], c₀, ρ₀), ρ₀ * g * 0.02; atol = 1e-4)
    @test hydrostatic_density([SVector(0.0, 0.6)], constants; water_level = 0.5) == [ρ₀]
    # 3D: the vertical coordinate is z
    @test hydrostatic_density([SVector(0.0, 5.0, 0.1), SVector(0.0, -5.0, 0.3)], constants)[1] > ρ₀
end

@testset "StillWedge polygon geometry" begin
    dx = 0.02
    polygons = still_wedge_2d_polygons()
    @test polygons.tank isa PolyArea
    @test polygons.water isa PolyArea
    @test nvertices(polygons.tank) == 14
    @test nvertices(polygons.water) == 7
    @test isapprox(Meshes.ustrip(measure(polygons.water)), 2.2 * 0.5 - 0.26^2)
    # tank: floor and walls, minus the V notch, plus the outer wedge triangle
    @test isapprox(Meshes.ustrip(measure(polygons.tank)),
                   (2.28 * 0.74 - 2.2 * 0.7) - 0.5 * 0.48 * 0.24 + 0.5 * 0.52 * 0.26)
    @test Point(1.1, 0.23) in polygons.tank      # wedge shell
    @test !(Point(1.1, 0.15) in polygons.tank)   # hollow inside the wedge
    @test !(Point(1.1, -0.02) in polygons.tank)  # no floor below the wedge
    @test !(Point(1.1, 0.15) in polygons.water)
    @test Point(1.1, 0.4) in polygons.water
    @test !(Point(1.1, 0.6) in polygons.water)

    @testset "particles reproduce the reference input" begin
        regions = [ParticleRegion("Bound", polygons.tank, Fixed),
                   ParticleRegion("Fluid", polygons.water, Fluid)]
        particles = sample_particles(regions, dx)
        @test [p.name for p in particles] == ["Bound", "Fluid"]
        @test [p.type for p in particles] == [Fixed, Fluid]
        bound, fluid = particles
        @test lattice_nodes(bound.positions, dx) == reference_nodes("Bound", dx)
        @test lattice_nodes(fluid.positions, dx) == reference_nodes("Fluid", dx)
        @test isempty(intersect(Set(bound.positions), Set(fluid.positions)))
        # coordinates are lattice nodes without floating point noise
        @test all(p -> all(x -> x == round(x; digits = 6), p), bound.positions)
        @test maximum(p -> p[2], fluid.positions) == 0.48
        @test minimum(p -> p[1], fluid.positions) == 0.02
    end

    @testset "offset shrinks a region" begin
        wall  = ParticleRegion("Bound", polygons.tank, Fixed)
        water = ParticleRegion("Fluid", polygons.water, Fluid)
        reference = sample_particles([wall, water], dx)[2].positions
        # the wedge faces are at 45°, so the nearest lattice points are dx/√2 away:
        # a 0.75dx shrink clears them but keeps the lattice-aligned layers, which
        # are already one spacing from the walls
        shrunk = ParticleRegion("Fluid", polygons.water, Fluid; offset = 0.75dx)
        fluid = sample_particles([wall, shrunk], dx)[2].positions
        @test fluid ⊆ reference
        @test length(fluid) < length(reference)
        @test minimum(p -> p[1], fluid) == 0.02
        gap = minimum(fluid) do p
            minimum(Meshes.ustrip(Meshes.evaluate(Meshes.Euclidean(), Point(p...), s))
                    for ring in rings(polygons.water) for s in segments(ring))
        end
        @test gap >= 0.75dx - 1e-9
        # a full spacing removes the next layer; negative offsets grow the region
        @test minimum(p -> p[1], sample_particles([wall,
            ParticleRegion("Fluid", polygons.water, Fluid; offset = dx)], dx)[2].positions) == 0.04
        grown = sample_particles([ParticleRegion("Fluid", polygons.water, Fluid;
                                                 offset = -dx, include_surface = true)], dx)
        @test minimum(p -> p[1], grown[1].positions) == -dx
        # zero offset is the default and unchanged
        @test ParticleRegion("Bound", polygons.tank, Fixed).offset == 0
    end

    mktempdir() do directory
        @testset "generated files" begin
            particles = generate_still_wedge_2d_example(directory; dx)
            @test isfile(joinpath(directory, "StillWedge2D_Geometry.vtkhdf"))
            @test isfile(joinpath(directory, "StillWedge2D_Dp0.02_Particles.vtkhdf"))

            # the CSV files load through the same path as the reference input
            geometry = [
                SPHGeometry{2, Float64}(CSVFile = joinpath(directory, "StillWedge2D_Dp0.02_Bound.csv"),
                                        GroupMarker = 1, Type = Fixed),
                SPHGeometry{2, Float64}(CSVFile = joinpath(directory, "StillWedge2D_Dp0.02_Fluid.csv"),
                                        GroupMarker = 2, Type = Fluid),
            ]
            loaded = AllocateDataStructures(geometry)
            @test length(loaded) == sum(length(p.positions) for p in particles)
            @test loaded.ID == 1:length(loaded)
            @test count(==(Fixed), loaded.Type) == length(particles[1].positions)
            @test lattice_nodes(loaded.Position[loaded.Type .== Fluid], dx) == reference_nodes("Fluid", dx)

            # boundaries at rest density, fluid hydrostatic like the reference (1 decimal)
            @test all(==(1000.0), loaded.Density[loaded.Type .== Fixed])
            reference = reference_densities("Fluid", dx)
            fluid = findall(==(Fluid), loaded.Type)
            @test all(fluid) do i
                round(loaded.Density[i]; digits = 1) ==
                    reference[Tuple(round.(Int, loaded.Position[i] ./ dx))]
            end
            @test particles[2].density == hydrostatic_density(particles[2].positions,
                SimulationConstants{Float64}(dx = dx, c₀ = 42.48576250492629))

            h5open(joinpath(directory, "StillWedge2D_Dp0.02_Particles.vtkhdf"), "r") do file
                root = file["VTKHDF"]
                @test attrs(root)["Type"] == "PolyData"
                @test read(root["NumberOfPoints"]) == [length(loaded)]
                @test read(root["Vertices"]["NumberOfCells"]) == [length(loaded)]
                @test sort(unique(read(root["PointData"]["GroupMarker"]))) == [1, 2]
                @test sort(unique(read(root["PointData"]["Type"]))) == Int8[Int8(Fluid), Int8(Fixed)]
                pressure = read(root["PointData"]["Pressure"])
                @test maximum(pressure) ≈ 1000 * 9.81 * 0.46 rtol = 1e-6
                @test minimum(pressure) ≈ 0 atol = 1e-6
            end
        end

        @testset "polygon VTKHDF" begin
            path = joinpath(directory, "geometry", "StillWedge.vtkhdf")
            @test SavePolygonVTKHDF(path, polygons) == path
            h5open(path, "r") do file
                root = file["VTKHDF"]
                @test attrs(root)["Type"] == "PolyData"
                @test attrs(root)["Version"] == Int32[2, 3]
                points = read(root["Points"])
                @test size(points) == (3, 14 + 7)
                @test read(root["NumberOfPoints"]) == [21]
                @test all(iszero, points[3, :])
                @test extrema(points[1, :]) == (-0.04, 2.24)
                @test extrema(points[2, :]) == (-0.04, 0.7)

                cells        = root["Polygons"]
                connectivity = read(cells["Connectivity"])
                offsets      = read(cells["Offsets"])
                regions      = read(root["CellData/Region"])
                ntriangles   = (14 - 2) + (7 - 2)
                @test read(cells["NumberOfCells"]) == [ntriangles]
                @test read(cells["NumberOfConnectivityIds"]) == [3 * ntriangles]
                @test offsets == collect(0:3:(3 * ntriangles))
                @test all(id -> 0 <= id < 21, connectivity)
                @test count(==(1), regions) == 12
                @test count(==(2), regions) == 5

                areas = zeros(2)
                for cell in eachindex(regions)
                    ids = connectivity[offsets[cell] + 1:offsets[cell + 1]] .+ 1
                    a, b, c = eachcol(points[1:2, ids])
                    signed_area = ((b[1] - a[1]) * (c[2] - a[2]) -
                                   (b[2] - a[2]) * (c[1] - a[1])) / 2
                    @test signed_area > 0
                    areas[regions[cell]] += signed_area
                end
                @test isapprox(areas, Meshes.ustrip.(measure.([polygons.tank, polygons.water])))

                for cell_type in ("Vertices", "Lines", "Strips")
                    group = root[cell_type]
                    @test read(group["NumberOfCells"]) == [0]
                    @test read(group["NumberOfConnectivityIds"]) == [0]
                    @test isempty(read(group["Connectivity"]))
                    @test read(group["Offsets"]) == [0]
                end
            end
            @test_throws ArgumentError SavePolygonVTKHDF(path, (; bad = Point(0.0, 0.0)))
        end
    end
end
