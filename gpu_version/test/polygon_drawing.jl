using Test
using SPHExampleGPU
using HDF5
using LinearAlgebra: norm
using Meshes
using StaticArrays

# Defines the showcase functions only: its script part runs from the command line.
include(joinpath(@__DIR__, "..", "example", "GenerateShapesShowcase.jl"))

drawn_area(shape) = Meshes.ustrip(measure(shape))
drawn_coordinates(shape::PolyArea) = [Tuple(Meshes.ustrip.(to(p))) for p in vertices(shape)]
drawn_rings(shape::PolyArea) =
    [[Tuple(Meshes.ustrip.(to(p))) for p in vertices(r)] for r in rings(shape)]
drawn_ring_area(ring) = sum(ring[i][1] * ring[mod1(i + 1, end)][2] -
                      ring[mod1(i + 1, end)][1] * ring[i][2] for i in eachindex(ring)) / 2
same_drawn_vertices(a, b) = Set(drawn_coordinates(a)) == Set(drawn_coordinates(b))
drawn_inside(shape, x...) = Point(x...) in shape

drawing_lattice_nodes(positions, dx) = Set(Tuple(round.(Int, collect(p) ./ dx)) for p in positions)
function drawn_nodes(shape, dx; type = Fixed, kwargs...)
    region = only(sample_particles([ParticleRegion("Shape", shape, type; kwargs...)], dx))
    return drawing_lattice_nodes(region.positions, dx)
end

@testset "Polygon drawing" begin
    @testset "filled shapes" begin
        r = rectangle((1, 2), 3, 2)
        @test drawn_coordinates(r) == [(1.0, 2.0), (4.0, 2.0), (4.0, 4.0), (1.0, 4.0)]
        @test drawn_area(r) ≈ 6
        @test same_drawn_vertices(rectangle((2.5, 3), 3, 2; centered = true), r)
        @test same_drawn_vertices(rectangle((0, 0), 2, 1; angle = π / 2),
                            polygon([(0, 0), (0, 2), (-1, 2), (-1, 0)]))
        turned = rectangle((0, 0), 2, 1; angle = π / 6, centered = true)
        @test drawn_area(turned) ≈ 2
        @test drawn_inside(turned, 0.0, 0.0)
        @test same_drawn_vertices(square((0, 0), 1), rectangle((0, 0), 1, 1))

        t = triangle((0, 0), (0, 1), (1, 0))             # clockwise input
        @test drawn_area(t) ≈ 0.5
        @test drawn_ring_area(only(drawn_rings(t))) > 0   # stored counter clockwise

        hexagon = regular_polygon((1, 1), 2, 6)
        @test nvertices(hexagon) == 6
        @test drawn_area(hexagon) ≈ 3 * sqrt(3) / 2 * 4
        @test (3.0, 1.0) in drawn_coordinates(hexagon)

        c = circle((1, -1), 0.5)
        @test nvertices(c) == 128
        @test drawn_area(c) ≈ 64 * 0.25 * sin(2π / 128)
        @test all(p -> norm(p .- (1, -1)) ≈ 0.5, drawn_coordinates(c))
        @test all(p -> p in drawn_coordinates(c), [(1.5, -1.0), (1.0, -0.5), (0.5, -1.0), (1.0, -1.5)])
        @test nvertices(circle((0, 0), 1; segments = 16)) == 16

        a = arc((0, 0), 2, 0, π / 2; segments = 4)
        @test length(a) == 5
        @test first(a) == (2.0, 0.0) && last(a) == (0.0, 2.0)
        @test all(p -> norm(p) ≈ 2, a)
        @test length(arc((0, 0), 1, π, 0)) == 65                 # clockwise, default count
        @test drawn_area(polygon(arc((0, 0), 1, 0, π))) ≈ 64 * sin(π / 64) / 2

        @test_throws ArgumentError rectangle((0, 0), -1, 1)
        @test_throws ArgumentError circle((0, 0), 0)
        @test_throws ArgumentError regular_polygon((0, 0), 1, 2)
        @test_throws ArgumentError arc((0, 0), 1, 0, 0)
        @test_throws ArgumentError triangle((0, 0), (1, 1), (2, 2))         # collinear
        @test_throws ArgumentError polygon([(0, 0), (1, 0)])
        @test_throws ArgumentError polygon([(0, 0), (1, 0), (1, NaN)])
        @test_throws ArgumentError polygon([(0, 0), (2, 2), (2, 0), (0, 1)])  # bow tie
    end

    @testset "polygons with holes" begin
        outer = [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)]        # closing vertex repeated
        p = polygon(outer; holes = [square((1, 1), 1), [(3, 3), (3.5, 3), (3.5, 3.5)]])
        @test length(rings(p)) == 3
        @test drawn_area(p) ≈ 16 - 1 - 0.125
        ring_list = drawn_rings(p)
        @test drawn_ring_area(ring_list[1]) > 0 && all(r -> drawn_ring_area(r) < 0, ring_list[2:end])
        @test !drawn_inside(p, 1.5, 1.5) && drawn_inside(p, 2.5, 1.5)

        # Holes of a PolyArea are kept when more are added.
        q = polygon(p; holes = [circle((3, 1), 0.3)])
        @test length(rings(q)) == 4

        @test_throws ArgumentError polygon(outer; holes = [square((5, 5), 1)])
        @test_throws ArgumentError polygon(outer; holes = [square((3.5, 1), 1)])  # crosses
        @test_throws ArgumentError polygon(outer; holes = [square((1, 1), 2), square((1.5, 1.5), 0.5)])
    end

    @testset "lines and polylines" begin
        @test drawn_area(line((0, 0), (2, 0); thickness = 0.1)) ≈ 0.2
        @test same_drawn_vertices(line((0, 0), (2, 0); thickness = 0.1),
                            rectangle((0, -0.05), 2, 0.1))
        @test same_drawn_vertices(line((0, 0), (2, 0); thickness = 0.1, side = :left),
                            rectangle((0, 0), 2, 0.1))
        @test same_drawn_vertices(line((0, 0), (2, 0); thickness = 0.1, side = :right),
                            rectangle((0, -0.1), 2, 0.1))
        @test same_drawn_vertices(line((0, 0), (2, 0); thickness = 0.1, side = :left, offset = 0.05),
                            rectangle((0, 0.05), 2, 0.1))
        @test same_drawn_vertices(line((0, 0), (2, 0); thickness = 0.1, side = :left, offset = -0.2),
                            rectangle((0, -0.2), 2, 0.1))
        slanted = line((0, 0), (3, 4); thickness = 0.2)
        @test drawn_area(slanted) ≈ 5 * 0.2
        @test drawn_inside(slanted, 1.5, 2.0)

        # The open dam-break tank: the path is the wetted surface, walls grow outwards.
        t = 0.06
        tank = polyline([(0, 3), (0, 0), (4, 0), (4, 3)]; thickness = t, side = :right)
        handmade = PolyArea([(-t, -t), (4 + t, -t), (4 + t, 3.0), (4.0, 3.0), (4.0, 0.0),
                             (0.0, 0.0), (0.0, 3.0), (-t, 3.0)])
        @test same_drawn_vertices(tank, handmade)
        @test drawn_area(tank) ≈ drawn_area(handmade)
        @test drawn_nodes(tank, 0.02) == drawn_nodes(handmade, 0.02)

        # The same path drawn with walls inwards keeps the outer faces on the path.
        inward = polyline([(0, 3), (0, 0), (4, 0), (4, 3)]; thickness = t, side = :left)
        @test Set(drawn_coordinates(inward)) ⊇ Set([(0.0, 3.0), (0.0, 0.0), (4.0, 0.0), (4.0, 3.0)])
        @test drawn_area(inward) ≈ 2 * 3 * t + (4 - 2t) * t

        ring = polyline([(0, 0), (1, 0), (1, 1), (0, 1)]; thickness = 0.1, closed = true)
        @test length(rings(ring)) == 2
        @test drawn_area(ring) ≈ 1.1^2 - 0.9^2

        # Sharp joints are bevelled on their convex side only.
        spike = [(0, 0), (1, 0.1), (0, 0.2)]
        @test nvertices(polyline(spike; thickness = 0.01)) == 7
        @test nvertices(polyline(spike; thickness = 0.01, miter_limit = 100)) == 6

        @test_throws ArgumentError line((0, 0), (1, 0); thickness = 0)
        @test_throws ArgumentError line((0, 0), (1, 0); thickness = 0.1, side = :outward)
        @test_throws ArgumentError line((0, 0), (0, 0); thickness = 0.1)
        @test_throws ArgumentError polyline([(0, 0), (1, 0), (0, 0)]; thickness = 0.1)
        # The inner offset overshoots the short leg.
        @test_throws ArgumentError polyline([(0, 0), (1, 0), (1, 0.1)]; thickness = 0.5,
                                            side = :left)
        @test_throws ArgumentError polyline([(0, 0), (1, 0)]; thickness = 0.1, miter_limit = 0.5)
    end

    @testset "outlines and offsets" begin
        interior = rectangle((0, 0), 10, 5)
        tank = outline(interior; thickness = 0.04)
        handmade = PolyArea([[(-0.04, -0.04), (10.04, -0.04), (10.04, 5.04), (-0.04, 5.04)],
                             [(0.0, 0.0), (0.0, 5.0), (10.0, 5.0), (10.0, 0.0)]])
        @test Set(map(Set, drawn_rings(tank))) == Set(map(Set, drawn_rings(handmade)))
        @test drawn_area(tank) ≈ 10.08 * 5.08 - 50
        @test drawn_area(outline(interior; thickness = 0.04, side = :inward)) ≈ 50 - 9.92 * 4.92
        @test drawn_area(outline(interior; thickness = 0.04, side = :center)) ≈ 10.04 * 5.04 - 9.96 * 4.96
        gapped = outline(interior; thickness = 0.04, offset = 0.01)
        @test drawn_area(gapped) ≈ 10.1 * 5.1 - 10.02 * 5.02
        @test !drawn_inside(gapped, -0.005, 2.0) && drawn_inside(gapped, -0.03, 2.0)
        @test outline(drawn_coordinates(interior); thickness = 0.04) isa PolyArea

        # A pipe: an inward wall keeps the circle's outer face.
        pipe = outline(circle((0, 0), 1); thickness = 0.1, side = :inward)
        # The inward miter offset of a regular polygon keeps its corners on rays.
        @test drawn_area(pipe) ≈ drawn_area(circle((0, 0), 1)) - drawn_area(circle((0, 0), 1 - 0.1 / cos(π / 128)))
        @test drawn_inside(pipe, 0.95, 0.0) && !drawn_inside(pipe, 0.5, 0.0)

        # Every ring gets a wall; outward at a hole points into the hole.
        holed = polygon(interior; holes = [square((4, 2), 1)])
        walls = outline(holed; thickness = 0.1)
        @test walls isa Multi
        @test length(parent(walls)) == 2
        @test drawn_area(walls) ≈ (10.2 * 5.2 - 50) + (1 - 0.8^2)
        @test drawn_inside(walls, 4.05, 2.5) && !drawn_inside(walls, 3.95, 2.5)
        @test length(parent(outline(Multi([interior, square((20, 0), 1)]); thickness = 0.1))) == 2

        @test drawn_area(offset_polygon(interior, 0.5)) ≈ 11 * 6
        @test drawn_area(offset_polygon(interior, -0.5)) ≈ 9 * 4
        @test drawn_area(offset_polygon(holed, 0.1)) ≈ 10.2 * 5.2 - 0.8^2
        @test drawn_area(offset_polygon(triangle((0, 0), (1, 0), (0, 1)), 0.0)) ≈ 0.5

        @test_throws ArgumentError outline(circle((0, 0), 1); thickness = 1.5, side = :inward)
        @test_throws ArgumentError outline(interior; thickness = 3, side = :inward)
        @test_throws ArgumentError offset_polygon(interior, -2.5)
        @test_throws ArgumentError offset_polygon(holed, 2.5)        # the hole collapses
        @test_throws ArgumentError offset_polygon(holed, -1.5)       # the hole crosses the outline
        @test_throws ArgumentError outline(interior; thickness = 0.1, side = :left)
    end

    @testset "transformations" begin
        r = rectangle((0, 0), 2, 1)
        @test drawn_coordinates(translate(r, (1, -1))) == [(1.0, -1.0), (3.0, -1.0), (3.0, 0.0), (1.0, 0.0)]
        @test same_drawn_vertices(rotate(r, π / 2), polygon([(0, 0), (0, 2), (-1, 2), (-1, 0)]))
        @test same_drawn_vertices(rotate(r, π; origin = (1, 0.5)), r)
        mirrored = mirror(triangle((0, 0), (1, 0), (0, 1)))
        @test Set(drawn_coordinates(mirrored)) == Set([(0.0, 0.0), (-1.0, 0.0), (0.0, 1.0)])
        @test drawn_ring_area(only(drawn_rings(mirrored))) > 0
        @test same_drawn_vertices(mirror(r; origin = (0, 2), direction = (1, 0)),
                            rectangle((0, 3), 2, 1))
        @test translate([(0, 0), (1, 0)], (1, 1)) == [(1.0, 1.0), (2.0, 1.0)]
        moved = translate(Multi([r, square((5, 5), 1)]), (1, 0))
        @test drawn_area(moved) ≈ 3
        @test_throws ArgumentError mirror(r; direction = (0, 0))
    end

    @testset "sampling drawn shapes" begin
        dx = 0.1
        # With 256 segments no lattice node lies between the polygon and the circle.
        disc = circle((0, 0), 0.95; segments = 256)
        expected = Set((i, j) for i in -10:10, j in -10:10 if hypot(i * dx, j * dx) < 0.95)
        @test drawn_nodes(disc, dx) == expected

        # A union of geometries is filled as one region.
        parts = (rectangle((0, 0), 1, 0.2), rectangle((0.4, 0), 0.2, 1))
        @test drawn_nodes(parts, dx) ==
              union(drawn_nodes(parts[1], dx), drawn_nodes(parts[2], dx))

        @test_throws ArgumentError sample_particles([ParticleRegion("Box", Box((0, 0), (1, 1)), Fixed)], dx)
    end

    @testset "conforming sampling" begin
        dx = 0.02
        min_spacing(ps) = minimum(norm(ps[i] - ps[j]) for i in eachindex(ps)
                                  for j in (i + 1):lastindex(ps))
        min_between(a, b) = minimum(norm(x - y) for x in a for y in b)
        conforming(shape; kwargs...) = only(sample_particles(
            [ParticleRegion("Shape", shape, Fixed; sampling = :conforming, kwargs...)], dx)).positions
        # Whether the particles lie on, and fill, circles of the `expected` radii (in dx);
        # the tolerance covers the sag of the 128 gon (< 0.005 dx here).
        on_radii(ps, centre, expected) =
            all(x -> any(r -> abs(norm(x - SVector(centre)) / dx - r) < 0.01, expected), ps) &&
            all(r -> any(x -> abs(norm(x - SVector(centre)) / dx - r) < 0.01, ps), expected)

        # A disc fills with concentric rings one spacing apart, down to its centre.
        disc = conforming(circle((0.5, 0.5), 7.5dx))
        @test on_radii(disc, (0.5, 0.5), 0.5:1.0:7.5)
        @test min_spacing(disc) > 0.85dx
        @test length(disc) == sum(round(Int, 2π * (k + 0.5)) for k in 0:7)

        # Straight edges keep their corners; the layers shrink one spacing at a time.
        tilted = square((0, 0), 5dx; angle = 0.3)
        tilted_particles = conforming(tilted)
        @test length(tilted_particles) == 20 + 12 + 4
        corners = [SVector(Meshes.ustrip.(to(p))...) for p in vertices(tilted)]
        @test all(c -> any(x -> norm(x - c) < 1e-9, tilted_particles), corners)
        @test min_spacing(tilted_particles) ≈ dx

        # A centred arc wall two spacings thick gets three evenly spaced layers.
        r = 15dx
        baffle = polyline(arc((1, 0.5), r, π, 2π); thickness = 2dx)
        baffle_particles = conforming(baffle)
        @test on_radii(baffle_particles, (1, 0.5), [14, 15, 16])
        @test min_spacing(baffle_particles) > 0.98dx
        # The ends are square to the first and last arc segments, a little off horizontal.
        @test all(x -> x[2] <= 0.5 + dx * sin(π / 64), baffle_particles)
        # Its ends collapse locally and still keep the spacing.
        thick = conforming(polyline(arc((1, 0.5), r, π, 2π); thickness = 3dx, side = :right))
        @test min_spacing(thick) > 0.95dx

        # A limited number of layers leaves the core to the lattice regions that follow.
        water = rectangle((0, 0), 1, 1)
        shell, fluid = sample_particles(
            [ParticleRegion("Shell", circle((0.5, 0.5), 0.2), Fixed; sampling = :conforming,
                            layers = 3),
             ParticleRegion("Fluid", water, Fluid)], dx)
        @test on_radii(shell.positions, (0.5, 0.5), [8, 9, 10])
        @test min_between(shell.positions, fluid.positions) >= dx / 2
        @test count(x -> norm(x - SVector(0.5, 0.5)) < 7dx, fluid.positions) > 0
        @test all(x -> !(7.5dx < norm(x - SVector(0.5, 0.5)) < 10.5dx), fluid.positions)
        @test all(x -> all(abs.(x ./ dx .- round.(x ./ dx)) .< 1e-6), fluid.positions)

        # Without the surface the first layer lies one spacing inside.
        inner = conforming(circle((0.5, 0.5), 7.5dx); include_surface = false, layers = 1)
        @test on_radii(inner, (0.5, 0.5), [6.5])
        @test on_radii(conforming(circle((0.5, 0.5), 7.5dx); offset = dx, layers = 1),
                       (0.5, 0.5), [6.5])

        # Particles of a conforming region give way to an earlier lattice wall.
        floor_ = rectangle((0, -3dx), 1, 3dx)
        wall, cylinder = sample_particles(
            [ParticleRegion("Floor", floor_, Fixed),
             ParticleRegion("Cylinder", circle((0.5, 6.5dx), 6.5dx), Fixed;
                            sampling = :conforming)], dx)
        @test min_between(wall.positions, cylinder.positions) >= dx / 2
        @test length(wall.positions) == length(drawn_nodes(floor_, dx))

        @test_throws ArgumentError ParticleRegion("Shape", water, Fixed; sampling = :grid)
        @test_throws ArgumentError ParticleRegion("Shape", water, Fixed; layers = 2)
        @test_throws ArgumentError ParticleRegion("Shape", water, Fixed;
                                                  sampling = :conforming, layers = 0)
        @test_throws ArgumentError sample_particles(
            [ParticleRegion("Block", prism(water, 0, 1), Fixed; sampling = :conforming)], dx)
    end

    @testset "prisms" begin
        dx = 0.1
        block = prism(rectangle((0, 0), 1, 0.5), 0, 0.3)
        @test block isa ExtrudedPolygon
        @test length(only(sample_particles([ParticleRegion("Block", block, Fixed)], dx)).positions) ==
              11 * 6 * 4
        fluid_block = only(sample_particles([ParticleRegion("Block", block, Fluid)], dx))
        @test length(fluid_block.positions) == 9 * 4 * 2
        @test all(x -> length(x) == 3, fluid_block.positions)
        # The offset applies along z as well as across.
        shrunk = only(sample_particles([ParticleRegion("Block", block, Fixed; offset = 0.1)], dx))
        @test length(shrunk.positions) == 9 * 4 * 2

        cylinder = prism(circle((0, 0), 0.95; segments = 256), -0.2, 0.2)
        cylinder_nodes = drawing_lattice_nodes(
            only(sample_particles([ParticleRegion("Cylinder", cylinder, Fixed)], dx)).positions, dx)
        @test length(cylinder_nodes) == 5 * length(drawn_nodes(circle((0, 0), 0.95; segments = 256), dx))

        moved = translate(rotate(block, π / 2), (1, 0, 1))
        @test (moved.bottom, moved.top) == (1.0, 1.3)
        @test Set(drawn_coordinates(moved.base)) == Set([(1.0, 0.0), (1.0, 1.0), (0.5, 1.0), (0.5, 0.0)])

        # A closed 3D tank from wall and floor prisms with water inside.
        t = 0.1
        interior = rectangle((0, 0), 1, 0.5)
        tank = (prism(outline(interior; thickness = t), -t, 0.6),
                prism(offset_polygon(interior, t), -t, 0.0))
        water = prism(interior, 0.0, 0.3)
        bound, fluid = sample_particles([ParticleRegion("Bound", tank, Fixed),
                                         ParticleRegion("Fluid", water, Fluid)], dx)
        @test length(fluid.positions) == 9 * 4 * 2
        @test length(bound.positions) == 13 * 8 * 8 - 9 * 4 * 6
        @test isempty(intersect(Set(bound.positions), Set(fluid.positions)))

        @test_throws ArgumentError prism(interior, 1, 0)
        @test_throws ArgumentError sample_particles([ParticleRegion("Mixed", (interior, water), Fixed)], dx)
        @test_throws ArgumentError sample_particles([ParticleRegion("Bound", tank, Fixed),
                                                     ParticleRegion("Flat", interior, Fluid)], dx)
    end

    @testset "shapes showcase example" begin
        mktempdir() do directory
            dx = 0.04
            shapes_2d = showcase_2d_shapes(; dx)
            @test length(rings(shapes_2d.water)) == 6
            @test !(Point(0.5, 0.3) in shapes_2d.water)
            @test !(Point(1.2, 0.1) in shapes_2d.water)
            @test !(Point(2.6, 0.15) in shapes_2d.water)
            @test !(Point(1.85, 0.45) in shapes_2d.water)
            @test !(Point(2.55, 0.45) in shapes_2d.water)
            @test Point(2.8, 0.55) in shapes_2d.water

            cases = generate_shapes_showcase(directory; dx)
            for (scene, particles) in pairs(cases)
                @test all(region -> !isempty(region.positions), particles)
                sets = [Set(region.positions) for region in particles]
                @test all(isempty(intersect(sets[i], sets[j]))
                          for i in eachindex(sets) for j in (i + 1):lastindex(sets))
            end
            @test all(x -> length(x) == 3, first(cases.three_d).positions)
            # Lattice and conforming regions keep at least half a spacing apart.
            two_d = [region.positions for region in cases.two_d]
            @test all(minimum(norm(x - y) for x in two_d[i] for y in two_d[j]) >= dx / 2
                      for i in eachindex(two_d) for j in (i + 1):lastindex(two_d))
            for case in ("Showcase2D", "Showcase3D")
                @test isfile(joinpath(directory, "$(case)_Geometry.vtkhdf"))
                @test isfile(joinpath(directory, "$(case)_Dp$(dx)_Particles.vtkhdf"))
                @test isfile(joinpath(directory, "$(case)_Dp$(dx)_Fluid.csv"))
            end
            h5open(joinpath(directory, "Showcase2D_Geometry.vtkhdf"), "r") do file
                root = file["VTKHDF"]
                regions = read(root["CellData/Region"])
                offsets = read(root["Polygons/Offsets"])
                cell_sizes = diff(offsets)
                @test all(==(3), cell_sizes[regions .== 4])
                @test any(>(3), cell_sizes[regions .!= 4])
            end
        end
    end

    @testset "polygon VTKHDF with prisms" begin
        mktempdir() do directory
            path = joinpath(directory, "shapes.vtkhdf")
            block = prism(rectangle((0, 0), 1, 0.5), 0, 0.3)
            holed = polygon(rectangle((5, 0), 2, 1);
                            holes = [circle((5.5, 0.5), 0.2; segments = 8)])
            holed_prism = prism(holed, 0, 0.5)
            SavePolygonVTKHDF(path, (; flat = circle((3, 0), 0.5; segments = 16),
                                       block, union = (block, prism(square((2, 2), 1), 0, 1)),
                                       holed, holed_prism))
            h5open(path, "r") do file
                root = file["VTKHDF"]
                regions = read(root["CellData/Region"])
                offsets = read(root["Polygons/Offsets"])
                cell_sizes = diff(offsets)
                @test count(==(1), regions) == 1
                @test only(cell_sizes[regions .== 1]) == 16
                @test count(==(2), regions) == 6
                @test all(==(4), cell_sizes[regions .== 2])
                @test count(==(3), regions) == 12
                @test all(==(4), cell_sizes[regions .== 3])

                holed_mesh = discretize(holed)
                ntriangles = length(elements(topology(holed_mesh)))
                sides = sum(length(vertices(ring)) for ring in rings(holed))
                @test count(==(4), regions) == ntriangles
                @test all(==(3), cell_sizes[regions .== 4])
                prism_sizes = cell_sizes[regions .== 5]
                @test count(==(3), prism_sizes) == 2 * ntriangles
                @test count(==(4), prism_sizes) == sides

                points = read(root["Points"])
                connectivity = read(root["Polygons/Connectivity"])
                holed_area = 0.0
                for cell in eachindex(regions)
                    regions[cell] == 4 || continue
                    ids = connectivity[offsets[cell] + 1:offsets[cell + 1]] .+ 1
                    signed_area = sum(
                        points[1, ids[i]] * points[2, ids[mod1(i + 1, length(ids))]] -
                        points[1, ids[mod1(i + 1, length(ids))]] * points[2, ids[i]]
                        for i in eachindex(ids)
                    ) / 2
                    @test signed_area > 0
                    holed_area += signed_area
                end
                @test isapprox(holed_area, Meshes.ustrip(measure(holed)))

                @test extrema(points[3, :]) == (0.0, 1.0)
            end
        end
    end
end
