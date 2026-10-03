using Test
using SPHExampleGPU
using StaticArrays
using StructArrays
using Meshes

@testset "SPHGeometry holds input particles" begin
    mktempdir() do dir
        for d in (2, 3)
            positions = [SVector{d, Float64}(ntuple(k -> k + 0.123456789, d)),
                         SVector{d, Float64}(ntuple(k -> k + 0.234567891, d))]
            density = [1000.0, 1001.0]
            path = joinpath(dir, "particles$(d).csv")
            write_particle_csv(path, positions; density, first_id = 10)
            loaded = SPHGeometry{d, Float32}(CSVFile = path,
                GroupMarker = 1, Type = Fluid)
            direct = SPHGeometry{d, Float32}(Particles = StructArray((
                Position = positions, Density = density, ID = [11, 12])),
                GroupMarker = 1, Type = Fluid)
            @test loaded.Particles isa StructArray
            rm(path) # Allocation uses stored data, not the source file.
            meta = SimulationMetaData{d, Float32}(SimulationName = "geometry",
                SaveLocation = dir, GPUDoublePosition = true)
            a = AllocateDataStructures([loaded], meta)
            b = AllocateDataStructures([direct], meta)
            @test a == b
            @test a.Position == positions
            @test eltype(a.Position) == SVector{d, Float64}
            a.Position[1] = zero(eltype(a.Position))
            @test loaded.Particles.Position == positions
            @test AllocateDataStructures([loaded], meta) == b
            @test eltype(AllocateDataStructures([direct]).Position) ==
                SVector{d, Float32}
        end

        regions = [ParticleRegion("wall", PolyArea([(0., 0.), (1., 0.),
                    (1., 1.), (0., 1.)]), Fixed)]
        region = only(sample_particles(regions, 0.5))
        geometry = SPHGeometry{2, Float32}(region.positions;
            Density = 1000, GroupMarker = 1, Type = region.type)
        fluid = SPHGeometry{2, Float32}([SVector(2., 2.)];
            Density = [1001], GroupMarker = 2, Type = Fluid)
        particles = AllocateDataStructures([geometry, fluid])
        @test particles.ID == collect(1:length(particles))
        @test count(==(Fixed), particles.Type) == length(region.positions)
        @test particles.GroupMarker[end] == 2

        source = StructArray((Position = [SVector(1., 2.)], Density = [1000.],
            ID = [20], Velocity = [SVector(3., 4.)],
            GhostPoints = [SVector(1., 3.)], GhostNormals = [SVector(0., 1.)]))
        prescribed = SPHGeometry{2, Float32}(Particles = source,
            GroupMarker = 3, Type = Moving,
            Motion = MotionDetails{2, Float32}(; Velocity = 1, StartTime = 0,
                Duration = 1, Direction = SVector(1, 0)))
        mixed = AllocateDataStructures([geometry, prescribed])
        @test length(unique(mixed.ID)) == length(mixed)
        @test mixed.ID[end] > 20 # Automatically assigned IDs follow explicit IDs.
        k = findfirst(==(20), mixed.ID)
        @test mixed.Velocity[k] == SVector(3f0, 4f0)
        @test mixed.GhostPoints[k] == SVector(1f0, 3f0)
        @test mixed.GhostNormals[k] == SVector(0f0, 1f0)
        @test prescribed.Motion !== nothing
        @test_throws ArgumentError AllocateDataStructures([prescribed, prescribed])
        @test_throws ArgumentError SPHGeometry{2, Float32}(
            Particles = source, CSVFile = "unused", GroupMarker = 1, Type = Fluid)
        @test_throws DimensionMismatch SPHGeometry{3, Float32}(
            Particles = source, GroupMarker = 1, Type = Fluid)
        @test_throws DimensionMismatch SPHGeometry{2, Float32}(
            region.positions; Density = [1000], GroupMarker = 1, Type = Fixed)
        @test_throws ArgumentError SPHGeometry{2, Float32}(
            Particles = StructArray((Position = [SVector(0., 0.)],)),
            GroupMarker = 1, Type = Fluid)
    end
end

include(joinpath(@__DIR__, "..", "example", "GenerateStillWedgeMDBC.jl"))
@testset "StillWedge generates geometry without CSV" begin
    constants = SimulationConstants{Float64}(dx = 0.02)
    geometry = still_wedge_2d_geometry(constants)
    @test length.(getproperty.(geometry, :Particles)) == [580, 2447]
    @test all(geom -> isempty(geom.CSVFile), geometry)
    particles = AllocateDataStructures(geometry)
    @test length(particles) == 3027
    @test all(isfinite, particles.Density)
end
