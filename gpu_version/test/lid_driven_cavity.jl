using Test
using SPHExampleGPU
using Meshes
using StaticArrays: SVector

include(joinpath(@__DIR__, "..", "example", "LidDrivenCavity2d.jl"))

@testset "Lid-driven cavity" begin
    dx = 0.05
    shapes = lid_driven_cavity_2d_shapes()
    constants = lid_driven_cavity_2d_constants(Float64; dx)

    @test shapes.walls isa PolyArea
    @test shapes.lid isa PolyArea
    @test shapes.fluid isa PolyArea
    @test Meshes.ustrip(measure(shapes.fluid)) ≈ 1.0
    @test Meshes.ustrip(measure(shapes.lid)) ≈
          LID_CAVITY_WALL_THICKNESS * (LID_CAVITY_LENGTH + 2LID_CAVITY_WALL_THICKNESS)
    @test Point(0.5, 1.02) in shapes.lid
    @test !(Point(0.5, 0.98) in shapes.lid)
    @test Point(-0.025, 1.02) in shapes.lid
    @test Point(1.025, 1.02) in shapes.lid
    @test Point(0.5, 0.5) in shapes.fluid
    @test constants.ρ₀ == 10.0
    @test constants.ν₀ == 0.01
    @test constants.c₀ == 10.0
    @test iszero(constants.g)
    @test constants.ρ₀ * LID_CAVITY_LID_SPEED * LID_CAVITY_LENGTH /
          LID_CAVITY_DYNAMIC_VISCOSITY == 100.0
    @test_throws ArgumentError lid_driven_cavity_2d_shapes(wall_thickness = 1.0)

    mktempdir() do directory
        input_dir = joinpath(directory, "input")
        regions = generate_lid_driven_cavity_2d(input_dir; dx)
        fixed, lid, fluid = regions
        @test [r.name for r in regions] == ["Fixed", "Lid", "Fluid"]
        @test all(r -> !isempty(r.positions), regions)
        @test all(r -> all(==(10.0), r.density), regions)
        @test all(isempty(intersect(Set(regions[i].positions), Set(regions[j].positions)))
                  for i in eachindex(regions) for j in (i + 1):length(regions))
        @test all(p -> -0.05 - 1e-12 <= p[1] <= 1.05 + 1e-12, lid.positions)
        @test all(p -> 1 - 1e-12 <= p[2] <= 1.05 + 1e-12, lid.positions)
        @test all(p -> 0 < p[1] < 1 && 0 < p[2] < 1, fluid.positions)
        @test all(p -> any(isapprox(p[1] / dx, k; atol = 1e-6) for k in -2:22) &&
                       any(isapprox(p[2] / dx, k; atol = 1e-6) for k in -2:22),
                  vcat(fixed.positions, lid.positions, fluid.positions))
        @test isfile(joinpath(input_dir, "LidDrivenCavity2D_Geometry.vtkhdf"))

        prefix = joinpath(input_dir, "LidDrivenCavity2D_Dp0.05")
        walls = vcat(fixed.positions, lid.positions)
        points, ghosts, normals =
            LoadBoundaryNormals(Val(2), Float64, "$(prefix)_GhostNodes.csv")
        @test length(ghosts) == length(walls)
        @test all(points[k] ≈ SVector(walls[k]...) for k in eachindex(walls))
        @test all(g -> 0 < g[1] < 1 && 0 < g[2] < 1, ghosts)
        @test all(k -> ghosts[k] == points[k] + normals[k], eachindex(ghosts))
        @test lid_driven_cavity_ghost_node((0.0, 0.5), dx) == (dx, 0.5)
        @test SVector(lid_driven_cavity_ghost_node((1.05, 0.5), dx)...) ≈ SVector(0.9, 0.5)
        @test SVector(lid_driven_cavity_ghost_node((0.5, 1.0), dx)...) ≈ SVector(0.5, 0.95)
        @test SVector(lid_driven_cavity_ghost_node((-0.05, -0.05), dx)...) ≈ SVector(0.1, 0.1)
        geometry = [
            SPHGeometry{2, Float32}(CSVFile = "$(prefix)_Fixed.csv",
                                    GroupMarker = 1, Type = Fixed),
            SPHGeometry{2, Float32}(CSVFile = "$(prefix)_Lid.csv",
                GroupMarker = 2, Type = Moving,
                Motion = MotionDetails{2, Float32}(
                    Velocity = 1.0f0, StartTime = 0.0f0, Duration = 1.0f0,
                    Direction = SVector{2, Float32}(1.0f0, 0.0f0),
                    MoveParticles = false)),
            SPHGeometry{2, Float32}(CSVFile = "$(prefix)_Fluid.csv",
                                    GroupMarker = 3, Type = Fluid),
        ]
        save_dir = joinpath(directory, "run")
        result = run_lid_driven_cavity_2d(; dx, simulation_time = 0.5,
            output_interval = 0.5, visualize = false, open_log_file = false,
            input_dir, save_location = save_dir)
        particles = result.particles

        lid_indices = findall(==(Moving), particles.Type)
        @test !isempty(lid_indices)
        @test all(i -> 1 - 1e-6 <= particles.Position[i][2] <= 1.05 + 1e-6,
                  lid_indices)
        @test all(i -> particles.Velocity[i] ≈ SVector(1.0f0, 0.0f0), lid_indices)
        fluid_indices = findall(==(Fluid), particles.Type)
        @test maximum(particles.Velocity[i][1] for i in fluid_indices) > 0.3f0
        @test all(i -> 0 < particles.Position[i][1] < 1 &&
                       0 < particles.Position[i][2] < 1, fluid_indices)
        @test all(i -> 8 < particles.Density[i] < 12, fluid_indices)
        boundary_indices = findall(!=(Fluid), particles.Type)
        @test all(i -> 10 <= particles.Density[i] < 12, boundary_indices)
    end
end

@testset "wall pairs do not change wall densities" begin
    dx = 0.02
    mktempdir() do directory
        regions = [ParticleRegion("Fixed", rectangle((0, 0), 0.2, 0.1), Fixed),
                   ParticleRegion("Slider", rectangle((0, 0.1), 0.2, 0.1), Moving)]
        sampled = sample_particles(regions, dx)
        next_id = 0
        for region in sampled
            next_id = write_particle_csv(joinpath(directory, region.name * ".csv"),
                                         region.positions; density = 1000.0, first_id = next_id)
        end

        geometry = [
            SPHGeometry{2, Float32}(CSVFile = joinpath(directory, "Fixed.csv"),
                                    GroupMarker = 1, Type = Fixed),
            SPHGeometry{2, Float32}(CSVFile = joinpath(directory, "Slider.csv"),
                GroupMarker = 2, Type = Moving,
                Motion = MotionDetails{2, Float32}(
                    Velocity = 1.0f0, StartTime = 0.0f0, Duration = 1.0f0,
                    Direction = SVector{2, Float32}(1.0f0, 0.0f0),
                    MoveParticles = false)),
        ]
        meta = SimulationMetaData{2, Float32, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(
            SimulationName = "WallPairs",
            SaveLocation = directory,
            SimulationTime = 0.05f0,
            OutputTimes = 0.05f0,
            VisualizeInParaview = false,
            OpenLogFile = false,
            GPUBoundaryForces = false,
        )
        particles = AllocateDataStructures(geometry, meta)
        RunSimulation(
            SimGeometry = geometry,
            SimMetaData = meta,
            SimConstants = SimulationConstants{Float32}(; dx = Float32(dx), c₀ = 20f0, g = 0f0),
            SimKernel = SPHKernelInstance{2, Float32}(WendlandC2(); dx = Float32(dx)),
            SimLogger = SimulationLogger(directory; to_console = false),
            SimParticles = particles,
            SimViscosity = Laminar(),
            SimDensityDiffusion = ZeroDensityDiffusion(),
            SimTimeStepping = SymplecticTimeStepping(),
        )
        @test meta.Iteration > 10
        @test all(==(1000f0), particles.Density)
        @test all(iszero, particles.Pressure)
    end
end
