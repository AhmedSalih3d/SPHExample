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
    @test Meshes.ustrip(measure(shapes.lid)) ≈ LID_CAVITY_WALL_THICKNESS
    @test Point(0.5, 1.02) in shapes.lid
    @test !(Point(0.5, 0.98) in shapes.lid)
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
        @test all(p -> -1e-12 <= p[1] <= 1 + 1e-12, lid.positions)
        @test all(p -> 1 - 1e-12 <= p[2] <= 1.05 + 1e-12, lid.positions)
        @test all(p -> 0 < p[1] < 1 && 0 < p[2] < 1, fluid.positions)
        @test all(p -> any(isapprox(p[1] / dx, k; atol = 1e-6) for k in -2:22) &&
                       any(isapprox(p[2] / dx, k; atol = 1e-6) for k in -2:22),
                  vcat(fixed.positions, lid.positions, fluid.positions))
        @test isfile(joinpath(input_dir, "LidDrivenCavity2D_Geometry.vtkhdf"))

        prefix = joinpath(input_dir, "LidDrivenCavity2D_Dp0.05")
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
        mkpath(save_dir)
        meta = SimulationMetaData{2, Float32, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(
            SimulationName = "LidDrivenCavityTest",
            SaveLocation = save_dir,
            SimulationTime = 0.03f0,
            OutputTimes = 0.03f0,
            VisualizeInParaview = false,
            ExportSingleVTKHDF = true,
            OpenLogFile = false,
        )
        particles = AllocateDataStructures(geometry, meta)
        initial_lid = Dict(particles.ID[i] => particles.Position[i]
                           for i in eachindex(particles.ID) if particles.Type[i] == Moving)
        RunSimulation(
            SimGeometry = geometry,
            SimMetaData = meta,
            SimConstants = lid_driven_cavity_2d_constants(Float32; dx),
            SimKernel = SPHKernelInstance{2, Float32}(WendlandC2();
                h = Float32(1.2 * sqrt(2) * dx)),
            SimLogger = SimulationLogger(save_dir; to_console = false),
            SimParticles = particles,
            SimViscosity = Laminar(),
            SimDensityDiffusion = ZeroDensityDiffusion(),
            SimTimeStepping = SymplecticTimeStepping(),
        )

        lid_indices = findall(==(Moving), particles.Type)
        @test !isempty(lid_indices)
        @test all(i -> particles.Position[i] == initial_lid[particles.ID[i]], lid_indices)
        @test all(i -> particles.Velocity[i] ≈ SVector(1.0f0, 0.0f0), lid_indices)
        fluid_indices = findall(==(Fluid), particles.Type)
        @test maximum(particles.Velocity[i][1] for i in fluid_indices) > 1f-6
    end
end
