module RZFlowTests

using Test, SPHExampleGPU, CUDA, StaticArrays, HDF5
include(joinpath(@__DIR__, "..", "example", "RZFlow2d", "FlowSetup.jl"))
const DZ = SPHExampleGPU.DampingZones

@testset "uniform flow relaxation" begin
    zone = rz_flow_case().relaxation
    center = SVector(0.75, 0.1)
    @test isbitstype(typeof(zone))
    @test DZ.normalized_damping(zone, center) ≈ 1
    @test DZ.normalized_damping(zone, SVector(0.6, 0.1)) == 0
    @test DZ.normalized_damping(zone, SVector(0.75, 0.2)) == 0
    @test DZ.normalized_damping(zone, SVector(0.75, 0.21)) == 0
    @test DZ.normalized_damping(zone, SVector(0.7, 0.1)) ≈
        DZ.normalized_damping(zone, SVector(0.8, 0.1))
    @test DZ.uniform_flow_speed(zone, 0) == 0
    @test DZ.uniform_flow_speed(zone, 2) ≈ 0.35
    @test DZ.uniform_flow_speed(zone, 9) == 0.2
    initial = SVector(0.1f0, 0.2f0)
    # At t = 1 s the target is (-0.3, 0); dt = 0.0005 gives a 50% blend.
    @test DZ.damped_velocity((zone,), center, initial, Fluid, 0.0005, 1.0) ≈
        SVector(-0.1f0, 0.1f0)
    @test DZ.damped_velocity((zone,), center, initial, Fluid, 1.0, 1.0) ≈
        SVector(-0.3f0, 0.0f0)
    for kind in (Fixed, Moving, Floating, Frozen)
        @test DZ.damped_velocity((zone,), center, initial, kind, 0.0005, 1.0) == initial
    end
    @test DZ.damped_velocity((zone,), SVector(1.0, 0.1), initial, Fluid, 1.0, 1.0) == initial
    @test DZ.prepare_damping(zone, Val(2)) == (zone,)
    @test_throws ArgumentError DZ.prepare_damping(zone, Val(3))
    @test_throws ArgumentError UniformFlowZone((0, 0), (1, 1);
        direction = (0, 0), times = (0, 1), speeds = (0, 1))
    @test_throws ArgumentError UniformFlowZone((0, 0), (1, 1);
        direction = (1, 0), times = (1, 0), speeds = (0, 1))
    direct = UniformFlowZone((0, 0), (1, 1);
        direction = (1, 0), times = (0, 1), speeds = (1, 1), coefdt = 0)
    @test DZ.damped_velocity((direct,), SVector(0.5, 0.5), initial, Fluid, 0.001, 1) ≈ SVector(1, 0)
    spatial = UniformFlowZone((0, 0, 0), (1, 2, 1);
        direction = (0, 1, 0), times = (0, 1), speeds = (1, 1), axis = 2)
    @test DZ.damped_velocity((spatial,), SVector(0.5, 1.0, 0.5),
        SVector(0.0f0, 0.0f0, 0.0f0), Fluid, 0.001, 1) ≈ SVector(0, 1, 0)
    absorber = DampingZone((0.0, 0.1), (0.75, 0.1); rate = 1)
    relaxed = DZ.damped_velocity((zone,), center, initial, Fluid, 0.0005, 1.0)
    @test DZ.damped_velocity((zone, absorber), center, initial, Fluid, 0.0005, 1.0) ≈
        DZ.damped_velocity((absorber,), center, relaxed, Fluid, 0.0005, 1.0)
end

@testset "reference flow geometry and output" begin
    case = rz_flow_case(; visualize = false)
    walls, water = case.geometry
    @test walls.Type == Fixed && water.Type == Fluid
    @test length(unique(walls.Particles.Position)) == length(walls.Particles.Position)
    @test isempty(intersect(Set(walls.Particles.Position), Set(water.Particles.Position)))
    @test extrema(first, walls.Particles.Position) == (0.0, 1.3)
    @test extrema(last, walls.Particles.Position) == (0.0, 0.8)
    @test maximum(last, water.Particles.Position) ≈ 0.445
    @test length(walls.Particles.Position) == 2847
    @test length(water.Particles.Position) == 21143
    @test SVector(0.22, 0.3) in walls.Particles.Position
    @test SVector(0.8, 0.22) in walls.Particles.Position
    @test all(p -> p[2] > 0 && p[1] > 0, water.Particles.Position)
    @test case.metadata.SimulationTime == 9
    @test case.constants.dx ≈ 0.005
    @test case.constants.α ≈ 0.05
    @test_throws ArgumentError rz_flow_case(; dx = 0.007)
    @test_throws ArgumentError rz_flow_case(; duration = -1)
    particles = AllocateDataStructures(case.geometry, case.metadata)
    @test length(particles.Position) == length(walls.Particles.Position) + length(water.Particles.Position)
    @test length(particles.Position) == 23990
    # Build output geometry without CUDA, catching unsupported-zone export errors.
    output = SPHExampleGPU.ProduceHDFVTK._damping_visualization_geometry(
        (case.relaxation,), particles.Position, case.kernel, 2)
    @test output !== nothing
    @test size(output.profile_points, 2) == 65
    @test maximum(output.profile_weights) ≈ 1
    @test first(output.profile_weights) == last(output.profile_weights) == 0
    mktempdir() do directory
        metadata = SimulationMetaData{2,Float32}(
            SimulationName = "RZFlowExport", SaveLocation = directory,
            OutputVariables = ["Type"], ExportSingleVTKHDF = true,
            VisualizeInParaview = false, OpenLogFile = false)
        writer = SPHExampleGPU.ProduceHDFVTK.SetupVTKOutput(
            metadata, particles, case.kernel, 2; damping = case.relaxation)
        try
            writer.save_particles(1, 0.0)
        finally
            writer.close_files()
        end
        h5open(joinpath(directory, "RZFlowExport.vtkhdf"), "r") do file
            @test haskey(file["VTKHDF"], "DampingZones")
            weights = read(file["VTKHDF/DampingProfiles/PointData/NormalizedDamping"])
            @test maximum(weights) ≈ 1
        end
    end
end

@testset "uniform flow CUDA kernel" begin
    if CUDA.functional()
        zone = rz_flow_case().relaxation
        points = CuArray([SVector(0.75, 0.1), SVector(0.75, 0.1), SVector(1.0, 0.1)])
        velocities = CuArray(fill(SVector(0.1f0, 0.2f0), 3))
        kinds = CuArray([Fluid, Fixed, Fluid])
        step = SPHExampleGPU.GPUStepState.HostStep(0.0005f0, 1.0 - 0.0005f0)
        DZ.launch_damping!((zone,), points, velocities, kinds, step)
        actual = Array(velocities)
        @test actual[1] ≈ SVector(-0.1f0, 0.1f0)
        @test actual[2] == actual[3] == SVector(0.1f0, 0.2f0)
    else
        @test_skip CUDA.functional()
    end
end

end
