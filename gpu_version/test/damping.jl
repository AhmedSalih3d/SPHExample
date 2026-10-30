module DampingTests

using Test, SPHExampleGPU, CUDA, StaticArrays, HDF5
include(joinpath(@__DIR__, "..", "example", "Damping", "DampingSetup.jl"))
const DZ = SPHExampleGPU.DampingZones

@testset "plane, box, and cylinder damping profiles" begin
    velocity2 = SVector(2.0f0, -3.0f0)
    plane = DampingZone((0, 0), (2, 0); overlimit = 0.5, rate = 2)
    @test DZ.normalized_damping(plane, SVector(1.0, 0.0)) ≈ 0.25
    @test DZ.normalized_damping(plane, SVector(2.25, 0.0)) == 1
    @test DZ.normalized_damping(plane, SVector(2.51, 0.0)) == 0
    @test DZ.damped_velocity((plane,), SVector(1.0, 0.0), velocity2, Fluid, 0.1) ≈
        velocity2 * 0.95

    box = DampingBox(
        ((1, 1, 0), (3, 3, 2)),
        ((0, 0, 0), (4, 4, 2));
        directions = (:left, :right, :front, :back),
        overlimit = 0.5,
        rate = 2,
    )
    @test DZ.normalized_damping(box, SVector(2.0, 2.0, 1.0)) == 0
    @test DZ.normalized_damping(box, SVector(0.5, 2.0, 1.0)) ≈ 0.25
    @test DZ.normalized_damping(box, SVector(0.5, 0.5, 1.0)) ≈ 0.25
    @test DZ.normalized_damping(box, SVector(-0.51, 2.0, 1.0)) == 0
    @test DZ.normalized_damping(box, SVector(0.5, 2.0, -0.01)) == 0
    velocity3 = SVector(2.0f0, -3.0f0, 4.0f0)
    @test DZ.damped_velocity((box,), SVector(0.5, 2.0, 1.0), velocity3, Fluid, 0.1) ≈
        velocity3 * 0.95
    @test DZ.damped_velocity((box,), SVector(0.5, 2.0, 1.0), velocity3, Fixed, 0.1) ==
        velocity3

    cylinder = DampingCylinder((0, 0, 0), (0, 0, 1), 1, 2;
        overlimit = 0.5, rate = 2)
    @test DZ.normalized_damping(cylinder, SVector(1.5, 0.0, 0.5)) ≈ 0.25
    @test DZ.normalized_damping(cylinder, SVector(2.25, 0.0, 100.0)) == 1
    @test DZ.normalized_damping(cylinder, SVector(2.51, 0.0, 0.5)) == 0
    @test DZ.damped_velocity((cylinder,), SVector(1.5, 0.0, 0.5), velocity3, Fluid, 0.1) ≈
        velocity3 * 0.95

    @test_throws ArgumentError DampingBox(
        ((1, 1), (2, 2)), ((0, 0), (3, 3)); directions = (),
    )
    @test_throws ArgumentError DampingBox(
        ((1, 1), (2, 2)), ((0, 0), (3, 3)); directions = :front,
    )
    @test_throws ArgumentError DampingBox(
        ((1, 1), (2, 2)), ((1, 0), (3, 3)); directions = :left,
    )
    @test_throws ArgumentError DampingCylinder((0, 0, 0), (0, 0, 1), 2, 1)
    @test_throws DimensionMismatch DampingCylinder((0, 0), (0, 1), 0, 1)
    @test_throws ArgumentError DZ.prepare_damping(cylinder, Val(2))
end

@testset "DualSPHysics 20_Damping case builders" begin
    for variant in DAMPING_VARIANTS
        dx = variant in (:plane_mdbc, :plane) ? 0.05 : 0.025
        case = damping_case20(; variant, dx, duration = 0.1, visualize = false)
        @test case.variant == variant
        expected_groups = variant in (:plane_mdbc, :plane) ? 3 :
            variant == :angle ? 2 : 6
        @test length(case.geometry) == expected_groups
        @test all(!isempty(geometry.Particles.Position) for geometry in case.geometry)
        if variant in (:plane_mdbc, :plane)
            @test case.metadata isa SimulationMetaData{3,Float32,NoShifting,NoKernelOutput,SimpleMDBC}
            @test case.geometry[1].Type == Fixed
            @test case.geometry[2].Type == Moving
            @test case.geometry[3].Type == Fluid
            @test hasproperty(case.geometry[1].Particles, :GhostPoints)
            @test hasproperty(case.geometry[2].Particles, :GhostPoints)
            @test case.motion isa PrescribedMotion
            @test length(case.motion.translation.times) == 361
            @test first(case.motion.translation.times) == 0
            @test last(case.motion.translation.times) == 9
            @test maximum(abs(point[1]) for point in case.motion.translation.values) ≈ 0.026 atol = 0.001
            @test (case.damping === nothing) == (variant == :plane)
        elseif variant == :angle
            @test case.metadata isa SimulationMetaData{3,Float32,NoShifting,NoKernelOutput,NoMDBC}
            @test case.damping isa DampingZone
            water = case.geometry[2].Particles.Position
            @test maximum(point[1] for point in water) <= 0.3 + dx
            @test maximum(point[3] for point in water) <= 0.3 + dx
            @test case.damping.origin ≈ SVector(0.8, 0.5, 0.0)
            @test case.damping.length ≈ sqrt(0.5^2 + 0.5^2)
        elseif variant == :box
            @test case.metadata isa SimulationMetaData{3,Float32,NoShifting,NoKernelOutput,NoMDBC}
            @test case.damping isa DampingBox
            drops = (case.geometry[4].Particles.Position, case.geometry[6].Particles.Position)
            @test all(minimum(point[3] for point in drop) >= 0.15 - dx for drop in drops)
            @test all(maximum(point[3] for point in drop) <= 0.25 + dx for drop in drops)
            @test minimum(point[1] for point in drops[2]) > 1.15
        else
            @test case.metadata isa SimulationMetaData{3,Float32,NoShifting,NoKernelOutput,NoMDBC}
            @test case.damping isa DampingCylinder
            drops = (case.geometry[4].Particles.Position, case.geometry[6].Particles.Position)
            @test all(minimum(point[3] for point in drop) >= 0.15 - dx for drop in drops)
            @test all(maximum(point[3] for point in drop) <= 0.25 + dx for drop in drops)
            @test minimum(point[1] for point in drops[2]) >= 1.35 - 0.06 - dx
            @test maximum(point[1] for point in drops[2]) <= 1.35 + 0.06 + dx
        end
        mktempdir() do directory
            geometry_path = joinpath(directory, case.metadata.SimulationName * "_Geometry.vtkhdf")
            SavePolygonVTKHDF(geometry_path, case.shapes)
            @test isfile(geometry_path)
        end
    end
    @test_throws ArgumentError damping_case20(variant = :unknown)
    @test_throws ArgumentError damping_case20(variant = :cylinder, dx = 0.3)
end

@testset "box and cylinder damping wireframes export" begin
    zones = (
        DampingBox(
            ((1, 1, 0), (3, 3, 2)),
            ((0, 0, 0), (4, 4, 2));
            directions = (:left, :right, :front, :back),
            overlimit = 0.25,
        ),
        DampingCylinder((0, 0, 0), (0, 0, 1), 1, 2; overlimit = 0.25),
    )
    positions = [
        SVector(-1.0, -1.0, -1.0), SVector(5.0, 5.0, 3.0),
        SVector(2.0, 2.0, 1.0), SVector(1.5, 0.0, 0.5),
    ]
    particles = (Position = positions, Type = fill(Fluid, length(positions)))
    kernel = SPHKernelInstance{3,Float64}(WendlandC2(); h = 0.1)

    mktempdir() do directory
        metadata = SimulationMetaData{3,Float64}(
            SimulationName = "DampingShapes",
            SaveLocation = directory,
            OutputVariables = ["Type"],
            ExportSingleVTKHDF = true,
            ExportGridCells = false,
            VisualizeInParaview = false,
            OpenLogFile = false,
        )
        output = SPHExampleGPU.ProduceHDFVTK.SetupVTKOutput(
            metadata, particles, kernel, 3; damping = zones,
        )
        try
            output.save_particles(1, 0.0)
        finally
            output.close_files()
        end

        h5open(joinpath(directory, "DampingShapes.vtkhdf"), "r") do file
            root = file["VTKHDF"]
            @test haskey(root, "DampingZones")
            @test haskey(root, "DampingProfiles")
            @test size(read(root["DampingZones/Points"]), 1) == 3
            @test size(read(root["DampingProfiles/Points"]), 1) == 3
            @test !isempty(read(root["DampingZones/Lines/NumberOfCells"]))
            @test !isempty(read(root["DampingProfiles/Lines/NumberOfCells"]))
            weights = vec(read(root["DampingZones/PointData/NormalizedDamping"]))
            @test all(0 .<= weights) && all(weights .<= 1)
        end
    end
end

if CUDA.functional()
    @testset "box and cylinder damping CUDA kernels" begin
        for (zone, position, velocity, expected) in (
            (
                DampingBox(((1, 1), (3, 3)), ((0, 0), (4, 4));
                    directions = (:left, :right), rate = 2),
                SVector(0.5, 2.0),
                SVector(2.0f0, -3.0f0),
                SVector(1.9f0, -2.85f0),
            ),
            (
                DampingCylinder((0, 0, 0), (0, 0, 1), 1, 2; rate = 2),
                SVector(1.5, 0.0, 0.5),
                SVector(2.0f0, -3.0f0, 4.0f0),
                SVector(1.9f0, -2.85f0, 3.8f0),
            ),
        )
            device_position = CuArray([position])
            device_velocity = CuArray([velocity])
            device_type = CuArray([Fluid])
            DZ.launch_damping!((zone,), device_position, device_velocity, device_type,
                HostStep(0.1f0, 0.0f0))
            @test only(Array(device_velocity)) ≈ expected
        end
    end

    @testset "falling box and cylinder drops reach their pools" begin
        for variant in (:box, :cylinder)
            mktempdir() do directory
                result = run_damping_case20(; variant, dx = 0.025,
                    duration = 0.12, output_interval = 0.12,
                    visualize = false, save_location = directory)
                @test result.simulation.final_state.complete
                drop = findall(==(4), result.particles.GroupMarker)
                @test !isempty(drop)
                @test minimum(result.particles.Position[i][3] for i in drop) < 0.14
                @test isfile(joinpath(directory, result.case.metadata.SimulationName * ".vtkhdf"))
            end
        end
    end

    @testset "moving Simple mDBC paddle follows its reference history" begin
        mktempdir() do directory
            result = run_damping_case20(; variant = :plane_mdbc, dx = 0.05,
                duration = 1.2, output_interval = 1.2,
                visualize = false, save_location = directory)
            moving = findall(==(Moving), result.particles.Type)
            @test result.simulation.final_state.complete
            @test maximum(result.particles.Position[i][1] for i in moving) > 1.0 + 0.005
            @test all(
                result.particles.GhostPoints[i] ≈
                    result.particles.Position[i] + result.particles.GhostNormals[i]
                for i in moving
            )
        end
    end
end

end
