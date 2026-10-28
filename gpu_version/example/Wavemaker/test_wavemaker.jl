module WavemakerTests

using Test, SPHExampleGPU, StaticArrays, CUDA

include(joinpath(@__DIR__, "Wavemaker.jl"))

const PM = SPHExampleGPU.PrescribedMotions

@testset "wave tank geometry and periodic seam" begin
    case = wavemaker_geometry(; dx = 0.05)
    all_positions = vcat(values(case.positions)...)
    @test length(unique(all_positions)) == length(all_positions)
    @test all(!isempty, values(case.positions))
    @test case.tank_length == 5.0
    @test case.tank_width == 2.0
    @test case.water_depth == 0.75
    @test case.beach_start == 1.0
    @test case.beach_height == 1.1
    @test case.periodicity.axis == 2
    @test case.periodicity.lower ≈ -0.025
    @test case.periodicity.upper ≈ 2.025
    rows = sort(unique(x[2] for x in all_positions))
    @test all(diff(rows) .≈ 0.05)
    @test first(rows) + (case.periodicity.upper - case.periodicity.lower) - last(rows) ≈ 0.05
    @test all(x -> case.periodicity.lower <= x[2] < case.periodicity.upper, all_positions)
    @test all(x -> x[1] > case.piston_thickness && 0 < x[3] < 0.75, case.positions.water)
    @test all(x -> x[3] > max(0, 0.275 * (x[1] - 1)), case.positions.water)
    @test_throws ArgumentError wavemaker_geometry(; dx = 0)
    @test_throws ArgumentError wavemaker_geometry(; dx = NaN)
    @test_throws ArgumentError wavemaker_geometry(; boundary_layers = 1)
end

@testset "reference sinusoidal piston schedule" begin
    motion = wavemaker_motion()
    origin = (0.0, 0.0, 0.0)
    for t in (0.0, 0.25, 0.5, 1.0, 2.0, 4.75)
        @test motion_position(motion, origin, t) ≈ SVector(0.25 * (1 - cos(pi * t)), 0, 0) atol = 2e-15
        @test motion_velocity(motion, origin, t) ≈ SVector(0.25pi * sin(pi * t), 0, 0) atol = 2e-15
    end
    # Each segment uses local time and adds to the previous displacement.
    for (start, frequency) in ((5.0, 0.75), (15.0, 0.5), (25.0, 0.75), (35.0, 0.5))
        t = start + 1 / (4frequency)
        @test motion_position(motion, origin, t) ≈ SVector(0.75, 0, 0) atol = 1e-14
        @test motion_velocity(motion, origin, t) ≈ SVector(0, 0, 0) atol = 1e-13
    end
    @test motion_position(motion, origin, 100.1) ≈ motion_position(motion, origin, 100.0)
    @test iszero(motion_velocity(motion, origin, 100.1))
    @test PM.validate_periodic_motion(motion, PeriodicBoundary{3}(2; min = 0, max = 2)) === nothing
end

@testset "periodic motion validation checks complete trajectories" begin
    for D in (2, 3), axis in 1:D
        periodicity = PeriodicBoundary{D}(axis; min = -1, max = 1)
        transverse = ntuple(d -> d == mod1(axis + 1, D) ? 0.1 : 0.0, D)
        axial = ntuple(d -> d == axis ? 0.1 : 0.0, D)
        origin = ntuple(_ -> 0.0, D)
        paths = (
            MotionSequence([SinusoidalSegment(1; amplitude = transverse, frequency = transverse)]),
            SampledTrajectory([0, 1], [origin, transverse]; velocities = [origin, transverse]),
        )
        for translation in paths
            node = PrescribedMotion(1; translation, center = origin)
            @test PM.validate_periodic_motion(MotionSystem(node), periodicity) === nothing
        end
        bad_paths = (
            MotionSequence([KinematicSegment(1; velocity = axial)]),
            MotionSequence([KinematicSegment(1; velocity = origin)]; jumps = [axial]),
            SampledTrajectory([0, 1], [origin, origin]; velocities = [axial, origin]),
            TrajectoryWindow(first(paths); finish_offset = axial),
        )
        for translation in bad_paths
            @test_throws ArgumentError PM.validate_periodic_motion(PrescribedMotion(1; translation, center = origin), periodicity)
        end
        rotation = MotionSequence([KinematicSegment(1; velocity = 0.1)])
        @test_throws ArgumentError PM.validate_periodic_motion(PrescribedMotion(1; rotation, center = origin), periodicity)
        parent = PrescribedMotion(1; group_marker = nothing, center = origin)
        child = PrescribedMotion(2; parent = 1, center = origin)
        @test_throws ArgumentError PM.validate_periodic_motion(MotionSystem((parent, child)), periodicity)
    end
end

if CUDA.functional()
    @testset "wave tank GPU integration with direct steps and CUDA graphs" begin
        for use_graph in (false, true)
            mktempdir() do dir
                result = run_wavemaker(; dx = 0.1, simulation_time = 0.3,
                    output_interval = 0.1, visualize = false, open_log_file = false,
                    use_graph, save_location = dir)
                particles = result.final_state.particles
                @test result.termination_reason === :completed
                @test result.final_state.time ≈ 0.3f0
                @test all(isfinite, particles.Density)
                @test all(x -> all(isfinite, x), particles.Position)
                @test all(x -> all(isfinite, x), particles.Velocity)
                @test result.diagnostics.frozen_particles == 0
                @test result.diagnostics.cell_list_rebuilds > 1
                @test isfile(joinpath(dir, "Wavemaker_Piston.csv"))
                @test isfile(joinpath(dir, "Wavemaker_Motion.vtkhdf"))
                case = wavemaker_geometry(; dx = 0.1)
                t = Float64(result.final_state.time)
                expected = [SVector{3,Float32}(motion_position(wavemaker_motion(), x, t)) for x in case.positions.piston]
                actual = particles.Position[particles.GroupMarker .== 2]
                @test sort(actual) ≈ sort(expected) atol = 2e-6
                velocity = SVector{3,Float32}(motion_velocity(wavemaker_motion(), (0, 0, 0), t))
                @test all(v -> isapprox(v, velocity; atol = 2e-6), particles.Velocity[particles.GroupMarker .== 2])
                if use_graph
                    @test result.diagnostics.captured_graphs > 0
                end
            end
        end
    end
end

end # module WavemakerTests
