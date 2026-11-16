using Test, SPHExampleGPU, CUDA
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06
using StaticArrays: SVector
module SimpleManualSandbox end

@testset "Manual motion and boundary options" begin
    trajectory = body_trajectory(motion_kind = :integrated)
    delayed = body_trajectory(motion_kind = :integrated, speed_scale = 2, start_delay = 4)
    @test maximum(abs(trajectory_state(trajectory, t)[3][1]) for t in 2:0.001:7) < 1e-6
    @test maximum(abs(trajectory_state(trajectory, t)[1][1] - trajectory_state(body_trajectory(), t)[1][1]) for t in 0:0.01:8) < 3e-5
    @test iszero(trajectory_state(delayed, 2)[1])
    @test trajectory_state(delayed, 7)[1] ≈ 2trajectory_state(trajectory, 3)[1]
    @test trajectory_state(delayed, 7)[2] ≈ 2trajectory_state(trajectory, 3)[2]
    @test_throws ArgumentError body_trajectory(motion_kind = :invalid)
    layout = (; tank_origin = (2.0, -1.0), tank_size = (4.0, 3.0),
        body_center = (3.0, 0.55), body_size = (1.0, 0.5))
    kernel = SPHKernelInstance{2,Float32}(WendlandC2(); h = Float32(sqrt(2) * 0.1))
    sampled = SphericTest06.sample_case_particles(layout, 0.1, kernel)
    geometry = SphericTest06.manual_geometry(sampled, 1000f0, Float32; boundary_mode = SimpleMDBC)
    for group in (geometry[1], geometry[3])
        @test all(group.Particles.GhostPoints .≈ group.Particles.Position .+ group.Particles.GhostNormals)
        @test all(x -> !iszero(x), group.Particles.GhostNormals)
    end
    @test all(p -> all(sampled.layout.tank_origin .<= p .<= sampled.layout.tank_origin + sampled.layout.tank_size), geometry[1].Particles.GhostPoints)
    @test all(p -> any(abs.(p - sampled.layout.body_center) .> sampled.layout.body_size / 2), geometry[3].Particles.GhostPoints)
    @test_throws ArgumentError SphericTest06.manual_geometry(sampled, 1000f0, Float32; boundary_mode = UpdatedMDBC)
end

@testset "Wide sponge leaves the body path undamped" begin
    layout = (; tank_origin = (-2.0, -2.0), tank_size = (14.0, 9.0))
    sponge = SphericTest06.edge_sponge(layout; width = 2.0, rate = 100.0)
    damping = SPHExampleGPU.DampingZones
    trajectory = body_trajectory(motion_kind = :integrated)
    for time in 0:0.1:8
        center = SVector(1.5, 2.5) + trajectory_state(trajectory, time)[1]
        for corner in (SVector(-0.5, -0.5), SVector(0.5, -0.5), SVector(-0.5, 0.5), SVector(0.5, 0.5))
            @test damping.normalized_damping(sponge, center + corner) == 0
        end
    end
    @test damping.normalized_damping(sponge, SVector(-1.0, 2.5)) ≈ 0.25
    @test damping.normalized_damping(sponge, SVector(-2.0, 2.5)) ≈ 1.0
end

if CUDA.functional()
    @testset "SimpleMDBC manual GPU run" begin
        mktempdir() do directory
            path = joinpath(@__DIR__, "manual_case.jl")
            source = replace(read(path, String),
                "boundary_mode    = NoMDBC" => "boundary_mode    = SimpleMDBC",
                r"dx\s*=\s*0\.025" => "dx = 0.1",
                r"duration\s*=\s*8\.0" => "duration = 0.02",
                "visualize_in_paraview = true" => "visualize_in_paraview = false",
                "SimDivergenceCleaning = DivergenceCleaning(kappa=10)" => "SimDivergenceCleaning = nothing",
                "joinpath(@__DIR__, \"runs\", run_tag, run_name)" => repr(joinpath(directory, "run")),
                "joinpath(@__DIR__, \"runs\", run_tag, \"paraview_force_plot\")" => repr(joinpath(directory, "plots")),
                "joinpath(@__DIR__, \"results\", run_tag)" => repr(joinpath(directory, "results")))
            run = include_string(SimpleManualSandbox, source, path)
            @test run.case.boundary_mode === SimpleMDBC
            @test run.case.motion_kind === :integrated
            @test run.case.domain_padding == 2.0
            @test run.case.sponge_width == 2.0
            @test run.case.damping !== nothing
            @test run.completed.result.final_state.complete
            @test length(run.completed.analysis.rows) == 2
            @test all(row -> isfinite(row.cd_pressure) && isfinite(row.cd_viscous), run.completed.analysis.rows)
        end
    end
end
