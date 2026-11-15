module VResolutionSimpleMDBCTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra
include(joinpath(@__DIR__, "vresolution_simple_mdbc.jl"))
using .VResolutionSimpleMDBC

@testset "Attached variable-resolution SimpleMDBC clearances and ghost symmetry" begin
    variants = vres_variants()
    @test length(variants) == 7
    @test length(unique(v.id for v in variants)) == 7
    for (coarse_n, ratio) in ((20, 2), (20, 4), (25, 4), (30, 4), (40, 4))
        case = build_vres_simple_mdbc_case(; reynolds = 100, coarse_n, ratio, boundary_mode = SimpleMDBC)
        @test case.fine_n == coarse_n * ratio
        @test !case.metadata.GPULegacyTimestep
        limits = VResolutionSimpleMDBC.VR.resolution_limits(
            case.vres_config.region, Float64(case.coarse_kernel.h), Float64(case.fine_kernel.h))
        # Fine outer box stays strictly inside (0, 10) × (0, 5) at both t = 0 and t = 8 s (dx_max = 7.47354 m).
        @test all(limits.fine_outer_lo .> SVector(0.0, 0.0))
        @test limits.fine_outer_hi[1] + 7.47354 < 10.0
        @test limits.fine_outer_hi[2] < 5.0
        # Shell-only vs full square must place identical boundary layers within H_fine of fluid.
        full = build_vres_simple_mdbc_case(; reynolds = 100, coarse_n, ratio, shell_only = false)
        shell_set = Set(case.fine_geometry[2].Particles.Position)
        for p in full.fine_geometry[2].Particles.Position
            if max(abs(p[1] - 1.5), abs(p[2] - 2.5)) > 0.5 - Float64(case.fine_kernel.H)
                @test p in shell_set
            end
        end
    end
    # PlanarShifting and BackgroundPressure wire cleanly into VariableResolution SimpleMDBC cases.
    shifted_case = build_vres_simple_mdbc_case(; reynolds = 100, coarse_n = 20, ratio = 2,
        shifting_mode = PlanarShifting, shifting_threshold = 0.0, background_pressure = 1200.0)
    @test shifted_case.coarse_constants.BackgroundPressure == Float32(1200.0)
    fine_c, _ = VResolutionSimpleMDBC.VR.fine_resolution_parameters(
        shifted_case.vres_config, shifted_case.coarse_constants, shifted_case.coarse_kernel)
    @test fine_c.BackgroundPressure == Float32(1200.0)
end

@testset "Saved post-fix variable-resolution results and plots are complete and consistent" begin
    res_root = joinpath(@__DIR__, "results", "vresolution_simple_mdbc")
    @test isfile(joinpath(res_root, "summary.csv"))
    @test isfile(joinpath(res_root, "diagnostic_ablation_points.csv"))
    @test isfile(joinpath(res_root, "vres_drag_histories.png"))
    @test isfile(joinpath(res_root, "vres_physics_ablation.png"))
    @test isfile(joinpath(res_root, "re150_c160_comparison.png"))
    summary_rows = collect(VResolutionSimpleMDBC.CSV.File(joinpath(res_root, "summary.csv")))
    @test length(summary_rows) >= 2
    for row in summary_rows
        @test row.frames == 401
        @test row.excluded_density == 0
        @test row.mass_drift_rel_max < 0.005
        @test row.motion_error_max < 1e-10
        @test isfile(joinpath(res_root, String(row.id), "native.csv"))
        @test isfile(joinpath(res_root, String(row.id), "binned.csv"))
        @test isfile(joinpath(res_root, String(row.id), "summary.toml"))
    end
end

if CUDA.functional() && get(ENV, "RUN_VRES_GPU_SMOKE", "0") == "1"
    @testset "Shell vs full square force equivalence and end-to-end variable-resolution smoke run" begin
        mktempdir() do dir
            c_shell = build_vres_simple_mdbc_case(; reynolds = 100, coarse_n = 20, ratio = 2,
                duration = 0.08, interval = 0.04, shell_only = true, save = joinpath(dir, "shell"))
            c_full = build_vres_simple_mdbc_case(; reynolds = 100, coarse_n = 20, ratio = 2,
                duration = 0.08, interval = 0.04, shell_only = false, save = joinpath(dir, "full"))
            info_shell = run_vres_case(c_shell; results_dir = joinpath(dir, "res_shell"))
            info_full = run_vres_case(c_full; results_dir = joinpath(dir, "res_full"))
            @test info_shell.frames == info_full.frames == 3
            @test info_shell.excluded_density == 0
            @test info_shell.motion_error_max < 1e-10
            @test info_shell.minimum_clearance > 0
            d_shell = VResolutionSimpleMDBC.CSV.File(joinpath(dir, "res_shell", "native.csv"))
            d_full = VResolutionSimpleMDBC.CSV.File(joinpath(dir, "res_full", "native.csv"))
            @test collect(d_shell.cd_pressure) ≈ collect(d_full.cd_pressure) rtol = 1e-5 atol = 1e-6
            @test collect(d_shell.cd_viscous) ≈ collect(d_full.cd_viscous) rtol = 1e-5 atol = 1e-6
        end
    end
end

end
