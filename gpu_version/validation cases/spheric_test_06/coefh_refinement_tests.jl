using Test, TOML
include(joinpath(@__DIR__, "coefh_refinement.jl"))
const R = CoefhRefinement

@testset "Raw refinement configuration and comparison" begin
    for re in (50, 100, 150), n in (20, 40, 80)
        case = R.configuration(re, n)
        @test case.layers == 5
        @test case.counts.fluid == 49n^2
        @test case.counts.square == n^2
        @test case.constants.ν₀ == Float32(1 / re)
        @test case.kernel.h == Float32(1.2sqrt(2) / n)
        @test case.metadata.GPULegacyTimestep
        @test case.motion_kind == :integrated
        @test case.constants.BackgroundPressure == 0
        @test case.damping === case.divergence_cleaning === nothing
    end
    record = TOML.parsefile(joinpath(@__DIR__, "results", "coefh_reynolds", "re100_bg0", "Re100_n040.toml"))
    @test R.validate_record(record, 100, 40) === record
    qualified = copy(record)
    qualified["kernel"] = "SPHExampleGPU.SPHKernels.WendlandC2"
    @test R.validate_record(qualified, 100, 40) === qualified
    for key in ("sound_speed", "background_pressure", "h", "cfl", "frames")
        changed = copy(record)
        changed[key] = -1
        @test_throws ErrorException R.validate_record(changed, 100, 40)
    end
    @test R.relative_difference([0., 4., 8.], [1., 1., 1.], [0., 8.], [1., 1.]) == 0
    @test R.relative_difference([0., 4., 8.], [2., 2., 2.], [0., 8.], [1., 1.]) ≈ 1
    @test R.relative_difference([0., 4., 8.], [1., 1., 1.], [0., 8.], [2., 2.]; residual = false) ≈ 0.5
    @test_throws ErrorException R.relative_difference([0., 4.], [1., 1.], [0., 8.], [1., 1.])
    mktempdir() do directory
        source = joinpath(@__DIR__, "results", "coefh_reynolds", "re100_bg0")
        R.import_control(source, directory, "Re100_n040", 100, 40)
        R.import_control(source, directory, "Re100_n040", 100, 40)
        @test R.history(directory, 100, 40).t[end] == 8
        write(joinpath(directory, "Re100_n040.csv"), "changed")
        @test_throws ErrorException R.import_control(source, directory, "Re100_n040", 100, 40)
    end
end
