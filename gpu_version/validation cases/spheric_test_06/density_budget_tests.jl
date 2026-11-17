using Test, CSV
include(joinpath(@__DIR__, "density_budget.jl"))

@testset "Read-only density stage replay" begin
    mktempdir() do directory
        for model in ("original", "fluid_only", "zero")
            control = DensityBudgetStudy.run_budget(model; trace = false, dx = 0.1,
                duration = 0.04, start = 0.0, output_root = directory)
            traced = DensityBudgetStudy.run_budget(model; trace = true, dx = 0.1,
                duration = 0.04, start = 0.0, output_root = directory)
            @test control.result.performance.steps == traced.result.performance.steps
            @test control.analysis.rows == traced.analysis.rows
            for key in (:Position, :Velocity, :Density, :ID)
                @test getproperty(control.result.final_state.particles, key) ==
                    getproperty(traced.result.final_state.particles, key)
            end
            data = CSV.File(joinpath(directory, "results", "density_budget",
                "$(model)_replay_n10", "density_budget.csv"))
            @test Set(data.stage) == Set(("predictor", "corrector"))
            @test all(isfinite, data.raw_density)
            @test all(>=(0), data.floor_increment)
            if model == "zero"
                @test all(iszero, data.diffusion_rate)
            elseif model == "fluid_only"
                @test all(row -> row.type == Int(DensityBudgetStudy.Fluid) ||
                    row.diffusion_rate == 0, data)
            end
        end
    end
end
