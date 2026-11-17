using SPHExampleGPU, CUDA, TOML, Test
using StaticArrays: SVector
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06

@testset "Zero-gravity fluid-only diffusion pair contract" begin
    for T in (Float32, Float64)
        constants = SimulationConstants{T}(2; dx = 0.0125, c₀ = 160, g = 0)
        kernel = SPHKernelInstance{2,T}(WendlandC2(); h = T(1.2sqrt(2)*0.0125))
        x = SVector{2,T}(0.01, 0)
        gradient = ∇Wᵢⱼ(kernel, sqrt(sum(abs2, x))*kernel.h⁻¹, x)
        for a in (Fluid, Fixed, Moving, Frozen, Floating), b in (Fluid, Fixed, Moving, Frozen, Floating)
            types = [a, b]
            args = (kernel, constants, nothing, x, gradient, sum(abs2, x),
                T(1001), T(1002), inv(T(1001)), inv(T(1002)), 1, 2, types)
            actual = SPHExampleGPU.SPHDensityDiffusionModels.compute_density_diffusion(LinearDensityDiffusion(), args...)
            reference = SPHExampleGPU.SPHDensityDiffusionModels.compute_density_diffusion(ZeroGravityLinearDensityDiffusion(), args...)
            if a == b == Fluid
                @test actual == reference
                @test !iszero(actual[1])
            else
                @test all(iszero, actual)
            end
        end
    end
end

let
    isempty(ARGS) || ARGS == ["test"] || error("Usage: fluid_only_ddt.jl [test]")
    if isempty(ARGS)
        directory = joinpath(@__DIR__, "results", "fluid_only_ddt")
        save = joinpath(@__DIR__, "runs", "fluid_only_ddt", "Re100_n080")
        record = joinpath(directory, "Re100_n080.toml")
        if isfile(record)
            m = TOML.parsefile(record)
            m["dx"] == 0.0125 && m["sound_speed"] == 160 &&
                m["h"] == Float64(Float32(1.2sqrt(2)*0.0125)) &&
                m["duration"] == 8 && m["frames"] == 401 &&
                last(split(m["density_diffusion_model"], '.')) == "LinearDensityDiffusion" &&
                m["gravity"] == 0 && m["motion_kind"] == "integrated" &&
                m["background_pressure"] == 0 || error("Saved fluid-only control differs")
            println("Verified completed fluid-only DDT control.")
        else
            case = build_case(; reynolds = 100, dx = 0.0125, duration = 8.0, interval = 0.02,
                sound_speed = 160.0, cfl = 0.2, kernel_h_factor = 1.2sqrt(2),
                density_diffusion_model = LinearDensityDiffusion(), save)
            case = merge(case, (; motion = PrescribedMotion(3;
                translation = body_trajectory(motion_kind = :integrated)), motion_kind = :integrated))
            mkpath(save)
            result = open(joinpath(save, "console.log"), "w") do io
                redirect_stdout(io) do
                    run_case(case; results_dir = directory)
                end
            end
            result.result.final_state.time == 8 && length(result.analysis.rows) == 401 ||
                error("Incomplete fluid-only DDT control")
            SphericTest06.write_force_plot_state(case; results_dir = directory,
                output_dir = joinpath(save, "paraview_force_plot"))
            println("Completed fluid-only DDT through 8 s.")
        end
    end
end
