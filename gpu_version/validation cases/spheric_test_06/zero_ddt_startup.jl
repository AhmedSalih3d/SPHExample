using SPHExampleGPU, CUDA, TOML
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06

let
    directory = joinpath(@__DIR__, "results", "zero_ddt_startup")
    save = joinpath(@__DIR__, "runs", "zero_ddt_startup", "Re100_n080")
    record = joinpath(directory, "Re100_n080.toml")
    if isfile(record)
        m = TOML.parsefile(record)
        m["dx"] == 0.0125 && m["sound_speed"] == 160 &&
            m["h"] == Float64(Float32(1.2sqrt(2)*0.0125)) &&
            m["duration"] == 0.9 && m["frames"] == 46 &&
            endswith(m["density_diffusion_model"], "ZeroDensityDiffusion") &&
            m["motion_kind"] == "integrated" && m["background_pressure"] == 0 ||
            error("Saved zero-DDT startup differs from the requested control")
        println("Verified completed zero-DDT startup; no simulation rerun.")
    else
        case = build_case(; reynolds = 100, dx = 0.0125, duration = 0.9, interval = 0.02,
            sound_speed = 160.0, cfl = 0.2, kernel_h_factor = 1.2sqrt(2),
            density_diffusion_model = ZeroDensityDiffusion(), save)
        case = merge(case, (; motion = PrescribedMotion(3;
            translation = body_trajectory(motion_kind = :integrated)), motion_kind = :integrated))
        mkpath(save)
        result = open(joinpath(save, "console.log"), "w") do io
            redirect_stdout(io) do
                run_case(case; results_dir = directory)
            end
        end
        result.result.final_state.time == 0.9 && length(result.analysis.rows) == 46 ||
            error("Incomplete zero-DDT startup")
        println("Completed Re=100 n=80 ZeroDensityDiffusion through 0.9 s.")
    end
end
