using SPHExampleGPU, CUDA, TOML
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06

let
    isempty(ARGS) || ARGS == ["full"] || error("Usage: zero_ddt_startup.jl [full]")
    full = ARGS == ["full"]
    duration = full ? 8.0 : 0.9
    expected_frames = full ? 401 : 46
    tag = full ? "zero_ddt_full" : "zero_ddt_startup"
    directory = joinpath(@__DIR__, "results", tag)
    save = joinpath(@__DIR__, "runs", tag, "Re100_n080")
    record = joinpath(directory, "Re100_n080.toml")
    if isfile(record)
        m = TOML.parsefile(record)
        m["dx"] == 0.0125 && m["sound_speed"] == 160 &&
            m["h"] == Float64(Float32(1.2sqrt(2)*0.0125)) &&
            m["duration"] == duration && m["frames"] == expected_frames &&
            endswith(m["density_diffusion_model"], "ZeroDensityDiffusion") &&
            m["motion_kind"] == "integrated" && m["background_pressure"] == 0 ||
            error("Saved zero-DDT startup differs from the requested control")
        println("Verified completed zero-DDT $duration s run; no simulation rerun.")
    else
        case = build_case(; reynolds = 100, dx = 0.0125, duration, interval = 0.02,
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
        result.result.final_state.time == duration && length(result.analysis.rows) == expected_frames ||
            error("Incomplete zero-DDT run")
        SphericTest06.write_force_plot_state(case; results_dir = directory,
            output_dir = joinpath(save, "paraview_force_plot"))
        println("Completed Re=100 n=80 ZeroDensityDiffusion through $duration s.")
    end
end
