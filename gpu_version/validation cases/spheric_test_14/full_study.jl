include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "full_case.jl"))
using CUDA

function full_study(; smoke = false)
    tag = smoke ? "full_mdbc_smoke" : "full_mdbc_study"
    output_root = joinpath(@__DIR__, "results", tag)
    mkpath(output_root)
    duration = smoke ? 0.05 : 4.5
    dx = smoke ? 0.05 : 0.025
    open(joinpath(output_root, "settings.toml"), "w") do io
        TOML.print(io, Dict("dimensions" => 3, "dx" => dx, "half_width" => 1.2,
            "duration" => duration, "output_interval" => 0.005, "kernel_h_factor" => 2.,
            "gpu" => CUDA.name(CUDA.device()), "julia" => string(VERSION), "cuda_jl" => string(pkgversion(CUDA)),
            "degrees_of_freedom" => "free surge/sway/heave/roll/pitch/yaw; experimental mass and center of gravity"))
    end
    for (mode, releases) in ((SimpleMDBC, RELEASES), (NoMDBC, (first(RELEASES),)))
        output = joinpath(output_root, string(nameof(mode)))
        for release in releases
            isfile(joinpath(output, release.name * ".toml")) && continue
            println("Starting full 3D ", nameof(mode), " / ", release.name); flush(stdout)
            case = build_full_case(release; dx, duration, boundary_mode = mode,
                save_location = joinpath(@__DIR__, "runs", tag, string(nameof(mode)), release.name))
            completed = run_case(release, case; output)
            title = merge(release, (; title = release.title * " | full 3D | $(nameof(mode))"))
            write_heave_state(title, completed; layout = "stacked", screenshot = true)
            println("Completed full 3D ", nameof(mode), " / ", release.name); flush(stdout)
            GC.gc(true); CUDA.reclaim()
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    all(==("--smoke"), ARGS) || error("Usage: full_study.jl [--smoke]")
    full_study(; smoke = "--smoke" in ARGS)
end
