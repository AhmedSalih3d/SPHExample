include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "full_case.jl"))
using CUDA

function updated_floating_study(; smoke = false)
    tag = smoke ? "floating_updated_smoke" : "floating_updated_study"
    output_root = joinpath(@__DIR__, "results", tag)
    mkpath(output_root)
    duration, dx = smoke ? (0.05, 0.05) : (4.5, 0.025)
    for (mode, releases, cfl) in ((UpdatedMDBC, RELEASES, 0.2),
            (UpdatedMDBCWithImpedance, (first(RELEASES),), 0.2), (UpdatedMDBC, (first(RELEASES),), 0.1))
        variant = string(nameof(mode)) * (cfl == 0.2 ? "" : "_CFL0.1")
        output = joinpath(output_root, variant)
        for release in releases
            isfile(joinpath(output, release.name * ".toml")) && continue
            case = build_full_case(release; dx, duration, cfl, boundary_mode = mode,
                save_location = joinpath(@__DIR__, "runs", tag, variant, release.name))
            println("Starting ", nameof(mode), " / ", release.name); flush(stdout)
            completed = run_case(release, case; output)
            title = merge(release, (; title = release.title * " | full 3D | $(nameof(mode))"))
            write_heave_state(title, completed; layout = "stacked", screenshot = true)
            println("Completed ", nameof(mode), " / ", release.name); flush(stdout)
            GC.gc(true); CUDA.reclaim()
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    all(==("--smoke"), ARGS) || error("Usage: updated_floating_study.jl [--smoke]")
    updated_floating_study(; smoke = "--smoke" in ARGS)
end
