# Full-length viscosity comparison at the settings of manual_case_2d.jl.
include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "planar_case.jl"))
using CUDA

function planar_viscosity_study(selection = "all")
    releases = selection == "all" ? RELEASES : filter(r -> r.name == selection, RELEASES)
    isempty(releases) && throw(ArgumentError("Unknown release: $selection"))
    tag = "planar_updated_viscosity"
    for release in releases, viscosity in (Laminar(), ZeroViscosity(), ArtificialViscosity(), LaminarSPS())
        model = string(nameof(typeof(viscosity)))
        output = joinpath(@__DIR__, "results", tag, model)
        if isfile(joinpath(output, release.name * ".csv")) && isfile(joinpath(output, release.name * ".toml"))
            continue
        end
        save_location = joinpath(@__DIR__, "runs", tag, model, release.name)
        case = build_planar_case(release; dx = 0.0125, half_width = 8.0, kernel_h_factor = 1.3,
            boundary_mode = UpdatedMDBC, viscosity, save_location, output_variables = ["Type"])
        println("Starting ", release.name, " / ", model)
        flush(stdout)
        completed = run_case(release, case; output, vertical_axis = 2)
        title = merge(release, (; title = release.title * " | planar UpdatedMDBC | " * model))
        write_heave_state(title, completed; layout = "stacked", screenshot = true, planar = true)
        println("Completed ", release.name, " / ", model)
        flush(stdout)
        GC.gc(true)
        CUDA.reclaim()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || error("Usage: planar_viscosity_study.jl [all or release]")
    planar_viscosity_study(isempty(ARGS) ? "all" : only(ARGS))
end
