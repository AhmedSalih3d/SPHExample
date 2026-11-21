# Matched to the 25 mm Laminar runs in results/floating_updated_study.
include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "full_case.jl"))
using CUDA

function viscosity_study(selection = "all")
    releases = selection == "all" ? RELEASES : filter(r -> r.name == selection, RELEASES)
    isempty(releases) && throw(ArgumentError("Unknown release: $selection"))
    tag = "updated_viscosity_study"
    for release in releases, viscosity in (ZeroViscosity(), ArtificialViscosity(), LaminarSPS())
        model = string(nameof(typeof(viscosity)))
        output = joinpath(@__DIR__, "results", tag, model)
        history = joinpath(output, release.name * ".csv")
        metrics = joinpath(output, release.name * "_metrics.csv")
        if isfile(history) && isfile(joinpath(output, release.name * ".toml"))
            isfile(metrics) || save_heave_plot(output, release.name)
            continue
        end
        save_location = joinpath(@__DIR__, "runs", tag, model, release.name)
        case = build_full_case(release; boundary_mode = UpdatedMDBC, viscosity, save_location,
            output_variables = ["Type"])
        @assert case.constants.α == Float32(0.01)
        println("Starting ", release.name, " / ", model)
        flush(stdout)
        completed = run_case(release, case; output)
        title = merge(release, (; title = release.title * " | UpdatedMDBC | " * model))
        write_heave_state(title, completed; layout = "stacked", screenshot = true)
        println("Completed ", release.name, " / ", model)
        flush(stdout)
        GC.gc(true)
        CUDA.reclaim()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || error("Usage: viscosity_study.jl [all or release]")
    viscosity_study(isempty(ARGS) ? "all" : only(ARGS))
end
