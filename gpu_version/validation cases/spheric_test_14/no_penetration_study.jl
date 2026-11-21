# Compare the normal contact option with the saved UpdatedMDBC control.
include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "full_case.jl"))

function run_no_penetration_study(; smoke = false)
    release = first(RELEASES)
    tag = smoke ? "no_penetration_smoke" : "no_penetration_study"
    dx, duration = smoke ? (0.05, 0.05) : (0.025, 4.5)
    save_location = joinpath(@__DIR__, "runs", tag, release.name)
    output = joinpath(@__DIR__, "results", tag)
    case = build_full_case(release; dx, duration, save_location,
        boundary_mode = UpdatedMDBCNoPenetration)
    completed = run_case(release, case; output)
    title = merge(release, (; title = release.title * " | UpdatedMDBC normal contact"))
    write_heave_state(title, completed; layout = "stacked", screenshot = true)
end

if abspath(PROGRAM_FILE) == @__FILE__
    all(==("--smoke"), ARGS) || error("Usage: no_penetration_study.jl [--smoke]")
    run_no_penetration_study(; smoke = "--smoke" in ARGS)
end
