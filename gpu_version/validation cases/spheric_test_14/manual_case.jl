# Full 3D geometry; running/including this file starts the selected release(s).
# julia --project=. -t 2,0 "validation cases/spheric_test_14/manual_case.jl"
include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "full_case.jl"))

function run_manual_case()
    selection = "float1_down" # "all", "float1_up", or "float2_down"
    boundary_mode = SimpleMDBC # NoMDBC is the matched DBC control.
    dx = 0.025
    half_width, depth = 1.2, 0.8
    wall_top_z = 0.25
    kernel_family, kernel_h_factor = WendlandC2(), 2.0
    viscosity, diffusion = Laminar(), LinearDensityDiffusion()
    c0, cfl = 25., 0.2
    sponge_width, sponge_rate = 0.5, 12.
    duration, output_interval = 4.5, 0.005 # 200 Hz history and particle output.
    pause_time = 0. # Optional hold; the reference release occurs at t=0.
    visualize_in_paraview = true

    tag = "full_$(nameof(boundary_mode))_dp$(dx)_half$(half_width)_h$(kernel_h_factor)_t$(duration)"
    releases = selection == "all" ? RELEASES : filter(r -> r.name == selection, RELEASES)
    isempty(releases) && throw(ArgumentError("Unknown release $selection"))
    for (index, release) in enumerate(releases)
        case = build_full_case(release; dx, half_width, depth, wall_top_z, kernel_family,
            kernel_h_factor, boundary_mode, viscosity, diffusion, c0, cfl, sponge_width, sponge_rate,
            duration, output_interval, pause_time, save_location = joinpath(@__DIR__, "runs", tag, release.name))
        completed = run_case(release, case; output = joinpath(@__DIR__, "results", tag))
        title = merge(release, (; title = release.title * " | full 3D | $(nameof(boundary_mode))"))
        state = write_heave_state(title, completed; layout = "stacked", screenshot = true)
        println("ParaView launcher: ", state)
        executable = Sys.which("paraview")
        visualize_in_paraview && index == length(releases) && executable !== nothing &&
            run(`$executable --state=$state`; wait = false)
    end
end

run_manual_case()
