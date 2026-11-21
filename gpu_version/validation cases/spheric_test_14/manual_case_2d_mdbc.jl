# Editable mDBC counterpart of manual_case_2d.jl. Running/including starts it.
include(joinpath(@__DIR__,"case.jl"))
include(joinpath(@__DIR__,"planar_case.jl"))

function run_manual_mdbc_case()
    selection = "float1_down" # or "all", "float1_up", "float2_down"
    boundary_mode = SimpleMDBC # NoMDBC is the matched DBC control.
    dx = 0.0125
    kernel_h_factor = 2.0
    duration = 4.5
    visualize = true
    tag = "manual_$(nameof(boundary_mode))_dp$(dx)_h$(kernel_h_factor)_t$(duration)"
    releases = selection == "all" ? RELEASES : filter(r -> r.name == selection,RELEASES)
    isempty(releases) && throw(ArgumentError("Unknown release $selection"))
    for release in releases
        directory = joinpath(@__DIR__,"runs",tag,release.name)
        case = build_planar_case(release;dx,kernel_h_factor,duration,boundary_mode,save_location=directory)
        output = run_case(release,case;output=joinpath(@__DIR__,"results",tag),vertical_axis=2)
        state = write_heave_state(merge(release,(;title=release.title*" | Cartesian 2D | $(nameof(boundary_mode))")),
            output;layout="stacked",screenshot=true,planar=true)
        println("ParaView launcher: ",state)
        executable = Sys.which("paraview")
        visualize && executable !== nothing && run(`$executable --state=$state`;wait=false)
    end
end

run_manual_mdbc_case()
