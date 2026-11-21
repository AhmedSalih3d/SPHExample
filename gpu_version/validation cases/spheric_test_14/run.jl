include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "full_case.jl"))

function setup(release; dx = 0.025, half_width = 1.2, duration = 4.5, save, boundary_mode = SimpleMDBC)
    return build_full_case(release; dx, half_width, duration, save_location = save, boundary_mode)
end

function run_case(release; dx, half_width, duration, boundary_mode = SimpleMDBC)
    tag = "full_$(nameof(boundary_mode))_dp$(dx)_half$(half_width)_h2.0_t$(duration)"
    save = joinpath(@__DIR__, "runs", tag, release.name)
    output = joinpath(@__DIR__, "results", tag)
    return run_case(release, setup(release; dx, half_width, duration, save, boundary_mode); output)
end

function main(args)
    visualize = "--no-paraview" ∉ args
    positional = filter(!=("--no-paraview"), args)
    length(positional) <= 5 || error("Usage: run.jl [dx] [half-width] [duration] [release or all] [boundary mode] [--no-paraview]")
    dx, half_width, duration = (i <= length(positional) ? parse(Float64, positional[i]) : value
        for (i, value) in enumerate((0.025, 1.2, 4.5)))
    cases = length(positional) >= 4 && positional[4] != "all" ? filter(r -> r.name == positional[4], RELEASES) : RELEASES
    mode_name = length(positional) == 5 ? positional[5] : "SimpleMDBC"
    modes = Dict(string(nameof(mode)) => mode for mode in (SimpleMDBC,NoMDBC,UpdatedMDBC,UpdatedMDBCWithImpedance))
    haskey(modes,mode_name) || throw(ArgumentError("Choose SimpleMDBC, NoMDBC, UpdatedMDBC or experimental UpdatedMDBCWithImpedance"))
    boundary_mode = modes[mode_name]
    isempty(cases) && error("Release must be float1_down, float1_up or float2_down")

    first_state = nothing
    for release in cases
        output = run_case(release; dx, half_width, duration, boundary_mode)
        visualize || continue
        try
            title = merge(release, (; title = release.title * " | full 3D | $(nameof(boundary_mode))"))
            state = write_heave_state(title, output; layout = "stacked", screenshot = true)
            println("ParaView launcher saved to ", state)
            first_state === nothing && (first_state = state)
        catch exception
            exception isa InterruptException && rethrow()
            @warn "Could not create the combined ParaView view for $(release.name)" exception
        end
    end
    if first_state !== nothing
        paraview = Sys.which("paraview")
        if paraview === nothing
            @warn "ParaView was not found on PATH. Open $first_state with File > Load State."
        else
            try
                run(`$paraview --state=$first_state`; wait = false)
            catch exception
                exception isa InterruptException && rethrow()
                @warn "Could not open ParaView. Open $first_state with File > Load State." exception
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
