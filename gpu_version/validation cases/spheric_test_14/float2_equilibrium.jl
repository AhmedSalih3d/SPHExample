include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "full_case.jl"))
using Statistics

"""Draft of the rounded Float 2 hull at the measured mass in fresh water."""
function hydrostatic_float2_draft(release, density)
    r = release.corner
    a = release.radius - r
    # Integrate pi*(a + sqrt(r^2 - (z-r)^2))^2 from the base to z=r.
    rounded_volume = pi * (a^2 * r + 2r^3 / 3 + pi * a * r^2 / 2)
    draft = r + (release.mass / density - rounded_volume) / (pi * release.radius^2)
    r < draft < release.height || error("Waterline must lie on the cylindrical side")
    return draft
end

function float2_study(half_width; wider = false, hydrostatic = false)
    release = only(filter(r -> r.name == "float2_down", RELEASES))
    tag = "float2_equilibrium_half$(half_width)"
    function simulate(label, setup; duration, pause_time, output_interval, width = half_width)
        output = joinpath(@__DIR__, "results", tag, label)
        if !isfile(joinpath(output, setup.name * ".toml"))
            case = build_full_case(setup; half_width = width, duration, pause_time, output_interval,
                output_variables = ["Type"], save_location = joinpath(@__DIR__, "runs", tag, label))
            run_case(setup, case; output)
        end
        return CSV.File(joinpath(output, setup.name * "_body.csv"))
    end
    offsets = [-0.025, 0.0, 0.025]
    loads = map(offsets) do offset
        rows = simulate("held_$(round(Int, 1000offset))mm", merge(release, (; offset));
            duration = 2., pause_time = 3., output_interval = 0.025)
        center = getproperty(rows, Symbol("Center:2"))
        maximum(abs.(center .- first(center))) < 1e-7 || error("Float moved during hold")
        mean(getproperty(rows, Symbol("Force:2"))[rows.Time .>= 1.])
    end
    weight = release.mass * 9.81
    all(diff(loads) .< 0) && last(loads) < weight < first(loads) || error("Loads do not bracket weight: $loads")
    i = findfirst(<=(weight), loads)
    shift = offsets[i-1] + (weight-loads[i-1]) * (offsets[i]-offsets[i-1]) / (loads[i]-loads[i-1])
    println("Held loads (N) ", loads, ", inferred equilibrium shift (m) ", shift)
    corrected = merge(release, (; draft = release.draft - shift))
    simulate("release", corrected;
        duration = 4.5, pause_time = 0., output_interval = 0.005)
    if wider
        simulate("wide_release", corrected; width = 1.5half_width,
            duration = 4.5, pause_time = 0., output_interval = 0.005)
    end
    if hydrostatic
        physical = merge(release, (; draft = hydrostatic_float2_draft(release, 1000.)))
        simulate("wide_hydrostatic", physical; width = 1.5half_width,
            duration = 4.5, pause_time = 0., output_interval = 0.005)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    all(in(("--wider", "--hydrostatic")), ARGS) ||
        error("Usage: float2_equilibrium.jl [--wider] [--hydrostatic]")
    float2_study(1.2; wider = "--wider" in ARGS, hydrostatic = "--hydrostatic" in ARGS)
end
