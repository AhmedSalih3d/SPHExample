# Run sequentially: julia --project=. -t 2,0 "validation cases/spheric_test_06/wide_sponge.jl"
using SPHExampleGPU, CUDA, CSV, TOML, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06
include(joinpath(@__DIR__, "noise_metrics.jl"))

const SOUND_SPEED = isempty(ARGS) ? 40.0 : parse(Float64, only(ARGS))
isfinite(SOUND_SPEED) && SOUND_SPEED > 0 || throw(ArgumentError("Sound speed must be positive and finite"))
const STUDY_TAG = SOUND_SPEED == 40 ? "wide_sponge" : "wide_sponge_c$(SOUND_SPEED == round(SOUND_SPEED) ? string(Int(SOUND_SPEED)) : string(SOUND_SPEED))"
const SPONGE_WORK = joinpath(@__DIR__, "runs", STUDY_TAG)
const SPONGE_RESULTS = joinpath(@__DIR__, "results", STUDY_TAG)
const VARIANTS = (("original_tank", 0.0, false), ("expanded_tank", 2.0, false), ("expanded_sponge", 2.0, true))

for (name, padding, enabled) in VARIANTS
    directory = joinpath(SPONGE_WORK, name)
    results = joinpath(SPONGE_RESULTS, name)
    if isfile(joinpath(results, "Re100_n040.toml"))
        TOML.parsefile(joinpath(results, "Re100_n040.toml"))["sound_speed"] == Float64(Float32(SOUND_SPEED)) ||
            error("Saved sound speed does not match the requested study")
        continue
    end
    mkpath(directory)
    path = joinpath(@__DIR__, "manual_case.jl")
    source = replace(read(path, String),
        r"c₀\s*=\s*[0-9.]+" => "c₀ = $SOUND_SPEED",
        "domain_padding = 2.0" => "domain_padding = $padding",
        "use_sponge     = true" => "use_sponge     = $enabled",
        "visualize_in_paraview = true" => "visualize_in_paraview = false",
        "joinpath(@__DIR__, \"runs\", run_tag, run_name)" => repr(directory),
        "joinpath(@__DIR__, \"runs\", run_tag, \"paraview_force_plot\")" => repr(joinpath(directory, "plots")),
        "joinpath(@__DIR__, \"results\", run_tag)" => repr(results))
    println("Starting $name"); flush(stdout)
    open(joinpath(directory, "console.log"), "w") do io
        redirect_stdout(io) do
            sandbox = Module(gensym(:SpongeRun))
            Core.eval(sandbox, :(include(path) = Base.include($sandbox, path)))
            Base.include_string(sandbox, source, path)
        end
    end
    println("Completed $name"); flush(stdout)
    GC.gc(true); CUDA.reclaim()
end

reference = read_reference(joinpath(REFERENCE_DIR, "Force_Re100.dat"))
rows = NamedTuple[]
for (name, padding, enabled) in VARIANTS
    directory = joinpath(SPONGE_RESULTS, name)
    data = CSV.File(joinpath(directory, "Re100_n040.csv"))
    state = TOML.parsefile(joinpath(directory, "Re100_n040.toml"))
    state["sound_speed"] == Float64(Float32(SOUND_SPEED)) || error("Wrong sound speed in $name")
    state["frames"] == 401 && state["frozen_particles"] == 0 || error("Incomplete $name")
    t, p, v = Float64.(data.time), Float64.(data.cd_pressure), Float64.(data.cd_viscous)
    metrics, _ = drag_metrics(t, p, v, reference)
    startup = findall(x -> 0.8 <= x <= 1.6, t)
    push!(rows, (; variant = name, sound_speed = SOUND_SPEED, padding, sponge = enabled,
        pressure_error = metrics["pressure_raw_nrmse"], viscous_error = metrics["viscous_raw_nrmse"],
        pressure_noise = metrics["pressure_highpass_rms"],
        pressure_minimum = minimum(p[startup]),
        pressure_mean_3_7 = bin_average(t, p, 3.0, 7.0), viscous_mean_3_7 = bin_average(t, v, 3.0, 7.0),
        minimum_clearance = state["minimum_clearance"], steps = state["steps"],
        particles = state["fluid_particles"] + state["square_particles"] + state["fixed_particles"]))
end
CSV.write(joinpath(SPONGE_RESULTS, "summary.csv"), rows)
foreach(println, rows)
