# Run sequentially: julia --project=. -t 2,0 "validation cases/spheric_test_06/wide_sponge.jl"
using SPHExampleGPU, CUDA, CSV, TOML, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06
include(joinpath(@__DIR__, "noise_metrics.jl"))

const SPONGE_WORK = joinpath(@__DIR__, "runs", "wide_sponge")
const SPONGE_RESULTS = joinpath(@__DIR__, "results", "wide_sponge")
const VARIANTS = (("original_tank", 0.0, false), ("expanded_tank", 2.0, false), ("expanded_sponge", 2.0, true))

for (name, padding, enabled) in VARIANTS
    directory = joinpath(SPONGE_WORK, name)
    results = joinpath(SPONGE_RESULTS, name)
    isfile(joinpath(results, "Re100_n040.toml")) && continue
    mkpath(directory)
    path = joinpath(@__DIR__, "manual_case.jl")
    source = replace(read(path, String),
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
    state["frames"] == 401 && state["frozen_particles"] == 0 || error("Incomplete $name")
    t, p, v = Float64.(data.time), Float64.(data.cd_pressure), Float64.(data.cd_viscous)
    metrics, _ = drag_metrics(t, p, v, reference)
    startup = findall(x -> 0.8 <= x <= 1.6, t)
    push!(rows, (; variant = name, padding, sponge = enabled,
        pressure_error = metrics["pressure_raw_nrmse"], viscous_error = metrics["viscous_raw_nrmse"],
        pressure_noise = metrics["pressure_highpass_rms"],
        pressure_minimum = minimum(p[startup]),
        pressure_mean_3_7 = bin_average(t, p, 3.0, 7.0), viscous_mean_3_7 = bin_average(t, v, 3.0, 7.0),
        minimum_clearance = state["minimum_clearance"], steps = state["steps"],
        particles = state["fluid_particles"] + state["square_particles"] + state["fixed_particles"]))
end
CSV.write(joinpath(SPONGE_RESULTS, "summary.csv"), rows)
foreach(println, rows)
