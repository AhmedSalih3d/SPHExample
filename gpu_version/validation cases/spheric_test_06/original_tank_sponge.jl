# Original 10×5 m tank; compare internal sponge widths without moving any walls.
# julia --project=. -t 2,0 "validation cases/spheric_test_06/original_tank_sponge.jl"
using SPHExampleGPU, CUDA, CSV, TOML, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06
include(joinpath(@__DIR__, "noise_metrics.jl"))

const WORK = joinpath(@__DIR__, "runs", "original_tank_sponge_c160")
const RESULTS = joinpath(@__DIR__, "results", "original_tank_sponge_c160")
const WIDTHS = (("sponge_w2", 2.0), ("sponge_w0p5", 0.5))

for (name, width) in WIDTHS
    results = joinpath(RESULTS, name)
    if isfile(joinpath(results, "Re100_n040.toml"))
        state = TOML.parsefile(joinpath(results, "Re100_n040.toml"))
        state["sound_speed"] == 160 && state["sponge_width"] == width &&
            state["sponge_rate"] == 100 && state["tank_size"] == [10, 5] || error("Saved settings differ")
        continue
    end
    directory = joinpath(WORK, name)
    case = build_case(; sound_speed = 160.0, save = directory)
    case = merge(case, (; motion = PrescribedMotion(3; translation = body_trajectory(motion_kind = :integrated)),
        motion_kind = :integrated, boundary_mode = NoMDBC,
        damping = SphericTest06.edge_sponge(case.layout; width, rate = 100.0),
        domain_padding = 0.0, sponge_width = width, sponge_rate = 100.0))
    mkpath(directory)
    println("Starting $name"); flush(stdout)
    open(joinpath(directory, "console.log"), "w") do io
        redirect_stdout(io) do
            run_case(case; results_dir = results)
        end
    end
    println("Completed $name"); flush(stdout)
    GC.gc(true); CUDA.reclaim()
end

reference = read_reference(joinpath(REFERENCE_DIR, "Force_Re100.dat"))
variants = [
    ("original_no_sponge", joinpath(@__DIR__, "results", "wide_sponge_c160", "original_tank")),
    [(name, joinpath(RESULTS, name)) for (name, _) in WIDTHS]...,
    ("expanded_no_sponge", joinpath(@__DIR__, "results", "wide_sponge_c160", "expanded_tank")),
    ("expanded_sponge", joinpath(@__DIR__, "results", "wide_sponge_c160", "expanded_sponge")),
]
rows, spectra = NamedTuple[], NamedTuple[]
for (name, directory) in variants
    data = CSV.File(joinpath(directory, "Re100_n040.csv"))
    state = TOML.parsefile(joinpath(directory, "Re100_n040.toml"))
    state["frames"] == 401 && state["frozen_particles"] == 0 || error("Incomplete $name")
    t, p, v = Float64.(data.time), Float64.(data.cd_pressure), Float64.(data.cd_viscous)
    metrics, spectrum = drag_metrics(t, p, v, reference)
    append!(spectra, [(; variant = name, row...) for row in spectrum])
    startup = findall(x -> 0.1 <= x <= 1.6, t)
    push!(rows, (; variant = name,
        pressure_error = metrics["pressure_raw_nrmse"], viscous_error = metrics["viscous_raw_nrmse"],
        pressure_noise = metrics["pressure_highpass_rms"], pressure_error_dominant_hz = metrics["pressure_dominant_hz"],
        startup_minimum = minimum(p[startup]),
        pressure_mean_3_7 = bin_average(t, p, 3., 7.), viscous_mean_3_7 = bin_average(t, v, 3., 7.),
        minimum_clearance = state["minimum_clearance"], steps = state["steps"]))
end
CSV.write(joinpath(RESULTS, "summary.csv"), rows)
CSV.write(joinpath(RESULTS, "spectra.csv"), spectra)
foreach(println, rows)
