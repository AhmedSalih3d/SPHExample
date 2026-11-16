# Optional arguments: L/dp (default 40), sound speed in m/s (default 160).
using SPHExampleGPU, CUDA, CSV, TOML, Statistics, LinearAlgebra
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06
include(joinpath(@__DIR__, "noise_metrics.jl"))

length(ARGS) <= 2 || throw(ArgumentError("Expected [L/dp] [sound_speed]"))
const RESOLUTION = isempty(ARGS) ? 40 : parse(Int, ARGS[1])
const SOUND_SPEED = length(ARGS) < 2 ? 160.0 : parse(Float64, ARGS[2])
RESOLUTION > 0 || throw(ArgumentError("Resolution L/dp must be positive"))
isfinite(SOUND_SPEED) && SOUND_SPEED > 0 || throw(ArgumentError("Sound speed must be positive and finite"))
const SPACING = 1.0 / RESOLUTION
const BASENAME = "Re100_n$(lpad(RESOLUTION, 3, '0'))"
const SPEED_TAG = isinteger(SOUND_SPEED) ? string(Int(SOUND_SPEED)) : string(SOUND_SPEED)
const STUDY_TAG = "motion_c$SPEED_TAG" * (RESOLUTION == 40 ? "" : "_n$(lpad(RESOLUTION, 3, '0'))")
const WORK = joinpath(@__DIR__, "runs", STUDY_TAG)
const RESULTS = joinpath(@__DIR__, "results", STUDY_TAG)
for kind in (:hermite, :integrated)
    directory = joinpath(WORK, string(kind))
    results = joinpath(RESULTS, string(kind))
    if isfile(joinpath(results, BASENAME * ".toml"))
        info = TOML.parsefile(joinpath(results, BASENAME * ".toml"))
        info["sound_speed"] == Float64(Float32(SOUND_SPEED)) && info["motion_kind"] == string(kind) &&
            info["dx"] == SPACING || error("Saved configuration differs")
        continue
    end
    base = build_case(; sound_speed = SOUND_SPEED, dx = SPACING, save = directory)
    case = merge(base, (; motion = PrescribedMotion(3; translation = body_trajectory(motion_kind = kind)),
        motion_kind = kind, boundary_mode = NoMDBC))
    mkpath(directory)
    println("Starting $kind"); flush(stdout)
    open(joinpath(directory, "console.log"), "w") do io
        redirect_stdout(io) do
            run_case(case; results_dir = results)
        end
    end
    println("Completed $kind"); flush(stdout)
    GC.gc(true); CUDA.reclaim()
end

reference = read_reference(joinpath(REFERENCE_DIR, "Force_Re100.dat"))
rows = NamedTuple[]
for kind in (:hermite, :integrated)
    directory = joinpath(RESULTS, string(kind))
    data = CSV.File(joinpath(directory, BASENAME * ".csv"))
    state = TOML.parsefile(joinpath(directory, BASENAME * ".toml"))
    state["frames"] == 401 && state["frozen_particles"] == 0 || error("Incomplete $kind")
    t, p, v = Float64.(data.time), Float64.(data.cd_pressure), Float64.(data.cd_viscous)
    metrics, _ = drag_metrics(t, p, v, reference)
    startup = findall(x -> 0.8 <= x <= 1.6, t)
    full_startup = findall(x -> 0.1 <= x <= 1.6, t)
    push!(rows, (; motion = kind, dx = SPACING, sound_speed = SOUND_SPEED,
        particles = state["fluid_particles"] + state["square_particles"] + state["fixed_particles"],
        pressure_error = metrics["pressure_raw_nrmse"],
        viscous_error = metrics["viscous_raw_nrmse"], pressure_noise = metrics["pressure_highpass_rms"],
        viscous_noise = metrics["viscous_highpass_rms"], startup_pressure_minimum = minimum(p[startup]),
        startup_pressure_minimum_0p1_1p6 = minimum(p[full_startup]),
        pressure_mean_3_7 = bin_average(t, p, 3., 7.), viscous_mean_3_7 = bin_average(t, v, 3., 7.),
        minimum_clearance = state["minimum_clearance"], steps = state["steps"]))
end
CSV.write(joinpath(RESULTS, "summary.csv"), rows)
foreach(println, rows)
