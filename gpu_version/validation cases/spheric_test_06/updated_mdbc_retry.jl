# Fresh matched DBC/UpdatedMDBC comparison at c0=160, original geometry/spacing.
# julia --project=. -t 2,0 "validation cases/spheric_test_06/updated_mdbc_retry.jl"
using SPHExampleGPU, CUDA, CSV, TOML
include(joinpath(@__DIR__, "mdbc_validation.jl"))
const M = MDBCValidation
const C = M.C
const PAPER_AUDIT = "--paper-audit" in ARGS
const CLEANING = "--cleaning" in ARGS
const SPEED_ARGS = filter(x -> x ∉ ("--paper-audit", "--cleaning"), ARGS)
length(SPEED_ARGS) <= 1 || throw(ArgumentError("Expected [--paper-audit] [--cleaning] [sound_speed]"))
CLEANING && !PAPER_AUDIT && throw(ArgumentError("Use --paper-audit for cleaning studies"))
const SOUND_SPEED = isempty(SPEED_ARGS) ? 160.0 : parse(Float64, only(SPEED_ARGS))
isfinite(SOUND_SPEED) && SOUND_SPEED > 0 || throw(ArgumentError("Sound speed must be positive and finite"))
PAPER_AUDIT || SOUND_SPEED == 160 || throw(ArgumentError("Use --paper-audit for corrected studies at another sound speed"))
const SPEED_TAG = isinteger(SOUND_SPEED) ? string(Int(SOUND_SPEED)) : string(SOUND_SPEED)
# This selects a new output family; the current solver determines the equations.
const STUDY_TAG = (PAPER_AUDIT ? "updated_mdbc_paper_audit_c" : "updated_mdbc_retry_c") * SPEED_TAG * (CLEANING ? "_cleaning" : "")
const CLEANING_MODEL = CLEANING ? DivergenceCleaning(kappa = 10, sigma = 1, courant = 0.1, pressure_mode = :absolute) : nothing
const WORK = joinpath(@__DIR__, "runs", STUDY_TAG)
const RESULTS = joinpath(@__DIR__, "results", STUDY_TAG)
const VARIANTS = (("dbc", NoMDBC, NoShifting),
    ("updated", UpdatedMDBC, NoShifting), ("updated_shifting", UpdatedMDBC, PlanarShifting))

for (name, boundary, shifting) in VARIANTS
    results = joinpath(RESULTS, name)
    if isfile(joinpath(results, "native.toml"))
        info = TOML.parsefile(joinpath(results, "native.toml"))
        info["sound_speed"] == Float64(Float32(SOUND_SPEED)) && info["boundary_mode"] == string(nameof(boundary)) &&
            info["shifting_mode"] == string(nameof(shifting)) || error("Saved configuration differs")
        get(info, "divergence_cleaning", false) == CLEANING || error("Saved cleaning configuration differs")
        if CLEANING
            info["cleaning_kappa"] == CLEANING_MODEL.kappa && info["cleaning_sigma"] == CLEANING_MODEL.sigma &&
                info["cleaning_courant"] == CLEANING_MODEL.courant &&
                info["cleaning_absolute_pressure"] == CLEANING_MODEL.absolute_pressure || error("Saved cleaning parameters differ")
        end
        continue
    end
    println("Starting $name"); flush(stdout)
    case = M.build_mdbc_case(; sound_speed = SOUND_SPEED, boundary_mode = boundary,
        shifting_mode = shifting, motion_kind = :integrated, legacy_timestep = true,
        save = joinpath(WORK, name))
    M.run_mdbc_case(case; results_dir = results, cleaning = CLEANING_MODEL)
    println("Completed $name"); flush(stdout)
    GC.gc(true); CUDA.reclaim()
end

reference = C.read_reference(joinpath(C.REFERENCE_DIR, "Force_Re100.dat"))
rows = NamedTuple[]
for (name, boundary, shifting) in VARIANTS
    directory = joinpath(RESULTS, name)
    data = CSV.File(joinpath(directory, "native.csv"))
    info = TOML.parsefile(joinpath(directory, "native.toml"))
    info["frames"] == 401 && info["particles"] == 84864 || error("Incomplete $name")
    t, p, v = Float64.(data.time), Float64.(data.cd_pressure), Float64.(data.cd_viscous)
    metrics, _ = C.drag_metrics(t, p, v, reference)
    push!(rows, (; variant = name, sound_speed = SOUND_SPEED, boundary = string(nameof(boundary)), shifting = string(nameof(shifting)),
        pressure_error = metrics["pressure_raw_nrmse"], viscous_error = metrics["viscous_raw_nrmse"],
        pressure_noise = metrics["pressure_highpass_rms"], minimum_clearance = info["minimum_clearance"],
        density_relative_max = info["density_relative_max"],
        pressure_mean_3_7 = C.bin_average(t, p, 3., 7.), viscous_mean_3_7 = C.bin_average(t, v, 3., 7.),
        steps = info["steps"]))
end
CSV.write(joinpath(RESULTS, "summary.csv"), rows)
foreach(println, rows)
