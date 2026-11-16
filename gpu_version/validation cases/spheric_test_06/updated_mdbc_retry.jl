# Fresh matched DBC/UpdatedMDBC comparison at c0=160, original geometry/spacing.
# julia --project=. -t 2,0 "validation cases/spheric_test_06/updated_mdbc_retry.jl"
using SPHExampleGPU, CUDA, CSV, TOML
include(joinpath(@__DIR__, "mdbc_validation.jl"))
const M = MDBCValidation
const C = M.C
const WORK = joinpath(@__DIR__, "runs", "updated_mdbc_retry_c160")
const RESULTS = joinpath(@__DIR__, "results", "updated_mdbc_retry_c160")
const VARIANTS = (("dbc", NoMDBC, NoShifting),
    ("updated", UpdatedMDBC, NoShifting), ("updated_shifting", UpdatedMDBC, PlanarShifting))

for (name, boundary, shifting) in VARIANTS
    results = joinpath(RESULTS, name)
    if isfile(joinpath(results, "native.toml"))
        info = TOML.parsefile(joinpath(results, "native.toml"))
        info["sound_speed"] == 160 && info["boundary_mode"] == string(nameof(boundary)) &&
            info["shifting_mode"] == string(nameof(shifting)) || error("Saved configuration differs")
        continue
    end
    println("Starting $name"); flush(stdout)
    case = M.build_mdbc_case(; sound_speed = 160.0, boundary_mode = boundary,
        shifting_mode = shifting, motion_kind = :integrated, legacy_timestep = true,
        save = joinpath(WORK, name))
    M.run_mdbc_case(case; results_dir = results)
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
    push!(rows, (; variant = name, boundary = string(nameof(boundary)), shifting = string(nameof(shifting)),
        pressure_error = metrics["pressure_raw_nrmse"], viscous_error = metrics["viscous_raw_nrmse"],
        pressure_noise = metrics["pressure_highpass_rms"], minimum_clearance = info["minimum_clearance"],
        density_relative_max = info["density_relative_max"],
        pressure_mean_3_7 = C.bin_average(t, p, 3., 7.), viscous_mean_3_7 = C.bin_average(t, v, 3., 7.),
        steps = info["steps"]))
end
CSV.write(joinpath(RESULTS, "summary.csv"), rows)
foreach(println, rows)
