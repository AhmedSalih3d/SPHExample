# Isolate the startup failure before spending time on full-history candidates.
using SPHExampleGPU, CUDA, CSV, TOML
include(joinpath(@__DIR__, "mdbc_validation.jl"))
const M = MDBCValidation
const C = M.C
const TAG = "cleaning_startup_c160"
const WORK = joinpath(@__DIR__, "runs", TAG)
const RESULTS = joinpath(@__DIR__, "results", TAG)
const SETTINGS = (
    (; name = "dbc_off_small", boundary = NoMDBC, shifting = NoShifting, kappa = 0),
    (; name = "dbc_k10_small", boundary = NoMDBC, shifting = NoShifting, kappa = 10),
    (; name = "dbc_k160_small", boundary = NoMDBC, shifting = NoShifting, kappa = 160),
    (; name = "shifted_k10_small", boundary = UpdatedMDBC, shifting = PlanarShifting, kappa = 10),
    (; name = "shifted_k160_small", boundary = UpdatedMDBC, shifting = PlanarShifting, kappa = 160),
    (; name = "dbc_off_tiny", boundary = NoMDBC, shifting = NoShifting, kappa = 0),
    (; name = "dbc_delta2", boundary = NoMDBC, shifting = NoShifting, kappa = 0),
    (; name = "dbc_delta4", boundary = NoMDBC, shifting = NoShifting, kappa = 0),
)

for s in SETTINGS
    directory = joinpath(RESULTS, s.name)
    cfl = startswith(s.name, "dbc_delta") ? 0.2 : s.name == "dbc_off_tiny" ? 0.025 : 0.05
    delta = s.name == "dbc_delta2" ? 0.2 : s.name == "dbc_delta4" ? 0.4 : 0.1
    if isfile(joinpath(directory, "native.toml"))
        info = TOML.parsefile(joinpath(directory, "native.toml"))
        info["sound_speed"] == 160.0 && info["cfl"] == Float64(Float32(cfl)) &&
            info["frames"] == 81 && info["boundary_mode"] == string(nameof(s.boundary)) &&
            info["shifting_mode"] == string(nameof(s.shifting)) &&
            info["divergence_cleaning"] == (s.kappa != 0) &&
            info["cleaning_kappa"] == s.kappa || error("Saved startup configuration differs")
        get(info, "density_diffusion_coefficient", Float64(Float32(0.1))) == Float64(Float32(delta)) || error("Saved diffusion coefficient differs")
        s.kappa == 0 || (info["cleaning_courant"] == 0.025 && info["cleaning_sigma"] == 1.0 &&
            info["cleaning_absolute_pressure"]) || error("Saved cleaning parameters differ")
        continue
    end
    println("Starting $(s.name)"); flush(stdout)
    case = M.build_mdbc_case(; sound_speed = 160.0, cfl, density_diffusion = delta, duration = 1.6,
        boundary_mode = s.boundary, shifting_mode = s.shifting, motion_kind = :integrated,
        legacy_timestep = true, save = joinpath(WORK, s.name))
    cleaning = s.kappa == 0 ? nothing : DivergenceCleaning(kappa = s.kappa, courant = 0.025)
    M.run_mdbc_case(case; results_dir = directory, cleaning)
    println("Completed $(s.name)"); flush(stdout)
    GC.gc(true); CUDA.reclaim()
end

rows = NamedTuple[]
for s in SETTINGS
    directory = joinpath(RESULTS, s.name)
    d = CSV.File(joinpath(directory, "native.csv"))
    t, p, v = Float64.(d.time), Float64.(d.cd_pressure), Float64.(d.cd_viscous)
    info = TOML.parsefile(joinpath(directory, "native.toml"))
    info["frames"] == 81 && info["particles"] == 84864 || error("Incomplete startup control")
    push!(rows, (; name = s.name, kappa = s.kappa, cfl = info["cfl"], pressure_min = minimum(p), pressure_max = maximum(p),
        minimum_clearance = info["minimum_clearance"], density_relative_max = info["density_relative_max"],
        steps = info["steps"]))
end
CSV.write(joinpath(RESULTS, "summary.csv"), rows)
foreach(println, rows)

if "--full" in ARGS
    directory = joinpath(RESULTS, "full_delta2")
    if !isfile(joinpath(directory, "native.toml"))
        println("Starting full_delta2"); flush(stdout)
        case = M.build_mdbc_case(; sound_speed = 160.0, cfl = 0.2, density_diffusion = 0.2,
            boundary_mode = NoMDBC, shifting_mode = NoShifting, motion_kind = :integrated,
            legacy_timestep = true, save = joinpath(WORK, "full_delta2"))
        M.run_mdbc_case(case; results_dir = directory)
        println("Completed full_delta2"); flush(stdout)
    end
    info = TOML.parsefile(joinpath(directory, "native.toml"))
    info["frames"] == 401 && info["particles"] == 84864 && info["sound_speed"] == 160.0 &&
        info["cfl"] == Float64(Float32(0.2)) && info["density_diffusion_coefficient"] == Float64(Float32(0.2)) &&
        !info["divergence_cleaning"] && info["boundary_mode"] == "NoMDBC" || error("Full-run configuration differs")
    data = CSV.File(joinpath(directory, "native.csv"))
    reference = C.read_reference(joinpath(C.REFERENCE_DIR, "Force_Re100.dat"))
    metrics, _ = C.drag_metrics(Float64.(data.time), Float64.(data.cd_pressure), Float64.(data.cd_viscous), reference)
    C.save_info(joinpath(RESULTS, "full_delta2_metrics.toml"), metrics)
    println(metrics)
end
