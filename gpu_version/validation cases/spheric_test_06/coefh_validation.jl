using SPHExampleGPU, CUDA, CSV, TOML
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06

const TAG = "coefh_validation"
const WORK = joinpath(@__DIR__, "runs", TAG)
const OUT = joinpath(@__DIR__, "results", TAG)
const SETTINGS = (
    (; name = "coef1_cfl02", coefh = 1.0, cfl = 0.2),
    (; name = "coef12_cfl02", coefh = 1.2, cfl = 0.2),
    (; name = "coef12_matched_dt", coefh = 1.2, cfl = 0.2 / 1.2),
    (; name = "coef12_cfl01", coefh = 1.2, cfl = 0.1),
)

for s in SETTINGS
    directory = joinpath(OUT, s.name)
    record = joinpath(directory, "Re100_n040.toml")
    if isfile(record)
        m = TOML.parsefile(record)
        m["h"] == Float64(Float32(s.coefh * sqrt(2) * 0.025)) &&
            m["cfl"] == Float64(Float32(s.cfl)) && m["wall_layers"] == 5 &&
            m["motion_kind"] == "integrated" && m["frames"] == 401 &&
            !m["divergence_cleaning"] && !m["edge_sponge"] || error("Saved configuration differs")
        continue
    end
    println("Starting $(s.name)"); flush(stdout)
    case = build_case(; sound_speed = 160.0, cfl = s.cfl,
        kernel_h_factor = s.coefh * sqrt(2), wall_layers = 5,
        save = joinpath(WORK, s.name))
    motion = PrescribedMotion(3; translation = body_trajectory(motion_kind = :integrated))
    case = merge(case, (; motion, motion_kind = :integrated))
    mkpath(case.metadata.SaveLocation)
    open(joinpath(case.metadata.SaveLocation, "console.log"), "w") do io
        redirect_stdout(io) do
            run_case(case; results_dir = directory)
        end
    end
    println("Completed $(s.name)"); flush(stdout)
    GC.gc(true); CUDA.reclaim()
end

include(joinpath(@__DIR__, "noise_metrics.jl"))
reference = read_reference(joinpath(REFERENCE_DIR, "Force_Re100.dat"))
analysis_rows = NamedTuple[]
acceptance_rows = NamedTuple[]
for (name, directory) in [("user", joinpath(@__DIR__, "results", "coefh"));
        [(s.name, joinpath(OUT, s.name)) for s in SETTINGS]]
    d = CSV.File(joinpath(directory, "Re100_n040.csv"))
    metrics, _ = drag_metrics(Float64.(d.time), Float64.(d.cd_pressure), Float64.(d.cd_viscous), reference)
    push!(analysis_rows, (; name, pressure_error = metrics["pressure_raw_nrmse"],
        viscous_error = metrics["viscous_raw_nrmse"], pressure_noise = metrics["pressure_highpass_rms"]))
    for row in compare_histories(directory).metrics
        push!(acceptance_rows, merge((; name), row))
    end
    if name == "user"
        for filename in ("binned.csv", "metrics.csv")
            mv(joinpath(directory, filename), joinpath(OUT, "user_" * filename); force = true)
        end
    end
end
CSV.write(joinpath(OUT, "julia_metrics.csv"), analysis_rows)
CSV.write(joinpath(OUT, "acceptance_checks.csv"), acceptance_rows)
foreach(println, analysis_rows)
