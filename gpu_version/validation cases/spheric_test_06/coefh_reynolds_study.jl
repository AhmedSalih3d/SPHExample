using SPHExampleGPU, CUDA, CSV, TOML, Statistics
include(joinpath(@__DIR__, "dual_compare.jl"))
const C = Test06DualComparison
const S = C.SphericTest06
const TAG = "coefh_reynolds"
const WORK = joinpath(@__DIR__, "runs", TAG)
const OUT = joinpath(@__DIR__, "results", TAG)
const MOTION = S.body_trajectory(motion_kind = :integrated)

for re in (50, 100, 150), background in (0, 1)
    name = "re$(re)_bg$(background)"
    directory = joinpath(OUT, name)
    record = joinpath(directory, "Re$(lpad(re,3,'0'))_n040.toml")
    # Keep the independently reproduced user's Re100 baseline as provenance.
    if re == 100 && background == 0 && !isfile(record)
        mkpath(directory)
        for suffix in (".csv", ".dat", ".toml", "_state.csv")
            cp(joinpath(@__DIR__, "results", "coefh", "Re100_n040" * suffix),
                joinpath(directory, "Re100_n040" * suffix))
        end
    end
    if isfile(record)
        m = TOML.parsefile(record)
        m["reynolds"] == re && m["background_pressure"] == background && m["sound_speed"] == 160 &&
            m["h"] == Float64(Float32(1.2sqrt(2)*0.025)) && m["cfl"] == Float64(Float32(.2)) &&
            m["frames"] == 401 && m["motion_kind"] == "integrated" && m["wall_layers"] == 5 &&
            !m["divergence_cleaning"] && !m["edge_sponge"] || error("Saved native configuration differs")
        continue
    end
    println("Starting Julia $name"); flush(stdout)
    case = S.build_case(; reynolds = re, sound_speed = 160.0, cfl = 0.2,
        kernel_h_factor = 1.2sqrt(2), wall_layers = 5, background_pressure = background,
        save = joinpath(WORK, name))
    case = merge(case, (; motion = PrescribedMotion(3; translation = MOTION), motion_kind = :integrated))
    mkpath(case.metadata.SaveLocation)
    open(joinpath(case.metadata.SaveLocation, "console.log"), "w") do io
        redirect_stdout(io) do
            S.run_case(case; results_dir = directory)
        end
    end
    println("Completed Julia $name"); flush(stdout)
    GC.gc(true); CUDA.reclaim()
end

for re in ("--native-only" in ARGS ? () : (50, 100, 150))
    name = "re$(re)_dual"
    directory = joinpath(OUT, name)
    record = joinpath(directory, "dual.toml")
    if isfile(record)
        m = TOML.parsefile(record)
        m["reynolds"] == re && m["coefh"] == 1.2 && m["sound_speed"] == 160 &&
            m["cfl"] == .2 && m["background_pressure"] == 0 && m["wall_layers"] == 5 &&
            m["frames"] == 401 && m["particles"] == 86100 || error("Saved Dual configuration differs")
        continue
    end
    println("Starting DualSPHysics Re=$re"); flush(stdout)
    prefix = C.prepare_dual(; reynolds = re, sound_speed = 160, motion_kind = :integrated,
        coefh = 1.2, wall_layers = 5, trajectory = MOTION, tag = "coefh_reynolds_manual_motion")
    C.verify_geometry(prefix, 40; coefh = 1.2, wall_layers = 5)
    run_directory = joinpath(WORK, name)
    info = C.dual_run(prefix, run_directory; reynolds = re)
    info.particles == 86100 && info.frames == 401 || error("Incomplete Dual run")
    C.extract_dual(run_directory, directory)
    # Independent final-frame geometry/impermeability diagnostic.
    final = C.dual_final_state(run_directory)
    fluid = filter(row -> row.Type == 3, final)
    body = filter(row -> row.Type == 1, final)
    length(fluid) == 78400 && length(body) == 1600 || error("Final Dual counts differ")
    x(row) = Float64(row[Symbol("Pos.x [m]")])
    z(row) = Float64(row[Symbol("Pos.z [m]")])
    center_x, center_z = mean(x, body), mean(z, body)
    clearance(row) = max(abs(x(row)-center_x)-.5, abs(z(row)-center_z)-.5)
    final_clearance = minimum(clearance, fluid)
    final_time = Float64(last(CSV.File(joinpath(directory, "dual.csv")).time))
    expected_center = 1.5 + first(trajectory_state(MOTION, final_time))[1]
    C.save_info(record, merge(info, (; reynolds = re, sound_speed = 160., coefh = 1.2,
        background_pressure = 0., cfl = .2, wall_layers = 5, fluid_particles = length(fluid),
        square_particles = length(body), motion = "integrated, linear position samples every 0.0001 s",
        final_inside = count(row -> clearance(row) < 0, fluid), final_clearance,
        final_motion_error = abs(center_x-expected_center), final_time,
        final_center_x = center_x, final_center_z = center_z,
        final_nominal_time_motion_error = abs(center_x-(1.5 + first(trajectory_state(MOTION, 8.0))[1])),
        definition_prefix = prefix)))
    println("Completed DualSPHysics Re=$re"); flush(stdout)
    GC.gc(true); CUDA.reclaim()
end

rows = NamedTuple[]
for re in (50, 100, 150), variant in ("--native-only" in ARGS ? ("bg0", "bg1") : ("bg0", "bg1", "dual"))
    directory = joinpath(OUT, "re$(re)_$(variant)")
    stem = variant == "dual" ? "dual" : "Re$(lpad(re,3,'0'))_n040"
    d = CSV.File(joinpath(directory, stem * ".csv"))
    m = TOML.parsefile(joinpath(directory, stem * ".toml"))
    if variant == "dual"
        if !haskey(m, "final_time")
            body = filter(row -> row.Type == 1, C.dual_final_state(joinpath(WORK, "re$(re)_dual")))
            center_x = mean(row -> Float64(row[Symbol("Pos.x [m]")]), body)
            m["final_nominal_time_motion_error"] = m["final_motion_error"]
            m["final_time"] = Float64(last(d.time))
            m["final_center_x"] = center_x
            m["final_motion_error"] = abs(center_x-(1.5 + first(trajectory_state(MOTION, m["final_time"]))[1]))
            C.save_info(joinpath(directory, "dual.toml"), m)
        end
        cp(m["definition_prefix"] * ".xml", joinpath(directory, "Case.xml"); force = true)
        cp(joinpath(WORK, "re$(re)_dual", "Run.csv"), joinpath(directory, "Run.csv"); force = true)
    end
    reference = S.read_reference(joinpath(S.REFERENCE_DIR, "Force_Re$(lpad(re,3,'0')).dat"))
    metrics, _ = C.drag_metrics(Float64.(d.time), Float64.(d.cd_pressure), Float64.(d.cd_viscous), reference)
    push!(rows, (; reynolds = re, variant, pressure_error = metrics["pressure_raw_nrmse"],
        viscous_error = metrics["viscous_raw_nrmse"], pressure_noise = metrics["pressure_highpass_rms"],
        pressure_mean_3_7 = S.bin_average(Float64.(d.time), Float64.(d.cd_pressure), 3., 7.),
        viscous_mean_3_7 = S.bin_average(Float64.(d.time), Float64.(d.cd_viscous), 3., 7.),
        steps = m["steps"], frames = m["frames"]))
end
CSV.write(joinpath(OUT, "--native-only" in ARGS ? "native_summary.csv" : "summary.csv"), rows)
foreach(println, rows)
