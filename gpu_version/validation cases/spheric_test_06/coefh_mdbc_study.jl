using SPHExampleGPU, CUDA, CSV, TOML
include(joinpath(@__DIR__, "mdbc_validation.jl"))
const M = MDBCValidation
const C = M.C
const S = C.SphericTest06
const OUT = joinpath(@__DIR__, "results", "coefh_mdbc")
const WORK = joinpath(@__DIR__, "runs", "coefh_mdbc")
const MOTION = S.body_trajectory(motion_kind = :integrated)
const SETTINGS = [(; reynolds = 100, name = "re100_dbc_control", boundary = NoMDBC);
    [(; reynolds = re, name = "re$(re)_$(name)", boundary)
        for re in (50,100,150) for (name,boundary) in (("simple",SimpleMDBC),("updated",UpdatedMDBC))]]

for s in SETTINGS
    directory = joinpath(OUT, s.name)
    record = joinpath(directory, "native.toml")
    if isfile(record)
        m = TOML.parsefile(record)
        m["reynolds"] == s.reynolds && m["sound_speed"] == 160 && m["h"] == Float64(Float32(1.2sqrt(2)*.025)) &&
            m["wall_layers"] == 5 && m["boundary_mode"] == string(nameof(s.boundary)) &&
            m["shifting_mode"] == "NoShifting" && m["background_pressure"] == 0 &&
            m["cfl"] == Float64(Float32(.2)) && m["frames"] == 401 && m["particles"] == 86100 &&
            !m["divergence_cleaning"] || error("Saved configuration differs")
        continue
    end
    println("Starting $(s.name)"); flush(stdout)
    case = M.build_mdbc_case(; reynolds = s.reynolds, sound_speed = 160., cfl = .2,
        kernel_h_factor = 1.2sqrt(2), wall_layers = 5, boundary_mode = s.boundary,
        shifting_mode = NoShifting, motion_kind = :integrated, legacy_timestep = true,
        save = joinpath(WORK, s.name))
    case = merge(case, (; motion = PrescribedMotion(3; translation = MOTION)))
    M.run_mdbc_case(case; results_dir = directory)
    println("Completed $(s.name)"); flush(stdout)
    GC.gc(true); CUDA.reclaim()
end

baseline = CSV.File(joinpath(@__DIR__, "results", "coefh", "Re100_n040.csv"))
control = CSV.File(joinpath(OUT, "re100_dbc_control", "native.csv"))
for field in (:time,:cd_pressure,:cd_viscous,:cy_pressure,:cy_viscous)
    all(getproperty(baseline,field) .== getproperty(control,field)) || error("DBC control does not reproduce $field")
end
println("DBC control exactly matches the supplied force history.")

rows = NamedTuple[]
for re in (50,100,150), mode in ("dbc","simple","updated")
    directory = mode == "dbc" ? joinpath(@__DIR__, "results", "coefh_reynolds", "re$(re)_bg0") : joinpath(OUT,"re$(re)_$(mode)")
    stem = mode == "dbc" ? "Re$(lpad(re,3,'0'))_n040" : "native"
    d = CSV.File(joinpath(directory, stem*".csv"))
    m = TOML.parsefile(joinpath(directory, stem*".toml"))
    reference = S.read_reference(joinpath(S.REFERENCE_DIR, "Force_Re$(lpad(re,3,'0')).dat"))
    metrics,_ = C.drag_metrics(Float64.(d.time), Float64.(d.cd_pressure), Float64.(d.cd_viscous), reference)
    push!(rows, (; reynolds = re, boundary = mode, pressure_error = metrics["pressure_raw_nrmse"],
        viscous_error = metrics["viscous_raw_nrmse"], pressure_noise = metrics["pressure_highpass_rms"],
        maximum_penetration_mm = 1000max(0,-m["minimum_clearance"]), density_relative_max = m["density_relative_max"],
        pressure_mean_3_7 = S.bin_average(Float64.(d.time),Float64.(d.cd_pressure),3.,7.),
        viscous_mean_3_7 = S.bin_average(Float64.(d.time),Float64.(d.cd_viscous),3.,7.),
        steps = m["steps"], frames = m["frames"]))
end
CSV.write(joinpath(OUT,"summary.csv"),rows)
foreach(println,rows)
