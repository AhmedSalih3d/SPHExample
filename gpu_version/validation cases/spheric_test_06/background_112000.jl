using SPHExampleGPU, CUDA, CSV, TOML, HDF5, StaticArrays
include(joinpath(@__DIR__, "dual_compare.jl"))
const C = Test06DualComparison
const S = C.SphericTest06
const OUT = joinpath(@__DIR__, "results", "background_112000")
const WORK = joinpath(@__DIR__, "runs", "background_112000")
const MOTION = S.body_trajectory(motion_kind = :integrated)
const SETTINGS = ((; name = "offset112000_c160", speed = 160., background = 112000.),
                  (; name = "stiffness112000_c28", speed = 28., background = 0.))

for s in SETTINGS
    directory = joinpath(OUT, s.name)
    record = joinpath(directory, "Re100_n040.toml")
    if isfile(record)
        m = TOML.parsefile(record)
        m["sound_speed"] == s.speed && m["background_pressure"] == s.background &&
            m["h"] == Float64(Float32(1.2sqrt(2)*.025)) && m["frames"] == 401 || error("Saved parameters differ")
        continue
    end
    println("Starting Julia $(s.name)"); flush(stdout)
    case = S.build_case(; reynolds = 100, sound_speed = s.speed, background_pressure = s.background,
        kernel_h_factor = 1.2sqrt(2), wall_layers = 5, save = joinpath(WORK, s.name))
    case = merge(case, (; motion = PrescribedMotion(3; translation = MOTION), motion_kind = :integrated))
    mkpath(case.metadata.SaveLocation)
    open(joinpath(case.metadata.SaveLocation, "console.log"), "w") do io
        redirect_stdout(io) do
            S.run_case(case; results_dir = directory)
        end
    end
    println("Completed Julia $(s.name)"); flush(stdout)
    GC.gc(true); CUDA.reclaim()
end

directory = joinpath(OUT, "dual_stiffness112000_c28")
record = joinpath(directory, "dual.toml")
if !isfile(record)
    println("Starting DSPH with B=112000 and c0=28"); flush(stdout)
    prefix = C.prepare_dual(; reynolds = 100, sound_speed = 28, coefh = 1.2, wall_layers = 5,
        motion_kind = :integrated, trajectory = MOTION, tag = "background112000_stiffness_control")
    C.verify_geometry(prefix, 40; coefh = 1.2, wall_layers = 5)
    run_directory = joinpath(WORK, "dual_stiffness112000_c28")
    info = C.dual_run(prefix, run_directory; reynolds = 100)
    info.frames == 401 && info.particles == 86100 || error("Incomplete DSPH run")
    C.extract_dual(run_directory, directory)
    cp(prefix * ".xml", joinpath(directory, "Case.xml"))
    cp(joinpath(run_directory, "Run.csv"), joinpath(directory, "Run.csv"))
    C.save_info(record, merge(info, (; sound_speed = 28., eos_stiffness = 112000.,
        background_pressure = 0., reynolds = 100, coefh = 1.2, cfl = .2)))
    println("Completed DSPH with B=112000"); flush(stdout)
else
    m = TOML.parsefile(record)
    m["sound_speed"] == 28 && m["eos_stiffness"] == 112000 && m["background_pressure"] == 0 &&
        m["reynolds"] == 100 && m["coefh"] == 1.2 && m["cfl"] == .2 &&
        m["frames"] == 401 && m["particles"] == 86100 || error("Saved DSPH parameters differ")
end

reference = S.read_reference(joinpath(S.REFERENCE_DIR, "Force_Re100.dat"))
rows = NamedTuple[]
for (name, dir, stem, speed, background) in (
        ("julia_c160_bg0", joinpath(@__DIR__, "results", "coefh"), "Re100_n040", 160., 0.),
        ("julia_c160_bg112000", joinpath(OUT, "offset112000_c160"), "Re100_n040", 160., 112000.),
        ("julia_c28_B112000", joinpath(OUT, "stiffness112000_c28"), "Re100_n040", 28., 0.),
        ("dsph_c160_bg0", joinpath(@__DIR__, "results", "coefh_reynolds", "re100_dual"), "dual", 160., 0.),
        ("dsph_c28_B112000", joinpath(OUT, "dual_stiffness112000_c28"), "dual", 28., 0.))
    d = CSV.File(joinpath(dir, stem * ".csv"))
    metrics, _ = C.drag_metrics(Float64.(d.time), Float64.(d.cd_pressure), Float64.(d.cd_viscous), reference)
    push!(rows, (; name, sound_speed = speed, background_pressure = background, eos_stiffness = 1000speed^2/7,
        pressure_error = metrics["pressure_raw_nrmse"], viscous_error = metrics["viscous_raw_nrmse"],
        pressure_noise = metrics["pressure_highpass_rms"]))
end
CSV.write(joinpath(OUT, "summary.csv"), rows)
foreach(println, rows)

# Separate the direct constant-pressure contribution on the same saved state.
# This is a diagnostic decomposition, not a rerun or a corrected force curve.
directory = joinpath(OUT, "offset112000_c160")
m = TOML.parsefile(joinpath(directory, "Re100_n040.toml"))
saved = CSV.File(joinpath(directory, "Re100_n040.csv"))
k32 = SPHKernelInstance{2,Float32}(WendlandC2(); h = Float32(m["h"]))
kernel = SPHKernelInstance{2,Float64}(WendlandC2(); h = Float64(k32.h), k = Float64(k32.k),
    αD = Float64(k32.αD), η² = Float64(k32.η²))
split_rows = NamedTuple[]
h5open(joinpath(WORK, "offset112000_c160", "MovingSquare.vtkhdf"), "r") do file
    root = haskey(file, "VTKHDF/Particles") ? file["VTKHDF/Particles"] : file["VTKHDF"]
    times, offsets, counts = read(root["Steps/Values"]), read(root["Steps/PointOffsets"]), read(root["NumberOfPoints"])
    for time in (0., 4., 8.)
        frame = findfirst(==(time), times)
        columns = offsets[frame]+1:offsets[frame]+counts[frame]
        position = [SVector{2,Float64}(p[1],p[2]) for p in eachcol(root["Points"][:,columns])]
        velocity = [SVector{2,Float64}(v[1],v[2]) for v in eachcol(root["PointData/Velocity"][:,columns])]
        density = Float64.(root["PointData/Density"][columns])
        types, markers = root["PointData/Type"][columns], root["PointData/GroupMarker"][columns]
        parts = map((0.,112000.)) do background
            constants = SimulationConstants{Float64}(2; dx = .025, ρ₀ = 1000., c₀ = 160., g = 0.,
                ν₀ = m["nu"], BackgroundPressure = background)
            S.frame_loads(position, velocity, density, types, markers, kernel, constants)
        end
        actual = only(filter(r -> r.time == time, collect(saved))).cd_pressure
        isapprox(parts[2].cd_pressure, actual; atol = 1e-9, rtol = 1e-9) || error("Pressure reconstruction mismatch")
        push!(split_rows, (; time, actual_pressure_cd = actual, gauge_pressure_part = parts[1].cd_pressure,
            constant_offset_part = parts[2].cd_pressure-parts[1].cd_pressure))
    end
end
CSV.write(joinpath(OUT, "offset_force_decomposition.csv"), split_rows)
foreach(println, split_rows)
