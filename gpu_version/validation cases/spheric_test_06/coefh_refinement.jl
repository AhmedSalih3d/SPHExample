module CoefhRefinement

using SPHExampleGPU, CUDA, CSV, TOML, SHA, LinearAlgebra
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06
include(joinpath(@__DIR__, "noise_metrics.jl"))

const OUT = joinpath(@__DIR__, "results", "coefh_refinement")
const WORK = joinpath(@__DIR__, "runs", "coefh_refinement")
const REYNOLDS = (100, 50, 150)
const RESOLUTIONS = (20, 40, 80)
stem(re, n) = "Re$(lpad(re, 3, '0'))_n$(lpad(n, 3, '0'))"

function configuration(re, n; cfl = 0.2)
    case = build_case(; reynolds = re, dx = 1 / n, sound_speed = 160.0, cfl,
        kernel_h_factor = 1.2sqrt(2), save = joinpath(WORK, "re$re", "n$n"))
    return merge(case, (; motion = PrescribedMotion(3; translation = body_trajectory(motion_kind = :integrated)),
        motion_kind = :integrated))
end

function validate_record(m, re, n; cfl = 0.2)
    expected_fixed = (10n + 10) * (5n + 10) - 50n^2
    expected = Dict(
        "reynolds" => re, "dx" => 1 / n, "sound_speed" => 160.0,
        "h" => Float64(Float32(1.2sqrt(2) / n)), "cfl" => Float64(Float32(cfl)),
        "nu" => Float64(Float32(1 / re)), "rho0" => 1000.0, "gravity" => 0.0,
        "density_diffusion" => Float64(Float32(0.1)), "background_pressure" => 0.0,
        "duration" => 8.0, "output_interval" => 0.02, "frames" => 401, "wall_layers" => 5,
        "motion_kind" => "integrated", "boundary_mode" => "NoMDBC",
        "shifting" => "NoShifting", "kernel" => "WendlandC2",
        "time_stepping" => "SymplecticTimeStepping", "viscosity_model" => "Laminar",
        "density_diffusion_model" => "ZeroGravityLinearDensityDiffusion",
        "divergence_cleaning" => false, "edge_sponge" => false, "start_delay" => 0.0,
        "domain_padding" => 0.0, "frozen_particles" => 0,
        "fluid_particles" => 49n^2, "square_particles" => n^2, "fixed_particles" => expected_fixed,
        "tank_origin" => [0.0, 0.0], "tank_size" => [10.0, 5.0],
        "body_center" => [1.5, 2.5], "body_size" => [1.0, 1.0],
        "reference_length" => 1.0, "reference_speed" => 1.0, "span" => 1.0,
    )
    for (key, value) in expected
        actual = get(m, key, nothing)
        type_key = key in ("boundary_mode", "shifting", "kernel", "time_stepping",
            "viscosity_model", "density_diffusion_model")
        same_type = type_key && actual isa AbstractString &&
            startswith(actual, "SPHExampleGPU.") && last(split(actual, '.')) == value
        (actual == value || same_type) ||
            error("Saved Re=$re n=$n differs at $key; use a separate study directory")
    end
    return m
end

function import_control(source, destination, name, re, n; cfl = 0.2)
    validate_record(TOML.parsefile(joinpath(source, name * ".toml")), re, n; cfl)
    mkpath(destination)
    for suffix in (".csv", ".dat", ".toml", "_state.csv")
        src, dest = joinpath(source, name * suffix), joinpath(destination, name * suffix)
        if isfile(dest)
            sha256(read(src)) == sha256(read(dest)) || error("Imported control changed: $dest")
        else
            cp(src, dest)
        end
    end
end

function history(directory, re, n)
    name = stem(re, n)
    d = CSV.File(joinpath(directory, name * ".csv"))
    s = CSV.File(joinpath(directory, name * "_state.csv"))
    length(d) == length(s) == 401 || error("Incomplete history: $directory/$name")
    all(k -> all(isfinite, getproperty(d, k)), propertynames(d)) || error("Nonfinite forces")
    all(k -> all(isfinite, getproperty(s, k)), propertynames(s)) || error("Nonfinite state")
    t = Float64.(d.time)
    first(t) == 0 && last(t) == 8 && all(>(0), diff(t)) || error("Invalid force times")
    t == Float64.(s.time) || error("State times differ")
    all(==(49n^2), s.fluid_particles) && all(==(n^2), s.square_particles) || error("Particle loss")
    return (; d, s, t, p = Float64.(d.cd_pressure), v = Float64.(d.cd_viscous))
end

function relative_difference(t, values, rt, target; residual = true)
    first(t) <= 0.1 && last(t) >= 8 && first(rt) <= 0.1 && last(rt) >= 8 ||
        error("Force comparison requires coverage of 0.1-8 s")
    knots = sort!(unique!([0.1; t[(t .> 0.1) .& (t .< 8)];
        rt[(rt .> 0.1) .& (rt .< 8)]; 8.0]))
    predicted = [SphericTest06.interpolate(t, values, x) for x in knots]
    reference = [SphericTest06.interpolate(rt, target, x) for x in knots]
    integral(z) = sum(diff(knots) .* (z[1:end-1].^2 + z[1:end-1].*z[2:end] + z[2:end].^2)) / 3
    return sqrt(integral(residual ? predicted - reference : predicted) / integral(reference))
end

function analyze()
    rows, differences = NamedTuple[], NamedTuple[]
    for re in REYNOLDS
        directory = joinpath(OUT, "re$re")
        reference = read_reference(joinpath(REFERENCE_DIR, "Force_Re$(lpad(re, 3, '0')).dat"))
        for n in RESOLUTIONS
            record = joinpath(directory, stem(re, n) * ".toml")
            isfile(record) || continue
            m = validate_record(TOML.parsefile(record), re, n)
            h = history(directory, re, n)
            metrics, _ = drag_metrics(h.t, h.p, h.v, reference)
            for (component, values, target) in (
                    ("pressure", h.p, reference[:, 2]), ("viscous", h.v, reference[:, 3]),
                    ("total", h.p + h.v, reference[:, 2] + reference[:, 3]))
                push!(rows, (; reynolds = re, n, dx = m["dx"], component,
                    raw_nrmse = relative_difference(h.t, values, reference[:, 1], target),
                    peak = maximum(values), peak_time = h.t[argmax(values)],
                    reference_peak = maximum(target), reference_peak_time = reference[argmax(target), 1],
                    mean_3_7 = bin_average(h.t, values, 3.0, 7.0),
                    mean_bias_3_7 = bin_average(h.t, values, 3.0, 7.0) /
                        bin_average(reference[:, 1], target, 3.0, 7.0) - 1,
                    pressure_noise = metrics["pressure_highpass_rms"],
                    penetration_mm = 1000max(0, -minimum(h.s.minimum_clearance)),
                    penetration_dp = max(0, -minimum(h.s.minimum_clearance)) / m["dx"],
                    density_relative_max = maximum(h.s.density_relative_max),
                    mach_max = m["mach_max"], motion_error = maximum(h.s.motion_error),
                    particles = m["fixed_particles"] + m["fluid_particles"] + m["square_particles"],
                    steps = m["steps"], frames = m["frames"]))
            end
        end
        for (coarse, fine) in ((20, 40), (40, 80))
            all(n -> isfile(joinpath(directory, stem(re, n) * ".toml")), (coarse, fine)) || continue
            a, b = history(directory, re, coarse), history(directory, re, fine)
            for (component, av, bv, target) in (
                    ("pressure", a.p, b.p, reference[:, 2]), ("viscous", a.v, b.v, reference[:, 3]),
                    ("total", a.p + a.v, b.p + b.v, reference[:, 2] + reference[:, 3]))
                difference = av - bv
                a.t == b.t || error("Refinement output times differ")
                push!(differences, (; reynolds = re, coarse, fine, component,
                    relative_to_reference = relative_difference(a.t, difference,
                        reference[:, 1], target; residual = false)))
            end
        end
    end
    isempty(rows) || CSV.write(joinpath(OUT, "summary.csv"), rows)
    isempty(differences) || CSV.write(joinpath(OUT, "resolution_changes.csv"), differences)
    return rows
end

function study(reynolds = REYNOLDS)
    all(re -> re in REYNOLDS, reynolds) || error("Choose Re=50,100,150")
    verify_reference()
    mkpath(OUT)
    for re in reynolds
        directory = joinpath(OUT, "re$re")
        source = joinpath(@__DIR__, "results", "coefh_reynolds", "re$(re)_bg0")
        import_control(source, directory, stem(re, 40), re, 40)
        for n in (20, 80)
            record = joinpath(directory, stem(re, n) * ".toml")
            if isfile(record)
                validate_record(TOML.parsefile(record), re, n)
                history(directory, re, n)
                continue
            end
            case = configuration(re, n)
            println("Starting raw refinement Re=$re n=$n"); flush(stdout)
            mkpath(case.metadata.SaveLocation)
            open(joinpath(case.metadata.SaveLocation, "console.log"), "w") do io
                redirect_stdout(io) do
                    run_case(case; results_dir = directory)
                end
            end
            println("Completed raw refinement Re=$re n=$n"); flush(stdout)
            GC.gc(true); CUDA.reclaim()
        end
        analyze()
    end
    control_source = joinpath(@__DIR__, "results", "coefh_validation", "coef12_cfl01")
    import_control(control_source, joinpath(OUT, "cfl01"), stem(100, 40), 100, 40; cfl = 0.1)
    provenance = NamedTuple[]
    for re in REYNOLDS, n in RESOLUTIONS
        path = joinpath(OUT, "re$re", stem(re, n) * ".csv")
        isfile(path) || continue
        source = n == 40 ? joinpath(@__DIR__, "results", "coefh_reynolds", "re$(re)_bg0",
            stem(re, n) * ".csv") : path
        push!(provenance, (; reynolds = re, n, cfl = 0.2, reused = n == 40,
            source = relpath(source, @__DIR__), sha256 = bytes2hex(sha256(read(source)))))
    end
    push!(provenance, (; reynolds = 100, n = 40, cfl = 0.1, reused = true,
        source = relpath(joinpath(control_source, stem(100, 40) * ".csv"), @__DIR__),
        sha256 = bytes2hex(sha256(read(joinpath(control_source, stem(100, 40) * ".csv"))))))
    CSV.write(joinpath(OUT, "provenance.csv"), provenance)
    return analyze()
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    CoefhRefinement.study(isempty(ARGS) ? CoefhRefinement.REYNOLDS : parse.(Int, ARGS))
end
