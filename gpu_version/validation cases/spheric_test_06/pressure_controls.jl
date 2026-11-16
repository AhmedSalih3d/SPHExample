module Test06PressureControls

using SPHExampleGPU, CUDA, CSV, TOML, Statistics
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06
include(joinpath(@__DIR__, "noise_metrics.jl"))

const CONTROLS = [
    (; name = "baseline", background = 0.0, delay = 0.0, cleaning = false, sponge = false),
    (; name = "background", background = 101325.0, delay = 0.0, cleaning = false, sponge = false),
    (; name = "hold", background = 0.0, delay = 4.0, cleaning = false, sponge = false),
    (; name = "background_hold", background = 101325.0, delay = 4.0, cleaning = false, sponge = false),
    (; name = "cleaning", background = 0.0, delay = 0.0, cleaning = true, sponge = false),
    (; name = "sponge", background = 0.0, delay = 0.0, cleaning = false, sponge = true),
    (; name = "background_cleaning", background = 101325.0, delay = 0.0, cleaning = true, sponge = false),
    (; name = "combined", background = 101325.0, delay = 4.0, cleaning = true, sponge = true),
]

function configuration(control, save; dx = 0.025, duration = 8.0 + control.delay)
    return build_case(; dx, duration, interval = 0.02, save, cfl = 0.1,
        background_pressure = control.background, start_delay = control.delay,
        damping = control.sponge ? SphericTest06.edge_sponge(; width = dx, rate = 100.0) : nothing,
        divergence_cleaning = control.cleaning ? DivergenceCleaning(kappa = 10, courant = 0.1) : nothing)
end

function analyze_control(control, directory)
    data = CSV.File(joinpath(directory, "Re100_n040.csv"))
    state = CSV.File(joinpath(directory, "Re100_n040_state.csv"))
    info = TOML.parsefile(joinpath(directory, "Re100_n040.toml"))
    time = Float64.(data.motion_time)
    pressure, viscous = Float64.(data.cd_pressure), Float64.(data.cd_viscous)
    reference = read_reference(joinpath(REFERENCE_DIR, "Force_Re100.dat"))
    metrics, spectra = drag_metrics(time, pressure, viscous, reference)
    CSV.write(joinpath(directory, "spectra.csv"), spectra)
    held = findall(t -> 0.0 <= t < control.delay, Float64.(state.time))
    return (; control = control.name, background_pressure = control.background, start_delay = control.delay,
        cleaning = control.cleaning, sponge_width = control.sponge ? 0.025 : 0.0,
        sponge_rate = control.sponge ? 100.0 : 0.0, cfl = 0.1,
        pressure_noise_rms = metrics["pressure_highpass_rms"],
        pressure_raw_nrmse = metrics["pressure_raw_nrmse"],
        viscous_raw_nrmse = metrics["viscous_raw_nrmse"],
        mean_pressure_drag = bin_average(time, pressure, 2.0, 6.5),
        mean_viscous_drag = bin_average(time, viscous, 2.0, 6.5),
        held_max_speed = isempty(held) ? missing : maximum(state.max_speed[held]),
        held_final_pressure_std = isempty(held) ? missing : state.pressure_std[last(held)],
        minimum_pressure = minimum(state.pressure_min),
        max_density_departure = maximum(state.density_relative_max),
        minimum_clearance = minimum(state.minimum_clearance),
        motion_error_max = info["motion_error_max"], mach_max = info["mach_max"],
        steps = info["steps"], frames = info["frames"], wall_seconds = info["wall_seconds"])
end

function study(names = getproperty.(CONTROLS, :name))
    all(name -> name in getproperty.(CONTROLS, :name), names) ||
        throw(ArgumentError("Unknown control; choose $(join(getproperty.(CONTROLS, :name), ", "))"))
    results = mkpath(joinpath(@__DIR__, "results", "pressure_controls"))
    work = joinpath(@__DIR__, "runs", "pressure_controls")
    rows = NamedTuple[]
    for control in CONTROLS
        directory = joinpath(results, control.name)
        record = joinpath(directory, "Re100_n040.toml")
        if control.name in names && !isfile(record)
            println("Pressure control: ", control.name)
            flush(stdout)
            case = configuration(control, joinpath(work, control.name))
            push!(case.metadata.OutputVariables, "Pressure")
            run_case(case; results_dir = directory)
            SphericTest06.write_force_plot_state(case; results_dir = directory,
                output_dir = joinpath(work, control.name, "paraview_force_plot"))
            GC.gc(true)
            CUDA.reclaim()
        end
        if isfile(record)
            info = TOML.parsefile(record)
            info["background_pressure"] == control.background && info["start_delay"] == control.delay &&
                info["cfl"] ≈ 0.1 && info["divergence_cleaning"] == control.cleaning &&
                info["edge_sponge"] == control.sponge && info["dx"] == 0.025 &&
                info["duration"] == 8.0 + control.delay ||
                error("Saved configuration does not match $(control.name); use a separate results directory")
            push!(rows, analyze_control(control, directory))
            CSV.write(joinpath(results, "summary.csv"), rows)
        end
    end
    return rows
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    Test06PressureControls.study(isempty(ARGS) ? getproperty.(Test06PressureControls.CONTROLS, :name) : ARGS)
end
