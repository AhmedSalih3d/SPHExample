# Run from gpu_version/: julia --project=. -t 2,0 benchmark/zipline.jl [output]
# Optional reference: set DUALSPHYSICS_ROOT and DUALSPHYSICS_ZIPLINE_CASE to an
# installation and a GenCase-generated CaseZipLine prefix (without extension).
include("zipline_case.jl")
using CSV, StaticArrays, LinearAlgebra, Serialization, TOML

function zipline_residuals(directory, case)
    rows = collect(CSV.File(joinpath(directory, "ZipLine3D_Floating.csv")))
    times = sort!(unique(row.Time for row in rows))
    mass = [length(g.Particles) * 0.000064 for g in case.geometry if g.Type == Floating]
    axis = normalize(SVector(11., 0., -8.4))
    gravity = SVector(0., 0., -9.81)
    translation, rotation, momentum, trajectory = 0., 0., 0., 0.
    initial_center = zero(axis)
    for time in times
        frame = sort(filter(row -> row.Time == time, rows); by = row -> row.Body)
        state = (;
            center = [SVector(ntuple(d -> row[Symbol("Center:", d - 1)], 3)) for row in frame],
            velocity = [SVector(ntuple(d -> row[Symbol("Velocity:", d - 1)], 3)) for row in frame],
            orientation = [SVector(ntuple(d -> row[Symbol("Orientation:", d - 1)], 4)) for row in frame])
        errors = rigid_joint_errors(case.dynamics, case.geometry, state)
        translation = max(translation, maximum(e.translation_error for e in errors))
        rotation = max(rotation, maximum(e.rotation_error for e in errors))
        center = sum(mass .* state.center) / sum(mass)
        time == 0 && (initial_center = center)
        if time <= 1.5 # before contact with the fixed container
            velocity = sum(mass .* state.velocity) / sum(mass)
            momentum = max(momentum, abs(dot(velocity - gravity * time, axis)))
            trajectory = max(trajectory, abs(dot(center - initial_center - gravity * time^2 / 2, axis)))
        end
    end
    return (; max_joint_translation_m = translation, max_hinge_angle_rad = rotation,
        preimpact_parallel_velocity_error_m_s = momentum, preimpact_parallel_position_error_m = trajectory)
end

function measure_zipline(directory; settings...)
    mkpath(directory)
    result = open(joinpath(directory, "console.log"), "w") do io
        redirect_stdout(io) do
            run_zipline_3d(; save_location = directory, visualize = false, open_log_file = false, settings...)
        end
    end
    case = prepare_zipline_3d(; settings...)
    residuals = zipline_residuals(directory, case)
    serialize(joinpath(directory, "result.jls"), result)
    metrics = Dict{String,Any}(string(k) => v for (k,v) in pairs(residuals))
    for key in (:steps, :simulated_time, :wall_time_seconds, :milliseconds_per_step)
        metrics[string(key)] = getproperty(result.performance, key)
    end
    for key in (:particle_count, :output_frames, :cell_list_rebuilds, :step_state_readbacks, :captured_graphs, :frozen_particles)
        metrics[string(key)] = getproperty(result.diagnostics, key)
    end
    metrics["iterations"] = case.dynamics.iterations
    metrics["max_timestep"] = something(case.dynamics.max_timestep, 0.)
    open(joinpath(directory, "metrics.toml"), "w") do io
        TOML.print(io, metrics)
    end
    println(basename(directory), " ", metrics)
    flush(stdout)
    return result, case
end

function measure_dual_zipline(directory, installation, case_prefix)
    mkpath(directory)
    executable = joinpath(installation, "bin", "windows", "DualSPHysics5.4_win64.exe")
    command = Cmd(`$executable -gpu $case_prefix $directory -svres`; dir = dirname(case_prefix))
    open(joinpath(directory, "console.log"), "w") do io
        run(pipeline(command; stdout = io, stderr = io))
    end
    row = only(CSV.File(joinpath(directory, "Run.csv"); delim = ';'))
    println(basename(directory), " ", (; steps = row.Steps, simulated_time = row.PhysicalTime,
        solver_seconds = row.TSimul, total_seconds = row.TTotal, frames = row.PartFiles))
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    directory = abspath(isempty(ARGS) ? joinpath(@__DIR__, "..", "output", "zipline_benchmark") : first(ARGS))
    reference = get(ENV, "DUALSPHYSICS_ZIPLINE_CASE", "")
    installation = get(ENV, "DUALSPHYSICS_ROOT", "")
    measure_zipline(joinpath(directory, "warm"); simulation_time = 0.02)
    # Unprofiled whole-solver runs. Reverse solver order on the second pair.
    measure_zipline(joinpath(directory, "native_1"))
    if !isempty(reference)
        measure_dual_zipline(joinpath(directory, "dual_1"), installation, reference)
        measure_dual_zipline(joinpath(directory, "dual_2"), installation, reference)
    end
    result, case = measure_zipline(joinpath(directory, "native_2"))
    measure_zipline(joinpath(directory, "half_dt"); max_timestep = 0.001)
    measure_zipline(joinpath(directory, "double_iterations"); iterations = 64)
    Base.invokelatest(case.save_surfaces, joinpath(directory, "native_2"), case.geometry)
end
