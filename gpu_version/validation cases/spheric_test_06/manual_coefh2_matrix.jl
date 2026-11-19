include("case.jl")
using .SphericTest06
using SPHExampleGPU, CUDA, CSV, Printf, TOML

const MODELS = (
    "zero-gravity-linear" => ZeroGravityLinearDensityDiffusion(),
    "zero" => ZeroDensityDiffusion(),
    "fluid-linear" => LinearDensityDiffusion(),
    "complex" => ComplexDensityDiffusion(),
    "fourtakas-inner" => FourtakasDensityDiffusion(),
    "fourtakas-full" => FourtakasDensityDiffusion(full = true),
)
const BOUNDARIES = ("dbc" => NoMDBC, "simple-mdbc" => SimpleMDBC)
const RESOLUTIONS = (20, 40, 80)
const REYNOLDS = (100, 50, 150)
const TAG = "manual_coefh2_matrix"
const RUNS = joinpath(@__DIR__, "runs", TAG)
const RESULTS = joinpath(@__DIR__, "results", TAG)

function verify_run(record, name, reynolds, n, boundary, diffusion)
    info = TOML.parsefile(record)
    expected = (
        "reynolds" => reynolds, "dx" => 1 / n, "duration" => 8.0, "output_interval" => 0.02,
        "sound_speed" => 160.0, "h" => Float64(Float32(2sqrt(2) / n)), "wall_layers" => 7,
        "cfl" => Float64(0.2f0), "boundary_mode" => string(boundary), "motion_kind" => "integrated",
        "shifting" => "PlanarShifting", "frames" => 401, "frozen_particles" => 0,
        "density_diffusion_model" => string(nameof(typeof(diffusion))),
    )
    all(((key, value),) -> info[key] == value, expected) || error("Settings or completion differ for $name")
    history = CSV.File(joinpath(dirname(record), "$name.csv"))
    length(history) == 401 && first(history.time) == 0 && last(history.time) == 8.0 ||
        error("Incomplete force history for $name")
end

function run_member(reynolds, n, wall_name, boundary, model_name, diffusion)
    name = @sprintf("Re%03d_n%03d", reynolds, n)
    group = joinpath(wall_name, model_name)
    save = joinpath(RUNS, group, name)
    results_dir = joinpath(RESULTS, group)
    record = joinpath(results_dir, "$name.toml")
    if !isfile(record)
        mkpath(save)
        println("Running $group/$name")
        flush(stdout)
        case = build_case(; reynolds, dx = 1 / n, sound_speed = 160.0, kernel_h_factor = 2sqrt(2),
            shifting = PlanarShifting, boundary_mode = boundary, motion_kind = :integrated,
            density_diffusion_model = diffusion, save)
        push!(case.metadata.OutputVariables, "Pressure")
        open(joinpath(save, "console.log"), "w") do io
            redirect_stdout(io) do
                run_case(case; results_dir)
            end
        end
    end
    verify_run(record, name, reynolds, n, boundary, diffusion)
    println("Verified $group/$name")
    flush(stdout)
end

function main(args)
    resolutions = isempty(args) ? RESOLUTIONS : parse.(Int, args)
    all(in(RESOLUTIONS), resolutions) || error("Select resolutions from 20, 40, and 80")
    failures = String[]
    for n in resolutions, reynolds in REYNOLDS, (model_name, diffusion) in MODELS, (wall_name, boundary) in BOUNDARIES
        try
            run_member(reynolds, n, wall_name, boundary, model_name, diffusion)
        catch err
            err isa InterruptException && rethrow()
            label = "$wall_name/$model_name/Re$reynolds/n$n"
            push!(failures, label)
            @error "Matrix member failed" label exception = (err, catch_backtrace())
        end
    end
    isempty(failures) || error("Failed cases: $(join(failures, ", "))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
