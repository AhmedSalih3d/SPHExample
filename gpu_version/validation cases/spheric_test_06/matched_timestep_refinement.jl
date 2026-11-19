include("case.jl")
using .SphericTest06
using SPHExampleGPU, CSV, TOML, Printf, CUDA

const TAG = "matched_timestep_refinement"
const RESOLUTIONS = (40, 80, 160)

function main(args)
    fine_time = args == ["fine-time"]
    resolutions = fine_time ? (160,) : isempty(args) ? RESOLUTIONS : parse.(Int, args)
    all(in(RESOLUTIONS), resolutions) || error("Select L/dp = 40, 80, or 160")
    tag = fine_time ? joinpath(TAG, "fine_time") : TAG
    duration = fine_time ? 0.9 : 8.0
    frames = round(Int, duration / 0.02) + 1
    for n in resolutions
        cfl = n / (fine_time ? 1600 : 800)
        name = @sprintf("Re100_n%03d", n)
        save = joinpath(@__DIR__, "runs", tag, name)
        results = joinpath(@__DIR__, "results", tag)
        record = joinpath(results, "$name.toml")
        if !isfile(record)
            println("Running $name, CFL=$cfl")
            flush(stdout)
            case = build_case(; dx = 1 / n, duration, sound_speed = 160.0, cfl, kernel_h_factor = 2sqrt(2),
                shifting = PlanarShifting, motion_kind = :integrated, save)
            mkpath(save)
            open(joinpath(save, "console.log"), "w") do io
                redirect_stdout(io) do
                    run_case(case; results_dir = results)
                end
            end
            GC.gc(true)
            CUDA.reclaim()
        end
        info = TOML.parsefile(record)
        expected = ("reynolds" => 100, "dx" => 1/n, "cfl" => Float64(Float32(cfl)),
            "h" => Float64(Float32(2sqrt(2)/n)), "sound_speed" => 160.0, "frames" => frames,
            "duration" => duration, "frozen_particles" => 0, "boundary_mode" => "NoMDBC",
            "shifting" => "PlanarShifting", "motion_kind" => "integrated",
            "density_diffusion_model" => "ZeroGravityLinearDensityDiffusion")
        all(pair -> info[first(pair)] == last(pair), expected) || error("Unexpected run configuration: $name")
        history = CSV.File(joinpath(results, "$name.csv"))
        length(history) == frames && all(isapprox(actual, target; atol = 1e-12, rtol = 0)
            for (actual, target) in zip(history.time, 0.0:0.02:duration)) ||
            error("Incomplete force history for $name")
        println("Verified $name, $(info["steps"]) steps")
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
