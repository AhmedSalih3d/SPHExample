# From gpu_version: julia --project=. -t 2,0 "validation cases/spheric_test_06/run.jl"
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06
using Printf
using CUDA
using TOML

function main(args)
    options = Dict{String,String}()
    iseven(length(args)) || error("Options require values: --dx, --re, --duration, --interval, --sound-speed, --tag")
    for i in 1:2:length(args)
        args[i] in ("--dx", "--re", "--duration", "--interval", "--sound-speed", "--tag") || error("Unknown option $(args[i])")
        options[args[i]] = args[i+1]
    end
    spacings = parse.(Float64, split(get(options, "--dx", "0.05,0.025,0.016666666666666666"), ','))
    reynolds_numbers = parse.(Int, split(get(options, "--re", "50,100,150"), ','))
    duration = parse(Float64, get(options, "--duration", "8"))
    interval = parse(Float64, get(options, "--interval", "0.02"))
    sound_speed = parse(Float64, get(options, "--sound-speed", "40"))
    tag = get(options, "--tag", "validation")
    occursin(r"^[A-Za-z0-9_-]+$", tag) || error("Tag must be a simple directory name")
    results = joinpath(@__DIR__, "results", tag)
    for dx in spacings, reynolds in reynolds_numbers
        name = @sprintf("Re%03d_n%03d", reynolds, round(Int, 1/dx))
        if isfile(joinpath(results, name * ".toml"))
            previous = TOML.parsefile(joinpath(results, name * ".toml"))
            all(isapprox(previous[key], value; rtol = 1e-12) for (key,value) in
                (("dx",dx), ("duration",duration), ("output_interval",interval), ("sound_speed",sound_speed))) ||
                error("Existing run settings differ; use a new --tag: $name")
            println("Already completed: ", name)
            continue
        end
        save = joinpath(@__DIR__, "runs", tag, name)
        case = build_case(; reynolds, dx, duration, interval, sound_speed, save)
        run_case(case; results_dir = results)
        GC.gc(true)
        CUDA.reclaim()
    end
    comparison = compare_histories(results)
    for row in comparison.metrics
        @printf("Re=%d dx=%.6f %s NRMSE=%.2f%% pass=%s\n", row.reynolds, row.dx, row.component, 100row.nrmse, row.drag_pass)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
