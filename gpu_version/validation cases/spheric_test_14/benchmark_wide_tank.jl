include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "full_case.jl"))
using CUDA

function benchmark_wide_tank()
    tag = "benchmark_3d_48m"
    output = joinpath(@__DIR__, "results", tag)
    for nominal in RELEASES
        pause_time = nominal.name == "float1_down" ? 0.725 : 0.0
        # Float 2's held-load draft measured on the 25 mm grid.
        release = nominal.name == "float2_down" ? merge(nominal, (; draft = 0.21988712735907776)) : nominal
        isfile(joinpath(output, release.name * ".toml")) && continue
        duration = 4.5 + pause_time
        case = build_full_case(release; half_width = 2.4, duration, pause_time,
            output_interval = 0.1, save_location = joinpath(@__DIR__, "runs", tag, release.name))
        case = merge(case, (; release_time = pause_time))
        state_sampling = (; times = collect(0.005:0.005:duration), floating_history = true)
        case.metadata.OutputTimes = state_sampling.times[20:20:end]
        run_case(release, case; output, state_sampling)
        GC.gc(true)
        CUDA.reclaim()
    end
end

benchmark_wide_tank()
