include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "full_case.jl"))
using CUDA

length(ARGS) <= 1 || error("Usage: benchmark_3d.jl [--hold | hold_seconds]")
pause_time = isempty(ARGS) ? 0.0 : only(ARGS) == "--hold" ? 0.060 : parse(Float64, only(ARGS))
isfinite(pause_time) && pause_time >= 0 || error("Hold time must be finite and nonnegative")
hold_samples = round(Int, pause_time / 0.005)
isapprox(pause_time, hold_samples * 0.005; atol = 1e-12, rtol = 0) ||
    error("Hold time must be a multiple of the 5 ms output interval")
held = pause_time > 0
for release in (held ? (first(RELEASES),) : RELEASES)
    legacy_hold = pause_time == 0.060
    tag = !held ? "benchmark_3d_25mm" : legacy_hold ? "benchmark_3d_25mm_hold" :
        "benchmark_3d_25mm_hold_$(round(Int, 1000pause_time))ms"
    output = joinpath(@__DIR__, "results", tag)
    isfile(joinpath(output, release.name * ".toml")) && continue
    duration = legacy_hold ? 4.5 : 4.5 + pause_time
    case = build_full_case(release; pause_time, duration, output_variables = ["Type"],
        save_location = joinpath(@__DIR__, "runs", tag, release.name))
    if held && !legacy_hold
        case = merge(case, (; release_time = pause_time))
    end
    run_case(release, case; output)
    GC.gc(true)
    CUDA.reclaim()
end
