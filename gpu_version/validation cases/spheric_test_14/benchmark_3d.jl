include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "full_case.jl"))
using CUDA

all(==("--hold"), ARGS) || error("Usage: benchmark_3d.jl [--hold]")
held = "--hold" in ARGS
for release in (held ? (first(RELEASES),) : RELEASES)
    tag = held ? "benchmark_3d_25mm_hold" : "benchmark_3d_25mm"
    output = joinpath(@__DIR__, "results", tag)
    isfile(joinpath(output, release.name * ".toml")) && continue
    case = build_full_case(release; pause_time = held ? 0.060 : 0., output_variables = ["Type"],
        save_location = joinpath(@__DIR__, "runs", tag, release.name))
    run_case(release, case; output)
    GC.gc(true)
    CUDA.reclaim()
end
