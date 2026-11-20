using SPHExampleGPU, CUDA
include("dual_compare.jl")
const C = Test06DualComparison
const S = C.SphericTest06
const TAG = "planar_shifting_benchmark"

function main(args)
    matched = "--matched-numerics" in args
    args = filter(!=("--matched-numerics"), args)
    length(args) in (1, 2) && first(args) in ("native", "dual") ||
        error("Expected native or dual and optional resolution (40 or 160)")
    solver = first(args)
    n = length(args) == 2 ? parse(Int, last(args)) : 160
    n in (40, 160) || error("Expected resolution 40 or 160")
    tag = n == 160 ? TAG : TAG * "_n$n"
    matched && (tag *= "_ddt2")
    work = joinpath(@__DIR__, "runs", tag)
    out = joinpath(@__DIR__, "results", tag)
    layout = S.build_case(; dx = 1/n, kernel_h_factor = 2sqrt(2), wall_layers = 7)
    particles = sum(values(layout.counts))
    for reynolds in (50, 100, 150)
        results = joinpath(out, solver, "re$reynolds")
        save = joinpath(work, solver, "re$reynolds")
        println("Starting $solver Re=$reynolds")
        flush(stdout)
        if solver == "native"
            numerics = matched ? (;
                density_diffusion_model = FourtakasDensityDiffusion(),
                time_stepping = DualSPHysicsSymplectic(density_min = 700.0, density_max = 1300.0),
                graph = false, gpu_clip_time_step_to_output = false) : (;)
            case = S.build_case(; reynolds, dx = 1/n, sound_speed = 160.0,
                cfl = 0.2, kernel_h_factor = 2sqrt(2), wall_layers = 7,
                motion_kind = :integrated, shifting = typeof(PlanarShifting(0.1)), save, numerics...)
            matched && (case.metadata.GPULanesPerParticle = 1)
            mkpath(save)
            open(joinpath(save, "console.log"), "w") do io
                redirect_stdout(io) do
                    S.run_case(case; results_dir = results)
                end
            end
        else
            prefix = C.prepare_dual(; n, reynolds, sound_speed = 160, coefh = 2.0,
                wall_layers = 7, motion_kind = :integrated, shifting = 3, tag,
                diffusion = matched ? 2 : 1,
                trajectory = S.body_trajectory(motion_kind = :integrated))
            C.verify_geometry(prefix, n; coefh = 2.0, wall_layers = 7)
            finished = all(file -> isfile(joinpath(save, file)), ("Run.csv", "pressure.csv", "total.csv"))
            info = finished ? C.dual_run_info(save) : C.dual_run(prefix, save; reynolds)
            info.frames == 401 && info.particles == particles || error("Unexpected DualSPHysics frame or particle count: $info")
            C.extract_dual(save, results; square_particles = n^2)
            C.save_info(joinpath(results, "dual.toml"), info)
            cp(prefix * ".xml", joinpath(results, "Case.xml"); force = true)
        end
        println("Completed $solver Re=$reynolds")
        flush(stdout)
        GC.gc(true)
        CUDA.reclaim()
    end
end

main(ARGS)
