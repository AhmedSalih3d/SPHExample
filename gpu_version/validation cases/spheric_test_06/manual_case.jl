# Run from gpu_version:
# julia --project=. -t 2,0 "validation cases/spheric_test_06/manual_case.jl"
# Edit this setup block for a single interactive SPHERIC Test 06 run.
using SPHExampleGPU
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06

let
    reynolds = 100
    dx = 0.025
    sound_speed = 40.0
    duration = 8.0
    output_interval = 0.02
    run_tag = "manual"
    n = round(Int, 1 / dx)
    run_name = "Re$(lpad(string(reynolds), 3, '0'))_n$(lpad(string(n), 3, '0'))"
    save_location = joinpath(@__DIR__, "runs", run_tag, run_name)
    plot_location = joinpath(@__DIR__, "runs", run_tag, "paraview_force_plot")
    results_location = joinpath(@__DIR__, "results", run_tag)

    case = SphericTest06.build_case(
        reynolds = reynolds,
        dx = dx,
        duration = duration,
        interval = output_interval,
        sound_speed = sound_speed,
        save = save_location,
        visualize_in_paraview = true,
    )
    SphericTest06.run_case(case; results_dir = results_location)

    pvpython = Sys.which("pvpython")
    if pvpython === nothing
        @warn "pvpython was not found; the simulation and force CSV are saved, but the ParaView comparison state was not generated."
    else
        plot_script = joinpath(@__DIR__, "paraview_force_plot.py")
        run(`$pvpython --no-mpi --force-offscreen-rendering $plot_script --results $results_location --output $plot_location --re $reynolds --n $n`)
    end
end
