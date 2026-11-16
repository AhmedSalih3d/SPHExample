# Run from gpu_version:
# julia --project=. -t 2,0 "validation cases/spheric_test_06/manual_case.jl"
# Edit the layout and simulation settings below for a manual SPHERIC Test 06 run.
using SPHExampleGPU
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06

let
    # Geometry is in metres. Tank and body edges must align with dx.
    layout = (
        tank_origin = (0.0, 0.0),
        tank_size = (10.0, 5.0),
        body_center = (1.5, 2.5),
        body_size = (1.0, 1.0),
    )

    reynolds = 100
    dx = 0.025
    density = 1000.0
    gravity = 0.0
    reference_speed = 1.0
    reference_length = layout.body_size[1]
    span = 1.0
    sound_speed = 40.0
    cfl = 0.2
    density_diffusion = 0.1
    artificial_viscosity_alpha = 0.0
    kernel = WendlandC2()
    kernel_h_factor = sqrt(2)
    wall_layers = nothing

    duration = 8.0
    output_interval = 0.02
    float_type = Float32
    viscosity_model = Laminar()
    density_diffusion_model = ZeroGravityLinearDensityDiffusion()
    time_stepping = SymplecticTimeStepping()
    shifting = NoShifting

    graph = true
    gpu_clip_time_step_to_output = true
    gpu_double_position = true
    gpu_legacy_timestep = true
    gpu_cell_subdivision = 1
    gpu_x_cell_subdivision = 1
    gpu_radial_interactions = false
    gpu_interaction_fastmath = false
    gpu_preload_interaction_types = false
    gpu_boundary_forces = false
    gpu_output_buffer_bytes = 16 * 2^20
    visualize_in_paraview = true

    run_tag = "manual_layout"
    n = round(Int, reference_length / dx)
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
        cfl = cfl,
        save = save_location,
        float_type = float_type,
        graph = graph,
        visualize_in_paraview = visualize_in_paraview,
        layout = layout,
        density = density,
        gravity = gravity,
        reference_speed = reference_speed,
        reference_length = reference_length,
        span = span,
        density_diffusion = density_diffusion,
        artificial_viscosity_alpha = artificial_viscosity_alpha,
        kernel = kernel,
        kernel_h_factor = kernel_h_factor,
        wall_layers = wall_layers,
        viscosity_model = viscosity_model,
        density_diffusion_model = density_diffusion_model,
        time_stepping = time_stepping,
        shifting = shifting,
        gpu_clip_time_step_to_output = gpu_clip_time_step_to_output,
        gpu_double_position = gpu_double_position,
        gpu_legacy_timestep = gpu_legacy_timestep,
        gpu_cell_subdivision = gpu_cell_subdivision,
        gpu_x_cell_subdivision = gpu_x_cell_subdivision,
        gpu_radial_interactions = gpu_radial_interactions,
        gpu_interaction_fastmath = gpu_interaction_fastmath,
        gpu_preload_interaction_types = gpu_preload_interaction_types,
        gpu_boundary_forces = gpu_boundary_forces,
        gpu_output_buffer_bytes = gpu_output_buffer_bytes,
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
