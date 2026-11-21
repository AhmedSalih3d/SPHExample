# Cartesian vertical section of SPHERIC Test 14.
# julia --project=. -t 2,0 "validation cases/spheric_test_14/manual_case_2d.jl"
include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "planar_case.jl"))

function manual_planar_case()
    selection = "all" # Or "float1_down", "float1_up", "float2_down".
    visualize_in_paraview = true

    dx = 0.0125
    depth = 0.8
    half_width = 8.0
    wall_top_z = 0.25
    sponge_width = 1.0
    sponge_rate = 12.0

    # Other choices: NoMDBC, SimpleMDBC, UpdatedMDBCNoPenetration, UpdatedMDBCWithImpedance.
    boundary_mode = UpdatedMDBC
    kernel_family = WendlandC2()
    kernel_h_factor = 1.3
    viscosity = Laminar() # Or ZeroViscosity(), ArtificialViscosity(), LaminarSPS().
    diffusion = LinearDensityDiffusion()
    c0 = 25.0
    cfl = 0.2
    nu = 1e-6
    alpha = 0.01
    delta = 0.1

    output_interval = 0.005 # 200 Hz.
    pause_time = 0.5 # Try 0.5 s to compare a held release with immediate release.
    duration = pause_time + 4.5
    output_variables = ["Type", "Velocity", "Pressure"]
    # Give runs with other edited settings distinct names to keep their output.
    run_name = "planar_$(nameof(boundary_mode))_$(nameof(typeof(viscosity)))_dp$(dx)_half$(half_width)" *
        "_h$(kernel_h_factor)_c$(c0)_hold$(pause_time)_t$(duration)"

    releases = selection == "all" ? RELEASES : filter(r -> r.name == selection, RELEASES)
    isempty(releases) && throw(ArgumentError("Unknown release $selection"))
    for (index, release) in enumerate(releases)
        show_view = visualize_in_paraview && index == length(releases)
        run_planar_case(release; visualize_in_paraview = show_view, boundary_mode, dx, depth,
            half_width, wall_top_z, sponge_width, sponge_rate, duration, output_interval, pause_time,
            kernel_family, kernel_h_factor, viscosity, diffusion, c0, cfl, nu, alpha, delta,
            output_variables, run_name)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    manual_planar_case()
end
