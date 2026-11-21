# Definitions only; load case.jl before this file.
"""
    build_full_case(release; boundary_mode=SimpleMDBC, ...)

Full 3D Test 14 float with the experimental mass and center of gravity, free in
all six rigid degrees of freedom. Boundary treatment and physical viscosity match
the updated planar controls. Domain size and resolution remain independent inputs.
"""
function build_full_case(release; dx = 0.025, half_width = 1.2, depth = 0.8, wall_top_z = 0.25,
        kernel_family = WendlandC2(), kernel_h_factor = 2., boundary_mode = SimpleMDBC,
        viscosity = Laminar(), diffusion = LinearDensityDiffusion(), c0 = 25., cfl = 0.2,
        sponge_width = 0.5, sponge_rate = 12., duration = 4.5, output_interval = 0.005,
        pause_time = 0., output_variables = ["Type", "Velocity", "Pressure"],
        save_location = joinpath(@__DIR__, "runs", "full_mdbc", release.name))
    boundary_mode in (NoMDBC, SimpleMDBC) || throw(ArgumentError("Floating UpdatedMDBC is unsupported; use NoMDBC or SimpleMDBC"))
    all(isfinite, (duration, output_interval, kernel_h_factor, pause_time, sponge_rate)) &&
        duration > 0 && output_interval > 0 && kernel_h_factor > 0 && pause_time >= 0 && sponge_rate >= 0 ||
        throw(ArgumentError("Invalid full-case time, kernel or damping settings"))
    0 < sponge_width < half_width - release.radius || throw(ArgumentError("Keep the sponge outside the float"))
    T = Float32
    constants = SimulationConstants{T}(3; dx, ρ₀ = 1000., g = 9.81, c₀ = c0, CFL = cfl, ν₀ = 1e-6, δᵩ = 0.1)
    kernel = SPHKernelInstance{3,T}(kernel_family; h = T(kernel_h_factor * dx))
    geometry, equilibrium_center_z = sample_geometry(release, constants, kernel;
        dx, half_width, depth, wall_top_z, pause_time, boundary_mode)
    metadata = SimulationMetaData{3,T,NoShifting,NoKernelOutput,boundary_mode,StoreLog}(
        SimulationName = "FreeHeave", SaveLocation = save_location, SimulationTime = duration,
        OutputTimes = output_interval, OutputVariables = copy(output_variables), ExportSingleVTKHDF = true,
        VisualizeInParaview = false, OpenLogFile = false, GPUDoublePosition = true,
        GPUUseGraph = true, GPUClipTimeStepToOutput = true)
    damping = DampingCylinder((0., 0., -depth), (0., 0., wall_top_z),
        half_width - sponge_width, half_width; rate = sponge_rate)
    return (; geometry, metadata, constants, kernel, damping, equilibrium_center_z, dx, half_width, depth,
        duration, viscosity, diffusion, stepping = SymplecticTimeStepping(), sponge_width, sponge_rate, pause_time,
        boundary_mode = string(nameof(boundary_mode)))
end
