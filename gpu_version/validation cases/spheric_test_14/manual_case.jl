# Full 3D geometry of SPHERIC Test 14.
# julia --project=. -t 2,0 "validation cases/spheric_test_14/manual_case.jl"
include(joinpath(@__DIR__, "case.jl"))

function run_manual_case(release; visualize_in_paraview)
    Dimensions = 3
    FloatType = Float32

    release_name = release.name
    dx = 0.025
    depth = 0.8
    half_width = 1.2 * 2
    wall_top_z = 0.25
    sponge_width = 0.5
    sponge_rate = 12.0
    duration = 4.5
    output_interval = 0.005 # 200 Hz, matching the experiment.
    pause_time = 0.0 # Optional initial hold; the benchmark releases at t = 0.

    boundary_mode = UpdatedMDBC # Select UpdatedMDBCNoPenetration to add normal contact.
    kernel_family = WendlandC2()
    kernel_h_factor = 2.0
    SimConstants = SimulationConstants{FloatType}(Dimensions;
        dx, ρ₀ = 1000.0, g = 9.81, c₀ = 25.0, ν₀ = 1e-6, CFL = 0.2, δᵩ = 0.1)
    SimKernel = SPHKernelInstance{Dimensions,FloatType}(kernel_family; h = FloatType(kernel_h_factor * dx))
    SimViscosity = Laminar()
    SimDensityDiffusion = LinearDensityDiffusion()
    SimTimeStepping = SymplecticTimeStepping()

    run_name = "full_$(nameof(boundary_mode))_dp$(dx)_half$(half_width)_h$(kernel_h_factor)_t$(duration)"
    save_location = joinpath(@__DIR__, "runs", run_name, release_name)
    result_location = joinpath(@__DIR__, "results", run_name)

    boundary_mode in (NoMDBC, SimpleMDBC, UpdatedMDBC, UpdatedMDBCWithImpedance, UpdatedMDBCNoPenetration) ||
        throw(ArgumentError("Unsupported floating boundary mode"))
    all(isfinite, (dx, depth, half_width, wall_top_z, duration, output_interval, pause_time, sponge_rate, kernel_h_factor)) &&
        dx > 0 && depth > 0 && half_width >= 0.8 && wall_top_z >= 0 && duration > 0 &&
        output_interval > 0 && pause_time >= 0 && sponge_rate >= 0 && kernel_h_factor > 0 ||
        throw(ArgumentError("Invalid geometry, time, kernel or damping settings"))
    nx, nz = round(Int, 2half_width / dx), round(Int, depth / dx)
    isapprox(nx * dx, 2half_width) && isapprox(nz * dx, depth) ||
        throw(ArgumentError("Spacing must divide tank width and water depth"))
    0 < sponge_width < half_width - release.radius ||
        throw(ArgumentError("Keep the damping zone outside the float"))

    # Sample the water and the complete float on one Cartesian lattice.
    equilibrium_center_z = release.cog - release.draft
    base = release.offset - release.draft
    layers = ceil(Int, SimKernel.H / dx)
    top = max(nz - 1, ceil(Int, (depth + base + release.height) / dx))
    wall_top = ceil(Int, (depth + wall_top_z) / dx)
    wall = SVector{Dimensions,Float64}[]
    water, body = similar(wall), similar(wall)
    for iz in -layers:max(top, wall_top), iy in -layers:nx+layers-1, ix in -layers:nx+layers-1
        x = (ix + 0.5 - nx/2) * dx
        y = (iy + 0.5 - nx/2) * dx
        z = (iz + 0.5) * dx - depth
        point = SVector(x, y, z)
        if iz < 0 || !(0 <= ix < nx && 0 <= iy < nx)
            iz <= wall_top && push!(wall, point)
        elseif in_float(point, release, base)
            push!(body, point)
        elseif iz < nz
            push!(water, point)
        end
    end
    isempty(body) && error("No body points at this spacing")

    # Use the experimental 3D mass and center of gravity. All rigid motions are free.
    floating = FloatingDetails{FloatType}(
        RelativeWeight = release.mass / (SimConstants.m₀ * length(body)),
        PauseTime = pause_time,
        Center = (0.0, 0.0, equilibrium_center_z + release.offset),
    )
    density(points) = hydrostatic_density(points, SimConstants; water_level = 0.0)

    if boundary_mode !== NoMDBC
        # Reflect tank ghosts across its physical side and bottom planes.
        function wall_ghost(point)
            surface = SVector(
                clamp(point[1], -half_width, half_width),
                clamp(point[2], -half_width, half_width),
                max(point[3], -depth),
            )
            return 2surface - point
        end
        wall_ghosts = wall_ghost.(wall)
        body_ghosts = [2float_surface_projection(point, release, base) - point for point in body]
        boundary_data(points, ghosts) = StructArray((;
            Position = points,
            Density = FloatType.(density(points)),
            GhostPoints = ghosts,
            GhostNormals = ghosts .- points,
        ))
        Tank = SPHGeometry{Dimensions,FloatType}(Particles = boundary_data(wall, wall_ghosts),
            GroupMarker = 1, Type = Fixed)
        Float = SPHGeometry{Dimensions,FloatType}(Particles = boundary_data(body, body_ghosts),
            GroupMarker = 3, Type = Floating, Floating = floating)
    else
        Tank = SPHGeometry{Dimensions,FloatType}(wall;
            Density = density(wall), GroupMarker = 1, Type = Fixed)
        Float = SPHGeometry{Dimensions,FloatType}(body;
            Density = density(body), GroupMarker = 3, Type = Floating, Floating = floating)
    end
    Water = SPHGeometry{Dimensions,FloatType}(water;
        Density = density(water), GroupMarker = 2, Type = Fluid)

    SimDamping = DampingCylinder((0.0, 0.0, -depth), (0.0, 0.0, wall_top_z),
        half_width - sponge_width, half_width; rate = sponge_rate)
    SimMetaData = SimulationMetaData{Dimensions,FloatType,NoShifting,NoKernelOutput,boundary_mode,StoreLog}(
        SimulationName = "FreeHeave", SaveLocation = save_location, SimulationTime = duration,
        OutputTimes = output_interval, OutputVariables = ["Type", "Velocity", "Pressure"],
        ExportSingleVTKHDF = true, VisualizeInParaview = false, OpenLogFile = false,
        GPUUseGraph = true, GPUDoublePosition = true, GPUClipTimeStepToOutput = true)

    case = (; geometry = [Tank, Water, Float], metadata = SimMetaData, constants = SimConstants,
        kernel = SimKernel, damping = SimDamping, equilibrium_center_z, dx, half_width, depth, duration,
        viscosity = SimViscosity, diffusion = SimDensityDiffusion, stepping = SimTimeStepping,
        sponge_width, sponge_rate, pause_time, boundary_mode = string(nameof(boundary_mode)))
    completed = run_case(release, case; output = result_location)

    title = merge(release, (; title = release.title * " | full 3D | $(nameof(boundary_mode))"))
    state = write_heave_state(title, completed; layout = "stacked", screenshot = true)
    println("ParaView launcher saved to ", state)
    if visualize_in_paraview
        paraview = Sys.which("paraview")
        if paraview === nothing
            @warn "Load $state using File > Load State in ParaView."
        else
            run(`$paraview --state=$state`; wait = false)
        end
    end
end

let
    # Edit this to "float1_down", "float1_up", "float2_down", or "all".
    selection = "float1_down"
    visualize_in_paraview = true
    releases = selection == "all" ? RELEASES : filter(r -> r.name == selection, RELEASES)
    isempty(releases) && throw(ArgumentError("Unknown release $selection"))
    for (index, release) in enumerate(releases)
        run_manual_case(release; visualize_in_paraview = visualize_in_paraview && index == length(releases))
    end
end
