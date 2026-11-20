# Edit the settings below and run from gpu_version:
# julia --project=. -t 2,0 "validation cases/spheric_test_14/manual_case.jl"
using SPHExampleGPU, StaticArrays, DelimitedFiles, CSV, TOML

const RELEASES = (
    (name = "float1_down", title = "Float 1, -74 mm release", file = "Fl1-0p074.txt", mass = 9.75, cog = 0.090,
     radius = 0.15, corner = 0.15, height = 0.29, draft = 0.187, offset = -0.074),
    (name = "float1_up", title = "Float 1, +76 mm release", file = "Fl1+0p76.txt", mass = 9.75, cog = 0.090,
     radius = 0.15, corner = 0.15, height = 0.29, draft = 0.187, offset = 0.076),
    (name = "float2_down", title = "Float 2, -88 mm release", file = "Fl2-0p088.txt", mass = 23.42, cog = 0.100,
     radius = 0.20, corner = 0.15, height = 0.33, draft = 0.23, offset = -0.088),
)

function run_manual_case()
    Dimensions = 3
    FloatType  = Float32

    # Choose float1_down, float1_up, or float2_down.
    release_name = "float1_down"
    release = only(filter(release -> release.name == release_name, RELEASES))

    # Geometry in metres. The water surface is at z = 0.
    dx         = 0.025
    half_width = 1.2
    depth      = 0.8
    wall_top_z = 0.2

    # h scales with dx. Both supported kernel families have support radius 2h.
    kernel_family   = WendlandC2()
    kernel_h_factor = 1.3  # h / dx

    sponge_width = 0.5
    sponge_rate  = 12.0
    duration = 4.5
    output_interval = 0.01
    visualize_in_paraview = true

    run_name        = "manual_dp$(dx)_half$(half_width)_t$(duration)"
    save_location   = joinpath(@__DIR__, "runs", run_name, release_name)
    result_location = joinpath(@__DIR__, "results", run_name)

    SimConstants = SimulationConstants{FloatType}(Dimensions;
        dx  = dx,
        ρ₀  = 1000.0,
        g   = 9.81,
        c₀  = 25.0,
        α   = 0.01,
        δᵩ  = 0.1,
        CFL = 0.2,
    )

    SimKernel = SPHKernelInstance{Dimensions, FloatType}(kernel_family;
        h = FloatType(kernel_h_factor * dx))
    SimulationGeometry, equilibrium_center_z = sample_geometry(
        release, SimConstants, SimKernel; dx, half_width, depth, wall_top_z)

    SimMetaData = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        NoMDBC,
        StoreLog,
    }(
        SimulationName          = "FreeHeave",
        SaveLocation            = save_location,
        SimulationTime          = duration,
        OutputTimes             = output_interval,
        OutputVariables         = ["Density", "Pressure", "Velocity", "Type", "GroupMarker"],
        ExportSingleVTKHDF      = true,
        VisualizeInParaview     = false, # Open the combined state after the heave CSV is saved.
        OpenLogFile             = false,
        GPUUseGraph             = true,
        GPUDoublePosition       = true,
        GPUClipTimeStepToOutput = true,
    )

    SimViscosity        = ArtificialViscosity()
    SimDensityDiffusion = FourtakasDensityDiffusion()
    SimTimeStepping     = SymplecticTimeStepping()
    SimDamping          = DampingCylinder((0.0, 0.0, -depth), (0.0, 0.0, wall_top_z),
        half_width - sponge_width, half_width; rate = sponge_rate)

    case = (; geometry = SimulationGeometry, metadata = SimMetaData, constants = SimConstants,
        kernel = SimKernel, damping = SimDamping, equilibrium_center_z, dx, half_width, depth, duration,
        viscosity = SimViscosity, diffusion = SimDensityDiffusion, stepping = SimTimeStepping)
    completed = run_case(release, case; output = result_location)

    state = write_heave_state(release, completed; layout = "stacked", screenshot = true)
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

function in_float(p, release, base)
    z = p[3] - base
    r = hypot(p[1], p[2])
    core = release.radius - release.corner
    return z <= release.height && hypot(max(r - core, 0), max(release.corner - z, 0)) <= release.corner
end

function sample_geometry(release, constants::SimulationConstants{T}, kernel; dx, half_width, depth,
                         wall_top_z = 0.2) where {T}
    all(isfinite, (dx, half_width, depth, wall_top_z)) && dx > 0 && half_width >= 0.8 && depth > 0 && wall_top_z >= 0 ||
        throw(ArgumentError("Invalid spacing or tank dimensions"))
    nx, nz = round(Int, 2half_width / dx), round(Int, depth / dx)
    isapprox(nx * dx, 2half_width) && isapprox(nz * dx, depth) ||
        throw(ArgumentError("Spacing must divide the tank width and water depth"))
    layers = ceil(Int, kernel.H / dx)
    base = release.offset - release.draft
    equilibrium_center_z = release.cog - release.draft
    water = SVector{3,Float64}[]
    wall = similar(water)
    body = similar(water)
    top = max(nz - 1, ceil(Int, (depth + base + release.height) / dx))
    wall_top = ceil(Int, (depth + wall_top_z) / dx)
    for iz in -layers:max(top, wall_top), iy in -layers:nx+layers-1, ix in -layers:nx+layers-1
        p = SVector((ix + 0.5 - nx/2) * dx, (iy + 0.5 - nx/2) * dx, (iz + 0.5) * dx - depth)
        if iz < 0 || !(0 <= ix < nx && 0 <= iy < nx)
            iz <= wall_top && push!(wall, p)
        elseif in_float(p, release, base)
            push!(body, p)
        elseif iz < nz
            push!(water, p)
        end
    end
    isempty(body) && error("Float has no particles at dx=$dx")
    floating = FloatingDetails{T}(RelativeWeight = release.mass / (constants.m₀ * length(body)),
        Center = (0.0, 0.0, equilibrium_center_z + release.offset))
    density(points) = hydrostatic_density(points, constants; water_level = 0.0)
    geometry = [
        SPHGeometry{3,T}(wall; Density = density(wall), GroupMarker = 1, Type = Fixed),
        SPHGeometry{3,T}(water; Density = density(water), GroupMarker = 2, Type = Fluid),
        SPHGeometry{3,T}(body; Density = density(body), GroupMarker = 3, Type = Floating, Floating = floating),
    ]
    return geometry, equilibrium_center_z
end

function run_case(release, case; output)
    save = case.metadata.SaveLocation
    (; dx, half_width, depth, duration) = case
    isfile(joinpath(save, "FreeHeave.vtkhdf")) && error("Run already exists at $save")
    reference_path = joinpath(@__DIR__, "reference", release.file)
    reference = readdlm(reference_path, Float64)
    size(reference, 2) == 2 && all(isfinite, reference) &&
        all(isapprox.(diff(reference[:,1]), 0.005; atol = 1e-10)) || error("Invalid reference history")
    mkpath(save)
    logger = SimulationLogger(save)
    host = AllocateDataStructures(case.geometry, case.metadata)
    result = try
        RunSimulation(SimGeometry = case.geometry, SimMetaData = case.metadata,
            SimConstants = case.constants, SimKernel = case.kernel, SimLogger = logger, SimParticles = host,
            SimViscosity = case.viscosity, SimDensityDiffusion = case.diffusion,
            SimTimeStepping = case.stepping, SimDamping = case.damping)
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    result.final_state.complete && result.diagnostics.frozen_particles == 0 || error("Incomplete float run")
    records = CSV.File(joinpath(save, "FreeHeave_Floating.csv"))
    heave = 1000 .* (getproperty(records, Symbol("Center:2")) .- case.equilibrium_center_z)
    time = records.Time
    length(time) > 1 && all(>(0), diff(time)) && isapprox(last(time), duration; atol = 1e-6) ||
        error("Incomplete or invalid floating-body history")
    isapprox(first(heave), 1000release.offset; atol = 1e-4) || error("Wrong initial heave datum")
    mkpath(output)
    history = joinpath(output, release.name * ".csv")
    CSV.write(history, (; time, heave_mm = heave,
        surge_mm = 1000 .* getproperty(records, Symbol("Center:0")),
        sway_mm = 1000 .* getproperty(records, Symbol("Center:1"))))
    open(joinpath(output, release.name * ".toml"), "w") do io
        TOML.print(io, Dict("dx" => dx, "half_width" => half_width, "depth" => depth, "duration" => duration,
            "kernel" => string(nameof(typeof(case.kernel.kernel))), "h" => Float64(case.kernel.h),
            "kernel_support_factor" => Float64(case.kernel.k),
            "mass" => release.mass, "draft" => release.draft, "offset" => release.offset,
            "fluid_particles" => length(case.geometry[2].Particles), "frames" => length(time),
            "steps" => result.performance.steps, "wall_seconds" => result.performance.wall_time_seconds))
    end
    save_heave_plot(output, release.name)
    return (; particles = joinpath(save, "FreeHeave.vtkhdf"), history, reference = reference_path)
end

function save_heave_plot(output, release_name)
    python = Sys.which("python")
    python === nothing && (python = Sys.which("python3"))
    python === nothing && error("Python with NumPy and Matplotlib is required to save the heave plot. Results are in $output.")
    script = joinpath(@__DIR__, "plot.py")
    run(`$python $script $(abspath(output)) --release $release_name`)
    println("Heave plot saved to ", joinpath(output, release_name * "_heave_comparison.png"))
end

"""Write a launcher that builds and saves the comparison inside ParaView."""
function write_heave_state(release, output; layout = "side-by-side", screenshot = false)
    (; particles, history, reference) = output
    script = joinpath(@__DIR__, "paraview_heave.py")
    state = joinpath(dirname(particles), "FreeHeave_Comparison.py")
    python_path(path) = repr(replace(abspath(path), '\\' => '/'))
    preview = screenshot ? "Path(" * python_path(replace(state, ".py" => ".png")) * ")" : "None"
    write(state, """
    from pathlib import Path
    from types import SimpleNamespace
    import runpy

    args = SimpleNamespace(
        particles=Path($(python_path(particles))),
        history=Path($(python_path(history))),
        reference=Path($(python_path(reference))),
        state=Path($(python_path(replace(state, ".py" => ".pvsm")))),
        screenshot=$preview,
        title=$(repr(release.title)),
        layout=$(repr(layout)),
    )
    runpy.run_path($(python_path(script)))["build_view"](args)
    """)
    return state
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_manual_case()
end
