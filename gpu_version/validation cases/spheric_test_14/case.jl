using SPHExampleGPU, StaticArrays, DelimitedFiles, CSV, TOML, StructArrays, LinearAlgebra

const RELEASES = (
    (name = "float1_down", title = "Float 1, -74 mm release", file = "Fl1-0p074.txt", mass = 9.75, cog = 0.090,
     radius = 0.15, corner = 0.15, height = 0.29, draft = 0.187, offset = -0.074),
    (name = "float1_up", title = "Float 1, +76 mm release", file = "Fl1+0p76.txt", mass = 9.75, cog = 0.090,
     radius = 0.15, corner = 0.15, height = 0.29, draft = 0.187, offset = 0.076),
    (name = "float2_down", title = "Float 2, -88 mm release", file = "Fl2-0p088.txt", mass = 23.42, cog = 0.100,
     radius = 0.20, corner = 0.15, height = 0.33, draft = 0.23, offset = -0.088),
)

function in_float(p, release, base)
    z = p[3] - base
    r = hypot(p[1], p[2])
    core = release.radius - release.corner
    return z <= release.height && hypot(max(r - core, 0), max(release.corner - z, 0)) <= release.corner
end

function sample_geometry(release, constants::SimulationConstants{T}, kernel; dx, half_width, depth,
                         wall_top_z = 0.2, pause_time = 0.0, boundary_mode = NoMDBC) where {T}
    boundary_mode in (NoMDBC, SimpleMDBC, UpdatedMDBC, UpdatedMDBCWithImpedance) ||
        throw(ArgumentError("Unsupported floating boundary mode"))
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
        PauseTime = pause_time,
        Center = (0.0, 0.0, equilibrium_center_z + release.offset))
    density(points) = hydrostatic_density(points, constants; water_level = 0.0)
    boundary_geometry(points, ghosts, type, group; kwargs...) = SPHGeometry{3,T}(;
        Particles = StructArray((; Position = points, Density = T.(density(points)),
            GhostPoints = ghosts, GhostNormals = ghosts .- points)), Type = type, GroupMarker = group, kwargs...)
    tank, solid = if boundary_mode !== NoMDBC
        wall_ghosts = [SVector(ntuple(d -> d < 3 ?
            (p[d] < -half_width ? -2half_width-p[d] : p[d] > half_width ? 2half_width-p[d] : p[d]) :
            (p[d] < -depth ? -2depth-p[d] : p[d]), 3)) for p in wall]
        body_ghosts = [2float_surface_projection(p, release, base)-p for p in body]
        (boundary_geometry(wall, wall_ghosts, Fixed, 1),
            boundary_geometry(body, body_ghosts, Floating, 3; Floating = floating))
    else
        (SPHGeometry{3,T}(wall; Density = density(wall), GroupMarker = 1, Type = Fixed),
            SPHGeometry{3,T}(body; Density = density(body), GroupMarker = 3, Type = Floating, Floating = floating))
    end
    geometry = [
        tank,
        SPHGeometry{3,T}(water; Density = density(water), GroupMarker = 2, Type = Fluid),
        solid,
    ]
    return geometry, equilibrium_center_z
end

"""Nearest physical surface of the complete axisymmetric float, including its top cap."""
function float_surface_projection(p, release, base)
    r = hypot(p[1], p[2])
    core = release.radius - release.corner
    center_z, top = base + release.corner, base + release.height
    candidates = [SVector(r, top), SVector(release.radius, clamp(p[3], center_z, top))]
    if r <= core
        push!(candidates, SVector(r, base))
    else
        offset = SVector(r - core, min(p[3] - center_z, 0.))
        direction = iszero(offset) ? SVector(1., 0.) : offset / norm(offset)
        push!(candidates, SVector(core + release.corner * direction[1], center_z + release.corner * direction[2]))
    end
    q = candidates[argmin(norm.(candidates .- Ref(SVector(r, p[3]))))]
    direction = iszero(r) ? SVector(1., 0.) : SVector(p[1], p[2]) / r
    return SVector(q[1] * direction[1], q[1] * direction[2], q[2])
end

function run_case(release, case; output, vertical_axis = 3)
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
    heave = 1000 .* (getproperty(records, Symbol("Center:", vertical_axis - 1)) .- case.equilibrium_center_z)
    time = records.Time
    length(time) > 1 && all(>(0), diff(time)) && isapprox(last(time), duration; atol = 1e-6) ||
        error("Incomplete or invalid floating-body history")
    isapprox(first(heave), 1000release.offset; atol = 1e-4) || error("Wrong initial heave datum")
    mkpath(output)
    CSV.write(joinpath(output, release.name * "_body.csv"), records)
    history = joinpath(output, release.name * ".csv")
    CSV.write(history, (; time, heave_mm = heave,
        surge_mm = 1000 .* getproperty(records, Symbol("Center:0")),
        sway_mm = vertical_axis == 3 ? 1000 .* getproperty(records, Symbol("Center:1")) : zeros(length(time))))
    open(joinpath(output, release.name * ".toml"), "w") do io
        config = Dict("dx" => dx, "half_width" => half_width, "depth" => depth, "duration" => duration,
            "dimensions" => vertical_axis, "kernel" => string(nameof(typeof(case.kernel.kernel))), "h" => Float64(case.kernel.h),
            "kernel_support_factor" => Float64(case.kernel.k),
            "mass" => release.mass, "draft" => release.draft, "offset" => release.offset,
            "fluid_particles" => length(case.geometry[2].Particles), "frames" => length(time),
            "steps" => result.performance.steps, "wall_seconds" => result.performance.wall_time_seconds)
        hasproperty(case, :line_mass) && (config["mass_per_span_kg_m"] = case.line_mass)
        hasproperty(case, :boundary_mode) && (config["boundary_mode"] = case.boundary_mode)
        merge!(config, Dict("sound_speed_m_s" => Float64(case.constants.c₀), "cfl" => Float64(case.constants.CFL),
            "viscosity" => string(nameof(typeof(case.viscosity))), "density_diffusion" => string(nameof(typeof(case.diffusion))),
            "kinematic_viscosity_m2_s" => Float64(case.constants.ν₀), "density_diffusion_coefficient" => Float64(case.constants.δᵩ),
            "boundary_particles" => length(case.geometry[1].Particles), "floating_particles" => length(case.geometry[3].Particles),
            "complete" => result.final_state.complete, "frozen_particles" => result.diagnostics.frozen_particles))
        for key in (:sponge_width, :sponge_rate, :pause_time)
            hasproperty(case, key) && (config[string(key)] = getproperty(case, key))
        end
        TOML.print(io, config)
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
function write_heave_state(release, output; layout = "side-by-side", screenshot = false, planar = false)
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
        planar=$(planar ? "True" : "False"),
    )
    runpy.run_path($(python_path(script)))["build_view"](args)
    """)
    return state
end
