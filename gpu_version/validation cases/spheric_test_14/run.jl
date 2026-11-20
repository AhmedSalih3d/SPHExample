using SPHExampleGPU, StaticArrays, DelimitedFiles, CSV, TOML

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

function setup(release; dx = 0.025, half_width = 1.2, duration = 4.5, save)
    depth = 0.8
    T = Float32
    duration > 0 || throw(ArgumentError("Duration must be positive"))
    constants = SimulationConstants{T}(3; dx, g = 9.81, c₀ = 25.0, CFL = 0.2, α = 0.01, δᵩ = 0.1)
    kernel = SPHKernelInstance{3,T}(WendlandC2(); h = T(1.3dx))
    geometry, equilibrium_center_z = sample_geometry(release, constants, kernel; dx, half_width, depth)
    damping = DampingCylinder((0.0, 0.0, -depth), (0.0, 0.0, 0.2),
        half_width - 0.5, half_width; rate = 12)
    metadata = SimulationMetaData{3,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(;
        SimulationName = "FreeHeave", SaveLocation = save, SimulationTime = duration,
        OutputTimes = 0.01, ExportSingleVTKHDF = true, OutputVariables = ["Type"],
        VisualizeInParaview = false, OpenLogFile = false, GPUDoublePosition = true,
        GPUClipTimeStepToOutput = true)
    return (; geometry, metadata, constants, kernel, damping, equilibrium_center_z, dx, half_width, depth, duration,
        viscosity = ArtificialViscosity(), diffusion = FourtakasDensityDiffusion(), stepping = SymplecticTimeStepping())
end

function run_case(release; dx, half_width, duration)
    tag = "dp$(dx)_half$(half_width)_t$(duration)"
    save = joinpath(@__DIR__, "runs", tag, release.name)
    output = joinpath(@__DIR__, "results", tag)
    return run_case(release, setup(release; dx, half_width, duration, save); output)
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
            "mass" => release.mass, "draft" => release.draft, "offset" => release.offset,
            "fluid_particles" => length(case.geometry[2].Particles), "frames" => length(time),
            "steps" => result.performance.steps, "wall_seconds" => result.performance.wall_time_seconds))
    end
    return (; particles = joinpath(save, "FreeHeave.vtkhdf"), history, reference = reference_path)
end

function write_heave_view(pvpython, release, output; layout = "side-by-side", screenshot = false)
    (; particles, history, reference) = output
    script = joinpath(@__DIR__, "paraview_heave.py")
    state = joinpath(dirname(particles), "FreeHeave_Comparison.pvsm")
    args = screenshot ? ["--screenshot", replace(state, ".pvsm" => ".png")] : String[]
    run(`$pvpython --no-mpi --force-offscreen-rendering $script $particles $history $reference --state $state --title $(release.title) --layout $layout $args`)
    isfile(state) || error("ParaView did not write $state")
    return state
end

function main(args)
    visualize = "--no-paraview" ∉ args
    positional = filter(!=("--no-paraview"), args)
    length(positional) <= 4 || error("Usage: run.jl [dx] [tank half-width] [duration] [release] [--no-paraview]")
    dx, half_width, duration = (i <= length(positional) ? parse(Float64, positional[i]) : value
        for (i, value) in enumerate((0.025, 1.2, 4.5)))
    cases = length(positional) == 4 ? filter(r -> r.name == positional[4], RELEASES) : RELEASES
    isempty(cases) && error("Release must be float1_down, float1_up or float2_down")

    pvpython = visualize ? Sys.which("pvpython") : nothing
    visualize && pvpython === nothing &&
        @warn "pvpython was not found on PATH. Add the ParaView bin directory to PATH to generate the heave view."
    first_state = nothing
    for release in cases
        output = run_case(release; dx, half_width, duration)
        pvpython === nothing && continue
        try
            state = write_heave_view(pvpython, release, output)
            println("ParaView state saved to ", state)
            first_state === nothing && (first_state = state)
        catch exception
            exception isa InterruptException && rethrow()
            @warn "Could not create the combined ParaView view for $(release.name)" exception
        end
    end
    if first_state !== nothing
        paraview = Sys.which("paraview")
        if paraview === nothing
            @warn "ParaView was not found on PATH. Open $first_state with File > Load State."
        else
            try
                run(`$paraview --state=$first_state`; wait = false)
            catch exception
                exception isa InterruptException && rethrow()
                @warn "Could not open ParaView. Open $first_state with File > Load State." exception
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
