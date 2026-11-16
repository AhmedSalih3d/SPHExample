module SphericTest06

using SPHExampleGPU
using StaticArrays: SVector
using LinearAlgebra: dot, norm
using HDF5
using CSV
using Printf
using SHA
using TOML
using CUDA

export read_reference, verify_reference, body_trajectory, build_case, run_case,
    pair_loads, frame_loads, analyze_history, compare_histories, bin_average,
    REFERENCE_DIR, CASE_DIR

const CASE_DIR = @__DIR__
const REFERENCE_DIR = joinpath(CASE_DIR, "reference")
const REFERENCE_FILES = ("Motion_Body.dat", "Force_Re050.dat", "Force_Re100.dat", "Force_Re150.dat")
const DEFAULT_LAYOUT = (;
    tank_origin = (0.0, 0.0),
    tank_size = (10.0, 5.0),
    body_center = (1.5, 2.5),
    body_size = (1.0, 1.0),
)

"""Read an official Tecplot-style numeric table, preserving its original units."""
function read_reference(path; columns = startswith(basename(path), "Motion") ? 4 : 3)
    rows = Vector{Float64}[]
    for line in eachline(path)
        stripped = strip(line)
        isempty(stripped) && continue
        if !occursin(r"^[+-]?[0-9.]", stripped)
            isempty(rows) || error("Unexpected text after numeric data in $path")
            continue
        end
        values = parse.(Float64, split(stripped))
        length(values) == columns && all(isfinite, values) || error("Invalid reference row in $path")
        push!(rows, values)
    end
    length(rows) > 1 || error("Reference table is empty: $path")
    table = permutedims(reduce(hcat, rows))
    all(>(0), diff(table[:, 1])) || error("Reference times must increase strictly: $path")
    return table
end

"""Reject changed or incomplete reference fixtures before running a comparison."""
function verify_reference(directory = REFERENCE_DIR)
    hashes = TOML.parsefile(joinpath(directory, "sha256.toml"))
    for name in REFERENCE_FILES
        path = joinpath(directory, name)
        bytes2hex(sha256(read(path))) == hashes[name] || error("Reference checksum mismatch: $name")
        read_reference(path)
    end
    return true
end

"""Scale the official motion; `motion_kind=:integrated` avoids rounded-position acceleration jitter."""
function body_trajectory(directory = REFERENCE_DIR; speed_scale = 1.0, start_delay = 0.0, motion_kind = :hermite)
    motion_kind in (:hermite, :integrated) || throw(ArgumentError("motion_kind must be :hermite or :integrated"))
    isfinite(speed_scale) && speed_scale > 0 ||
        throw(ArgumentError("speed_scale must be positive and finite"))
    isfinite(start_delay) && start_delay >= 0 ||
        throw(ArgumentError("start_delay must be finite and nonnegative"))
    data = read_reference(joinpath(directory, "Motion_Body.dat"))
    if motion_kind === :integrated
        # Integrate velocity rather than differentiating rounded displacement records.
        data[1, 4] = 1.5
        for i in 2:size(data, 1)
            data[i, 4] = data[i-1, 4] + (data[i-1, 3] + data[i, 3]) * (data[i, 1] - data[i-1, 1]) / 2
        end
    end
    times = data[:, 1] .+ start_delay
    positions = [SVector(speed_scale * (x - 1.5), 0.0) for x in data[:, 4]]
    velocities = [SVector(speed_scale * u, 0.0) for u in data[:, 3]]
    if start_delay > 0
        pushfirst!(times, 0.0)
        pushfirst!(positions, zero(first(positions)))
        pushfirst!(velocities, zero(first(velocities)))
    end
    return SampledTrajectory(times, positions; velocities)
end

"""Construct DBC or SimpleMDBC tank/body geometry, reflecting ghost points into fluid."""
function manual_geometry(sampled, density, ::Type{T}; boundary_mode = NoMDBC) where {T}
    boundary_mode in (NoMDBC, SimpleMDBC) || throw(ArgumentError("Manual case supports NoMDBC or SimpleMDBC"))
    layout = sampled.layout
    lower, upper = layout.tank_origin, layout.tank_origin + layout.tank_size
    function wall_geometry(points, marker, type)
        boundary_mode === NoMDBC && return SPHGeometry{2,T}(points; Density = density, GroupMarker = marker, Type = type)
        offsets = map(points) do p
            if type == Fixed
                return SVector(ntuple(d -> p[d] < lower[d] ? 2(lower[d] - p[d]) :
                    p[d] > upper[d] ? 2(upper[d] - p[d]) : 0.0, 2))
            end
            local_p = p - layout.body_center
            gap = layout.body_size / 2 - abs.(local_p)
            # Reflect tied corner distances along both normals to preserve symmetry.
            return SVector(ntuple(d -> isapprox(gap[d], minimum(gap); atol = 1e-12, rtol = 0) ?
                2 * (local_p[d] >= 0 ? 1 : -1) * gap[d] : 0.0, 2))
        end
        particles = particle_struct_array(points, density;
            GhostPoints = points .+ offsets, GhostNormals = SVector{2,T}.(offsets))
        return SPHGeometry{2,T}(; Particles = particles, GroupMarker = marker, Type = type)
    end
    return [wall_geometry(sampled.walls, 1, Fixed),
        SPHGeometry{2,T}(sampled.fluid; Density = density, GroupMarker = 2, Type = Fluid),
        wall_geometry(sampled.body, 3, Moving)]
end

"""A quadratic fluid-velocity sponge at all four tank edges; corners use the maximum ramp."""
function edge_sponge(layout = DEFAULT_LAYOUT; width, rate = 100.0)
    origin = point2(layout.tank_origin, "layout.tank_origin")
    size = point2(layout.tank_size, "layout.tank_size")
    all(>(0), size) && isfinite(width) && 0 < 2width < minimum(size) ||
        throw(ArgumentError("Sponge width must be positive and less than half the tank size"))
    isfinite(rate) && rate >= 0 || throw(ArgumentError("Sponge rate must be finite and nonnegative"))
    upper = origin + size
    return DampingBox((Tuple(origin .+ width), Tuple(upper .- width)), (Tuple(origin), Tuple(upper));
        overlimit = 0.0, rate, factors = (1.0, 1.0), directions = :all)
end

function point2(value, name)
    length(value) == 2 || throw(ArgumentError("$name must have two coordinates"))
    point = SVector{2,Float64}(value)
    all(isfinite, point) || throw(ArgumentError("$name coordinates must be finite"))
    return point
end

"""Sample the tank, water and square on the benchmark's cell-centered lattice."""
function sample_case_particles(layout, dx, kernel_instance; wall_layers = nothing)
    isfinite(dx) && 0 < dx <= 0.1 || throw(ArgumentError("dx must lie in (0, 0.1]"))
    layout isa NamedTuple || throw(ArgumentError("layout must be a named tuple"))
    all(name -> hasproperty(layout, name),
        (:tank_origin, :tank_size, :body_center, :body_size)) ||
        throw(ArgumentError("layout must define tank_origin, tank_size, body_center, and body_size"))
    tank_origin = point2(layout.tank_origin, "layout.tank_origin")
    tank_size = point2(layout.tank_size, "layout.tank_size")
    body_center = point2(layout.body_center, "layout.body_center")
    body_size = point2(layout.body_size, "layout.body_size")
    all(>(0), tank_size) || throw(ArgumentError("tank_size must be positive"))
    all(>(0), body_size) || throw(ArgumentError("body_size must be positive"))
    all(body_size .< tank_size) || throw(ArgumentError("body_size must be smaller than tank_size"))
    body_lower = body_center - body_size / 2
    body_upper = body_center + body_size / 2
    tank_upper = tank_origin + tank_size
    all(body_lower .>= tank_origin) && all(body_upper .<= tank_upper) ||
        throw(ArgumentError("body must lie inside the tank"))

    tank_cells = round.(Int, tank_size ./ dx)
    body_cells = round.(Int, body_size ./ dx)
    all(>(0), tank_cells) && all(>(0), body_cells) ||
        throw(ArgumentError("tank and body dimensions must contain at least one particle cell"))
    all(isapprox(tank_cells[d] * dx, tank_size[d]; atol = 1e-10, rtol = 0) for d in 1:2) ||
        throw(ArgumentError("dx must divide both tank dimensions"))
    all(isapprox(body_cells[d] * dx, body_size[d]; atol = 1e-10, rtol = 0) for d in 1:2) ||
        throw(ArgumentError("dx must divide both body dimensions"))
    body_offset = (body_lower - tank_origin) ./ dx
    body_start = round.(Int, body_offset)
    all(isapprox(body_offset[d], body_start[d]; atol = 1e-10, rtol = 0) for d in 1:2) ||
        throw(ArgumentError("body lower corner must align with the particle lattice"))

    nx, ny = tank_cells
    layers = if wall_layers === nothing
        ceil(Int, Float64(kernel_instance.H) / dx) + 1
    else
        wall_layers isa Integer && wall_layers > 0 ||
            throw(ArgumentError("wall_layers must be a positive integer"))
        Int(wall_layers)
    end
    layers * dx > kernel_instance.H ||
        throw(ArgumentError("wall_layers must extend beyond the kernel support"))
    walls, fluid, body = (SVector{2,Float64}[] for _ in 1:3)
    for j in -layers:ny+layers-1, i in -layers:nx+layers-1
        position = tank_origin + SVector((i + 0.5) * dx, (j + 0.5) * dx)
        if !(0 <= i < nx && 0 <= j < ny)
            push!(walls, position)
        elseif body_start[1] <= i < body_start[1] + body_cells[1] &&
                body_start[2] <= j < body_start[2] + body_cells[2]
            push!(body, position)
        else
            push!(fluid, position)
        end
    end
    isempty(fluid) && throw(ArgumentError("layout must leave fluid particles in the tank"))
    return (; walls, fluid, body, layers,
        layout = (; tank_origin, tank_size, body_center, body_size),
        counts = (; fixed = length(walls), fluid = length(fluid), square = length(body)))
end

"""
    build_case(; reynolds = 100, dx = 0.025, layout = DEFAULT_LAYOUT, ...)

Build the SPHERIC Test 06 moving-body geometry on a cell-centered lattice.
`layout` supplies `tank_origin`, `tank_size`, `body_center`, and `body_size`
in metres. Tank and body edges must align with `dx`; `wall_layers` must cover
the kernel support. No-shifting DBC remains the benchmark default.
"""
function build_case(; reynolds = 100, dx = 0.025, duration = 8.0, interval = 0.02,
        sound_speed = 40.0, cfl = 0.2, save = joinpath(CASE_DIR, "runs", "case"),
        float_type = Float32, graph = true, visualize_in_paraview = false,
        layout = DEFAULT_LAYOUT, density = 1000.0, gravity = 0.0,
        reference_speed = 1.0, reference_length = nothing, span = 1.0,
        density_diffusion = 0.1, artificial_viscosity_alpha = 0.0,
        background_pressure = 0.0, start_delay = 0.0, damping = nothing, divergence_cleaning = nothing,
        kernel = WendlandC2(), kernel_h_factor = sqrt(2), wall_layers = nothing,
        viscosity_model = Laminar(),
        density_diffusion_model = ZeroGravityLinearDensityDiffusion(),
        time_stepping = SymplecticTimeStepping(), shifting = NoShifting,
        gpu_clip_time_step_to_output = true, gpu_double_position = true,
        gpu_legacy_timestep = true, gpu_cell_subdivision = 1,
        gpu_x_cell_subdivision = 1, gpu_radial_interactions = false,
        gpu_interaction_fastmath = false, gpu_preload_interaction_types = false,
        gpu_boundary_forces = false, gpu_output_buffer_bytes = 16 * 2^20)
    reynolds in (50, 100, 150) || throw(ArgumentError("Reynolds number must be 50, 100, or 150"))
    isfinite(dx) && 0 < dx <= 0.1 || throw(ArgumentError("dx must lie in (0, 0.1]"))
    all(x -> isfinite(x) && x > 0,
        (duration, interval, sound_speed, cfl, reference_speed, span, kernel_h_factor)) ||
        throw(ArgumentError("duration, interval, sound speed, CFL, reference speed, span, and kernel_h_factor must be positive and finite"))
    isfinite(density) && density > 0 || throw(ArgumentError("density must be positive and finite"))
    isfinite(gravity) || throw(ArgumentError("gravity must be finite"))
    isfinite(density_diffusion) && density_diffusion >= 0 ||
        throw(ArgumentError("density_diffusion must be finite and nonnegative"))
    isfinite(artificial_viscosity_alpha) && artificial_viscosity_alpha >= 0 ||
        throw(ArgumentError("artificial_viscosity_alpha must be finite and nonnegative"))
    shifting isa Type || throw(ArgumentError("shifting must be a shifting mode type such as NoShifting"))
    isfinite(start_delay) && start_delay >= 0 ||
        throw(ArgumentError("start_delay must be finite and nonnegative"))
    isfinite(background_pressure) && background_pressure >= 0 ||
        throw(ArgumentError("background_pressure must be finite and nonnegative"))
    duration <= start_delay + 8.0 || throw(ArgumentError("The force comparison ends after 8 s of motion"))
    interval <= duration || throw(ArgumentError("Output interval exceeds duration"))

    T = float_type
    kernel_instance = SPHKernelInstance{2,T}(kernel; h = T(kernel_h_factor * dx))
    sampled = sample_case_particles(layout, dx, kernel_instance; wall_layers)
    (; tank_origin, tank_size, body_center, body_size) = sampled.layout
    (; walls, fluid, body, layers) = sampled

    characteristic_length = reference_length === nothing ? body_size[1] : Float64(reference_length)
    isfinite(characteristic_length) && characteristic_length > 0 ||
        throw(ArgumentError("reference_length must be positive and finite"))
    viscosity_model isa Union{Laminar,ZeroViscosity,ArtificialViscosity} ||
        throw(ArgumentError("Drag postprocessing supports Laminar, ZeroViscosity, or ArtificialViscosity"))
    constants = SimulationConstants{T}(2; dx, ρ₀ = density, g = gravity, c₀ = sound_speed,
        ν₀ = reference_speed * characteristic_length / reynolds,
        α = artificial_viscosity_alpha, δᵩ = density_diffusion, CFL = cfl,
        BackgroundPressure = background_pressure)
    geometry = [
        SPHGeometry{2,T}(walls; Density = constants.ρ₀, GroupMarker = 1, Type = Fixed),
        SPHGeometry{2,T}(fluid; Density = constants.ρ₀, GroupMarker = 2, Type = Fluid),
        SPHGeometry{2,T}(body; Density = constants.ρ₀, GroupMarker = 3, Type = Moving),
    ]
    times = collect(interval:interval:duration)
    isempty(times) || isapprox(last(times), duration; atol = 1e-12, rtol = 0) || push!(times, duration)
    times[end] = duration
    metadata = SimulationMetaData{2,T,shifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = "MovingSquare", SaveLocation = abspath(save), SimulationTime = duration,
        OutputTimes = times, VisualizeInParaview = visualize_in_paraview, OpenLogFile = false,
        ExportSingleVTKHDF = true, OutputVariables = ["Density", "Velocity", "Type", "GroupMarker"],
        GPUUseGraph = graph, GPUClipTimeStepToOutput = gpu_clip_time_step_to_output,
        GPUDoublePosition = gpu_double_position, GPULegacyTimestep = gpu_legacy_timestep,
        GPUCellSubdivision = gpu_cell_subdivision, GPUXCellSubdivision = gpu_x_cell_subdivision,
        GPURadialInteractions = gpu_radial_interactions,
        GPUInteractionFastMath = gpu_interaction_fastmath,
        GPUPreloadInteractionTypes = gpu_preload_interaction_types,
        GPUBoundaryForces = gpu_boundary_forces,
        GPUOutputBufferBytes = gpu_output_buffer_bytes)
    motion = PrescribedMotion(3; translation = body_trajectory(; speed_scale = reference_speed, start_delay))
    case_layout = (; tank_origin, tank_size, body_center, body_size)
    return (; constants, kernel = kernel_instance, geometry, metadata, motion, reynolds,
        dx = Float64(dx), duration = Float64(duration), interval = Float64(interval), layers,
        layout = case_layout, tank_origin, tank_size, body_center, body_size,
        body_marker = 3, reference_speed = Float64(reference_speed),
        reference_length = characteristic_length, span = Float64(span),
        viscosity_model, density_diffusion_model, time_stepping, shifting, start_delay, damping, divergence_cleaning,
        counts = sampled.counts)
end

"""Pressure and selected viscous force on body particle i from fluid particle j."""
@inline function pair_loads(position_i, position_j, velocity_i, velocity_j, density_i, density_j,
        kernel, constants; viscosity_model = Laminar())
    separation = position_i - position_j
    distance_squared = dot(separation, separation)
    if !(0 < distance_squared < kernel.H²)
        return zero(separation), zero(separation)
    end
    q = sqrt(distance_squared) * kernel.h⁻¹
    gradient = ∇Wᵢⱼ(kernel, q, separation)
    pressure_i = EquationOfStateGamma7(density_i, constants.c₀, constants.ρ₀, inv(constants.ρ₀), constants.BackgroundPressure)
    pressure_j = EquationOfStateGamma7(density_j, constants.c₀, constants.ρ₀, inv(constants.ρ₀), constants.BackgroundPressure)
    pressure_force = -constants.m₀^2 * (pressure_i + pressure_j) / (density_i * density_j) * gradient
    viscosity, _ = compute_viscosity(viscosity_model, kernel, constants, nothing, separation,
        velocity_i - velocity_j, gradient, distance_squared, density_i, density_j,
        inv(density_i), inv(density_j), 1, 2)
    wall_factor = viscosity_model isa ArtificialViscosity ? viscosity_model.boundary_factor : 1
    return pressure_force, constants.m₀ * wall_factor * viscosity
end

"""Integrate only body/fluid pairs; tank and internal solid forces are excluded."""
function frame_loads(position, velocity, density, types, markers, kernel, constants;
        body_marker = 3, viscosity_model = Laminar(), reference_speed = 1.0,
        reference_length = 1.0, span = 1.0)
    all(x -> isfinite(x) && x > 0, (reference_speed, reference_length, span)) ||
        throw(ArgumentError("force reference speed, length, and span must be positive and finite"))
    fluid_cells = Dict{Tuple{Int,Int},Vector{Int}}()
    cell(x) = (floor(Int, x[1] / kernel.H), floor(Int, x[2] / kernel.H))
    for j in eachindex(types)
        types[j] == Int(Fluid) || continue
        push!(get!(() -> Int[], fluid_cells, cell(position[j])), j)
    end
    pressure_force = zero(SVector{2,Float64})
    viscous_force = zero(pressure_force)
    for i in eachindex(types)
        markers[i] == body_marker && types[i] == Int(Moving) || continue
        ix, iy = cell(position[i])
        for cy in iy-1:iy+1, cx in ix-1:ix+1
            neighbors = get(fluid_cells, (cx, cy), nothing)
            neighbors === nothing && continue
            for j in neighbors
                pressure, viscous = pair_loads(position[i], position[j], velocity[i], velocity[j],
                    density[i], density[j], kernel, constants; viscosity_model)
                pressure_force += pressure
                viscous_force += viscous
            end
        end
    end
    # Positive drag resists the prescribed +x motion.
    scale = -2 / (constants.ρ₀ * reference_speed^2 * reference_length * span)
    return (; cd_pressure = scale * pressure_force[1], cd_viscous = scale * viscous_force[1],
        cy_pressure = scale * pressure_force[2], cy_viscous = scale * viscous_force[2])
end

"""Recover drag from accepted output states, independently of stale acceleration buffers."""
function analyze_history(path, case; coefficient_path, diagnostics_path, trajectory = body_trajectory())
    # Float64 postprocessing of saved Float32 state avoids reduction-roundoff noise.
    constants = SimulationConstants{Float64}(2; dx = case.dx, ρ₀ = case.constants.ρ₀,
        c₀ = case.constants.c₀, g = case.constants.g, ν₀ = case.constants.ν₀,
        α = case.constants.α, δᵩ = case.constants.δᵩ, BackgroundPressure = case.constants.BackgroundPressure)
    kernel = SPHKernelInstance{2,Float64}(case.kernel.kernel; h = Float64(case.kernel.h),
        k = Float64(case.kernel.k), αD = Float64(case.kernel.αD), η² = Float64(case.kernel.η²))
    body_half_size = case.body_size / 2
    tank_upper = case.tank_origin + case.tank_size
    rows, checks = NamedTuple[], NamedTuple[]
    h5open(path, "r") do file
        root = haskey(file,"VTKHDF/Particles") ? file["VTKHDF/Particles"] : file["VTKHDF"]
        times = read(root["Steps/Values"])
        offsets = read(root["Steps/PointOffsets"])
        counts = read(root["NumberOfPoints"])
        for frame in eachindex(times)
            columns = offsets[frame]+1:offsets[frame]+counts[frame]
            position = [SVector{2,Float64}(x[1], x[2]) for x in eachcol(root["Points"][:, columns])]
            velocity = [SVector{2,Float64}(v[1], v[2]) for v in eachcol(root["PointData/Velocity"][:, columns])]
            density = Float64.(root["PointData/Density"][columns])
            types = root["PointData/Type"][columns]
            markers = root["PointData/GroupMarker"][columns]
            all(isfinite, density) && all(>(0), density) && all(x -> all(isfinite, x), position) &&
                all(v -> all(isfinite, v), velocity) || error("Nonfinite state at t=$(times[frame])")
            fluid = findall(==(Int(Fluid)), types)
            square = findall(==(case.body_marker), markers)
            count(==(Int(Frozen)), types) == 0 || error("Frozen particles at t=$(times[frame])")
            length(fluid) == case.counts.fluid && length(square) == case.counts.square || error("Particle loss")
            displacement, expected_velocity, _ = trajectory_state(trajectory, times[frame])
            center = case.body_center + displacement
            motion_error = norm(sum(position[square]) / length(square) - center)
            velocity_error = maximum(v -> norm(v - expected_velocity), velocity[square])
            # A positive clearance means the particle center is inside the fluid domain.
            minimum_clearance = minimum(fluid) do i
                x, y = position[i]
                body_clearance = max(abs(x-center[1]) - body_half_size[1],
                    abs(y-center[2]) - body_half_size[2])
                min(x - case.tank_origin[1], tank_upper[1] - x,
                    y - case.tank_origin[2], tank_upper[2] - y, body_clearance)
            end
            loads = frame_loads(position, velocity, density, types, markers, kernel, constants;
                body_marker = case.body_marker, viscosity_model = case.viscosity_model,
                reference_speed = case.reference_speed, reference_length = case.reference_length,
                span = case.span)
            pressure = [EquationOfStateGamma7(density[i], constants.c₀, constants.ρ₀) for i in fluid]
            mean_pressure = sum(pressure) / length(pressure)
            start_delay = get(case, :start_delay, 0.0)
            push!(rows, (; time = times[frame], motion_time = times[frame] - start_delay, loads...))
            push!(checks, (; time = times[frame], motion_error, velocity_error,
                density_relative_max = maximum(abs, density[fluid] ./ constants.ρ₀ .- 1),
                max_speed = maximum(norm, velocity[fluid]), minimum_clearance,
                pressure_mean = mean_pressure + constants.BackgroundPressure,
                pressure_std = sqrt(sum(p -> (p - mean_pressure)^2, pressure) / length(pressure)),
                pressure_min = minimum(pressure) + constants.BackgroundPressure,
                kinetic_energy = constants.m₀ * sum(i -> sum(abs2, velocity[i]), fluid) / 2,
                fluid_particles = length(fluid), square_particles = length(square)))
        end
    end
    CSV.write(diagnostics_path, checks)
    open(coefficient_path, "w") do io
        println(io, "# time Cd_pressure Cd_viscous")
        for row in rows
            @printf(io, "%.9e %.9e %.9e\n", row.time, row.cd_pressure, row.cd_viscous)
        end
    end
    CSV.write(replace(coefficient_path, ".dat" => ".csv"), rows)
    return (; rows, checks)
end

"""Run the configured GPU case, save its force history and complete run metadata."""
function run_case(case; results_dir = joinpath(CASE_DIR, "results"))
    verify_reference()
    CUDA.functional() || error("An NVIDIA CUDA GPU is required")
    save = case.metadata.SaveLocation
    mkpath(save)
    isfile(joinpath(save, "MovingSquare.vtkhdf")) && error("Refusing to overwrite an existing run: $save")
    mkpath(results_dir)
    logger = SimulationLogger(save)
    particles = AllocateDataStructures(case.geometry, case.metadata)
    result = try
        RunSimulation(SimGeometry = case.geometry, SimMetaData = case.metadata,
            SimConstants = case.constants, SimKernel = case.kernel, SimLogger = logger,
            SimParticles = particles, SimViscosity = case.viscosity_model,
            SimDensityDiffusion = case.density_diffusion_model,
            SimTimeStepping = case.time_stepping, SimMotion = case.motion,
            SimDamping = get(case, :damping, nothing), SimDivergenceCleaning = get(case, :divergence_cleaning, nothing))
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    return save_run_results(case, result; results_dir)
end

"""Check a completed simulation and write the benchmark force history and run record."""
function save_run_results(case, result; results_dir = joinpath(CASE_DIR, "results"))
    result.final_state.complete && result.termination_reason == :completed || error("Incomplete run")
    result.diagnostics.frozen_particles == 0 || error("Run froze particles")
    save = case.metadata.SaveLocation
    mkpath(results_dir)
    n = round(Int, case.reference_length / case.dx)
    name = @sprintf("Re%03d_n%03d", case.reynolds, n)
    analysis = analyze_history(joinpath(save, "MovingSquare.vtkhdf"), case;
        coefficient_path = joinpath(results_dir, name * ".dat"),
        diagnostics_path = joinpath(results_dir, name * "_state.csv"), trajectory = case.motion.translation)
    metadata = Dict{String,Any}(
        "reynolds" => case.reynolds, "dx" => case.dx, "duration" => case.duration,
        "output_interval" => case.interval, "rho0" => Float64(case.constants.ρ₀),
        "gravity" => Float64(case.constants.g), "reference_speed" => case.reference_speed,
        "background_pressure" => Float64(case.constants.BackgroundPressure),
        "start_delay" => Float64(get(case, :start_delay, 0.0)),
        "motion_kind" => string(get(case, :motion_kind, :hermite)),
        "boundary_mode" => string(get(case, :boundary_mode, NoMDBC)),
        "divergence_cleaning" => get(case, :divergence_cleaning, nothing) !== nothing,
        "edge_sponge" => get(case, :damping, nothing) !== nothing,
        "domain_padding" => Float64(get(case, :domain_padding, 0.0)),
        "sponge_width" => Float64(get(case, :sponge_width, 0.0)),
        "sponge_rate" => Float64(get(case, :sponge_rate, 0.0)),
        "reference_length" => case.reference_length, "span" => case.span,
        "tank_origin" => collect(case.tank_origin), "tank_size" => collect(case.tank_size),
        "body_center" => collect(case.body_center), "body_size" => collect(case.body_size),
        "nu" => Float64(case.constants.ν₀), "sound_speed" => Float64(case.constants.c₀),
        "cfl" => Float64(case.constants.CFL), "density_diffusion" => Float64(case.constants.δᵩ),
        "h" => Float64(case.kernel.h), "wall_layers" => case.layers,
        "kernel" => string(typeof(case.kernel.kernel)),
        "viscosity_model" => string(typeof(case.viscosity_model)),
        "density_diffusion_model" => string(typeof(case.density_diffusion_model)),
        "time_stepping" => string(typeof(case.time_stepping)),
        "shifting" => string(case.shifting),
        "fixed_particles" => case.counts.fixed, "fluid_particles" => case.counts.fluid,
        "square_particles" => case.counts.square, "julia_version" => string(VERSION),
        "cuda_package_version" => string(pkgversion(CUDA)), "gpu" => CUDA.name(CUDA.device()),
        "cuda_driver_version" => string(CUDA.driver_version()),
        "steps" => result.performance.steps, "frames" => result.diagnostics.output_frames,
        "rebuilds" => result.diagnostics.cell_list_rebuilds,
        "wall_seconds" => result.performance.wall_time_seconds, "frozen_particles" => result.diagnostics.frozen_particles,
        "motion_error_max" => maximum(x -> x.motion_error, analysis.checks),
        "motion_velocity_error_max" => maximum(x -> x.velocity_error, analysis.checks),
        "density_relative_max" => maximum(x -> x.density_relative_max, analysis.checks),
        "mach_max" => maximum(x -> x.max_speed / case.constants.c₀, analysis.checks),
        "minimum_clearance" => minimum(x -> x.minimum_clearance, analysis.checks),
        "execution_path" => string(result.configuration.execution.path),
        "solver_commit" => strip(read(`git -C $CASE_DIR rev-parse HEAD`, String)))
    open(joinpath(results_dir, name * ".toml"), "w") do io
        TOML.print(io, metadata; sorted = true)
    end
    println("Completed ", name, ": ", result.performance.steps, " steps, ", result.performance.wall_time_seconds, " s wall time")
    return (; result, analysis, metadata)
end

"""Write a Python state that builds the combined view and exports its image inside ParaView."""
function write_force_plot_state(case; results_dir, output_dir)
    mkpath(output_dir)
    n = round(Int, case.reference_length / case.dx)
    state = joinpath(output_dir, @sprintf("ForceComparison_Re%03d.py", case.reynolds))
    plot_script = joinpath(CASE_DIR, "paraview_force_plot.py")
    simulation_state = joinpath(case.metadata.SaveLocation, "MovingSquare_SingleVTKHDFStateFile.py")
    python_path(path) = repr(replace(abspath(path), '\\' => '/'))
    write(state, """
    from pathlib import Path
    from types import SimpleNamespace
    import runpy

    args = SimpleNamespace(
        results=Path($(python_path(results_dir))),
        output=Path($(python_path(output_dir))),
        n=$n,
        component="both",
        simulation_state=Path($(python_path(simulation_state))),
    )
    plotting = runpy.run_path($(python_path(plot_script)))
    plotting["make_force_plot"]($(case.reynolds), args)
    """)
    return state
end

"""Piecewise-linear interpolation; snap roundoff-sized endpoint differences, reject extrapolation."""
function interpolate(times, values, time)
    lower, upper = first(times), last(times)
    tolerance = 8eps(max(abs(lower), abs(upper), 1.0))
    if lower - tolerance <= time < lower
        time = lower
    elseif upper < time <= upper + tolerance
        time = upper
    end
    first(times) <= time <= last(times) || throw(ArgumentError("Comparison would extrapolate at t=$time"))
    index = min(searchsortedlast(times, time), length(times)-1)
    fraction = (time-times[index]) / (times[index+1]-times[index])
    return (1-fraction)*values[index] + fraction*values[index+1]
end

"""Time integral over a bin using all knots, rather than sample-count averaging."""
function bin_average(times, values, left, right)
    right > left || throw(ArgumentError("Bin width must be positive"))
    nodes = [left; times[(times .> left) .& (times .< right)]; right]
    samples = [interpolate(times, values, time) for time in nodes]
    return sum((samples[1:end-1] + samples[2:end]) .* diff(nodes)) / (2 * (right-left))
end

"""
Compare the entire shared interval in 0.1 s time bins, suppressing the official
FD data's documented high-frequency numerical oscillations. Raw data survive.
The 10% normalized RMS and 5% resolution-change criteria are project choices,
not official SPHERIC acceptance thresholds.
"""
function compare_histories(results_dir = joinpath(CASE_DIR, "results"); bin_width = 0.1)
    verify_reference()
    metrics, binned = NamedTuple[], NamedTuple[]
    for name in sort(readdir(results_dir))
        occursin(r"^Re\d{3}_n\d+\.csv$", name) || continue
        metadata = TOML.parsefile(joinpath(results_dir, replace(name, ".csv" => ".toml")))
        data = CSV.File(joinpath(results_dir, name))
        reference = read_reference(joinpath(REFERENCE_DIR, @sprintf("Force_Re%03d.dat", metadata["reynolds"])))
        times = Float64.(data.time) .- get(metadata, "start_delay", 0.0)
        start = max(first(times), reference[1,1])
        stop = min(last(times), reference[end,1])
        edges = collect(ceil(start/bin_width)*bin_width:bin_width:floor((stop+1e-10)/bin_width)*bin_width)
        length(edges) >= 2 || continue
        for (component, column) in ((:pressure, 2), (:viscous, 3))
            values = Float64.(getproperty(data, Symbol("cd_", component)))
            predicted = [bin_average(times, values, left, right) for (left,right) in zip(edges[1:end-1], edges[2:end])]
            target = [bin_average(reference[:,1], reference[:,column], left, right) for (left,right) in zip(edges[1:end-1], edges[2:end])]
            nrmse = norm(predicted-target) / norm(target)
            late = findall(>=(4.0), edges[1:end-1])
            late_bias = isempty(late) ? NaN : (sum(predicted[late]) / sum(target[late]) - 1)
            full_interval = last(times) >= 8.0 && first(edges) <= 0.1 && last(edges) >= 8.0
            state_ok = metadata["frozen_particles"] == 0 && metadata["motion_error_max"] < 1e-7 &&
                metadata["motion_velocity_error_max"] < 1e-6 && metadata["density_relative_max"] < 0.01 &&
                metadata["minimum_clearance"] >= 0 && metadata["mach_max"] < 0.1
            push!(metrics, (; reynolds = metadata["reynolds"], dx = metadata["dx"], component,
                start = first(edges), stop = last(edges), bins = length(target), nrmse,
                late_mean_relative_bias = late_bias, full_interval, state_ok,
                drag_pass = full_interval && nrmse <= 0.10))
            for k in eachindex(target)
                push!(binned, (; reynolds = metadata["reynolds"], dx = metadata["dx"], component,
                    time = (edges[k]+edges[k+1])/2, prediction = predicted[k], reference = target[k]))
            end
        end
    end
    isempty(metrics) && error("No completed coefficient histories in $results_dir")
    CSV.write(joinpath(results_dir, "metrics.csv"), metrics)
    CSV.write(joinpath(results_dir, "binned.csv"), binned)
    resolution = NamedTuple[]
    for reynolds in (50,100,150), component in (:pressure,:viscous)
        spacings = sort(unique(row.dx for row in metrics if row.reynolds == reynolds && row.component == component))
        length(spacings) >= 2 || continue
        fine = filter(row -> row.reynolds == reynolds && row.component == component && row.dx == spacings[1], binned)
        coarse = filter(row -> row.reynolds == reynolds && row.component == component && row.dx == spacings[2], binned)
        getproperty.(fine, :time) == getproperty.(coarse, :time) || error("Resolution comparison has different time bins")
        change = norm(getproperty.(fine,:prediction)-getproperty.(coarse,:prediction)) / norm(getproperty.(fine,:reference))
        push!(resolution, (; reynolds, component, fine_dx = spacings[1], coarse_dx = spacings[2],
            relative_rms_change = change, resolution_pass = change <= 0.05))
    end
    isempty(resolution) || CSV.write(joinpath(results_dir, "resolution.csv"), resolution)
    return (; metrics, resolution)
end

end
