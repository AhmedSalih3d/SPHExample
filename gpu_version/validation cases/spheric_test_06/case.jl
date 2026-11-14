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

"""Absolute tabulated square-center x is converted to translation from x = 1.5."""
function body_trajectory(directory = REFERENCE_DIR)
    data = read_reference(joinpath(directory, "Motion_Body.dat"))
    return SampledTrajectory(data[:, 1], [SVector(x - 1.5, 0.0) for x in data[:, 4]];
        velocities = [SVector(u, 0.0) for u in data[:, 3]])
end

"""
    build_case(; reynolds = 100, dx = 0.025, duration = 8, interval = 0.02, save, ...)

SPHERIC Test 06 in SI units with unit transverse depth: a 10 × 5 tank,
unit square at [1,2] × [2,3], zero gravity, and nu = Umax L / Re.
Cell-centered particles put the physical wall halfway between solid and fluid
rows. Four solid layers cover the Wendland support; the square is filled.
The existing DBC wall treatment approximates the specified no-slip walls.
"""
function build_case(; reynolds = 100, dx = 0.025, duration = 8.0, interval = 0.02,
        sound_speed = 40.0, cfl = 0.2, save = joinpath(CASE_DIR, "runs", "case"),
        float_type = Float32, graph = true)
    reynolds in (50, 100, 150) || throw(ArgumentError("Reynolds number must be 50, 100, or 150"))
    isfinite(dx) && 0 < dx <= 0.1 || throw(ArgumentError("dx must lie in (0, 0.1]"))
    all(x -> isfinite(x) && x > 0, (duration, interval, sound_speed, cfl)) ||
        throw(ArgumentError("duration, interval, sound speed and CFL must be positive and finite"))
    duration <= 8.0 || throw(ArgumentError("The force comparison ends at t = 8 s"))
    interval <= duration || throw(ArgumentError("Output interval exceeds duration"))
    nx, ny, ns = round.(Int, (10 / dx, 5 / dx, 1 / dx))
    all(isapprox(n * dx, length; atol = 1e-10, rtol = 0) for (n, length) in ((nx, 10), (ny, 5), (ns, 1))) ||
        throw(ArgumentError("dx must divide the unit square exactly"))
    T = float_type
    constants = SimulationConstants{T}(2; dx, ρ₀ = 1000, g = 0, c₀ = sound_speed,
        ν₀ = 1 / reynolds, α = 0, δᵩ = 0.1, CFL = cfl)
    kernel = SPHKernelInstance{2,T}(WendlandC2(); h = T(sqrt(2) * dx))
    layers = ceil(Int, Float64(kernel.H) / dx) + 1
    walls, fluid, body = (SVector{2,Float64}[] for _ in 1:3)
    for j in -layers:ny+layers-1, i in -layers:nx+layers-1
        position = SVector((i + 0.5) * dx, (j + 0.5) * dx)
        if !(0 <= i < nx && 0 <= j < ny)
            push!(walls, position)
        elseif ns <= i < 2ns && 2ns <= j < 3ns
            push!(body, position)
        else
            push!(fluid, position)
        end
    end
    geometry = [
        SPHGeometry{2,T}(walls; Density = constants.ρ₀, GroupMarker = 1, Type = Fixed),
        SPHGeometry{2,T}(fluid; Density = constants.ρ₀, GroupMarker = 2, Type = Fluid),
        SPHGeometry{2,T}(body; Density = constants.ρ₀, GroupMarker = 3, Type = Moving),
    ]
    times = collect(interval:interval:duration)
    isempty(times) || isapprox(last(times), duration; atol = 1e-12, rtol = 0) || push!(times, duration)
    times[end] = duration
    metadata = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = "MovingSquare", SaveLocation = abspath(save), SimulationTime = duration,
        OutputTimes = times, VisualizeInParaview = false, OpenLogFile = false,
        ExportSingleVTKHDF = true, OutputVariables = ["Density", "Velocity", "Type", "GroupMarker"],
        GPUUseGraph = graph, GPUClipTimeStepToOutput = true, GPUDoublePosition = true,
        GPUCellSubdivision = 1, GPUXCellSubdivision = 1, GPURadialInteractions = false,
        GPUInteractionFastMath = false, GPUPreloadInteractionTypes = false,
        GPUBoundaryForces = false, GPUOutputBufferBytes = 16 * 2^20)
    motion = PrescribedMotion(3; translation = body_trajectory())
    return (; constants, kernel, geometry, metadata, motion, reynolds, dx = Float64(dx),
        duration = Float64(duration), interval = Float64(interval), layers,
        counts = (; fixed = length(walls), fluid = length(fluid), square = length(body)))
end

"""Pressure and laminar force on solid i from fluid j (equal particle masses)."""
@inline function pair_loads(position_i, position_j, velocity_i, velocity_j, density_i, density_j, kernel, constants)
    separation = position_i - position_j
    distance_squared = dot(separation, separation)
    if !(0 < distance_squared < kernel.H²)
        return zero(separation), zero(separation)
    end
    q = sqrt(distance_squared) * kernel.h⁻¹
    gradient = ∇Wᵢⱼ(kernel, q, separation)
    pressure_i = EquationOfStateGamma7(density_i, constants.c₀, constants.ρ₀)
    pressure_j = EquationOfStateGamma7(density_j, constants.c₀, constants.ρ₀)
    pressure_force = -constants.m₀^2 * (pressure_i + pressure_j) / (density_i * density_j) * gradient
    viscosity, _ = compute_viscosity(Laminar(), kernel, constants, nothing, separation,
        velocity_i - velocity_j, gradient, distance_squared, density_i, density_j,
        inv(density_i), inv(density_j), 1, 2)
    return pressure_force, constants.m₀ * viscosity
end

"""Integrate only square/fluid pairs; tank and internal solid forces are excluded."""
function frame_loads(position, velocity, density, types, markers, kernel, constants)
    fluid_cells = Dict{Tuple{Int,Int},Vector{Int}}()
    cell(x) = (floor(Int, x[1] / kernel.H), floor(Int, x[2] / kernel.H))
    for j in eachindex(types)
        types[j] == Int(Fluid) || continue
        push!(get!(() -> Int[], fluid_cells, cell(position[j])), j)
    end
    pressure_force = zero(SVector{2,Float64})
    viscous_force = zero(pressure_force)
    for i in eachindex(types)
        markers[i] == 3 && types[i] == Int(Moving) || continue
        ix, iy = cell(position[i])
        for cy in iy-1:iy+1, cx in ix-1:ix+1
            neighbors = get(fluid_cells, (cx, cy), nothing)
            neighbors === nothing && continue
            for j in neighbors
                pressure, viscous = pair_loads(position[i], position[j], velocity[i], velocity[j],
                    density[i], density[j], kernel, constants)
                pressure_force += pressure
                viscous_force += viscous
            end
        end
    end
    # Positive drag resists the prescribed +x motion. rho Umax^2 L b / 2 = rho / 2.
    scale = -2 / constants.ρ₀
    return (; cd_pressure = scale * pressure_force[1], cd_viscous = scale * viscous_force[1],
        cy_pressure = scale * pressure_force[2], cy_viscous = scale * viscous_force[2])
end

"""Recover drag from accepted output states, independently of stale acceleration buffers."""
function analyze_history(path, case; coefficient_path, diagnostics_path)
    trajectory = body_trajectory()
    # Float64 postprocessing of saved Float32 state avoids reduction-roundoff noise.
    constants = SimulationConstants{Float64}(2; dx = case.dx, ρ₀ = case.constants.ρ₀,
        c₀ = case.constants.c₀, g = 0, ν₀ = case.constants.ν₀)
    kernel = SPHKernelInstance{2,Float64}(WendlandC2(); h = Float64(case.kernel.h))
    rows, checks = NamedTuple[], NamedTuple[]
    h5open(path, "r") do file
        root = file["VTKHDF"]
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
            square = findall(==(3), markers)
            count(==(Int(Frozen)), types) == 0 || error("Frozen particles at t=$(times[frame])")
            length(fluid) == case.counts.fluid && length(square) == case.counts.square || error("Particle loss")
            displacement, expected_velocity, _ = trajectory_state(trajectory, times[frame])
            center = SVector(1.5, 2.5) + displacement
            motion_error = norm(sum(position[square]) / length(square) - center)
            velocity_error = maximum(v -> norm(v - expected_velocity), velocity[square])
            # A positive clearance means the particle center is inside the fluid domain.
            minimum_clearance = minimum(fluid) do i
                x, y = position[i]
                square_clearance = max(abs(x-center[1]), abs(y-center[2])) - 0.5
                min(x, 10-x, y, 5-y, square_clearance)
            end
            loads = frame_loads(position, velocity, density, types, markers, kernel, constants)
            push!(rows, (; time = times[frame], loads...))
            push!(checks, (; time = times[frame], motion_error, velocity_error,
                density_relative_max = maximum(abs, density[fluid] ./ constants.ρ₀ .- 1),
                max_speed = maximum(norm, velocity[fluid]), minimum_clearance,
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

"""Run the unmodified GPU solver, save the force history and complete run metadata."""
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
            SimParticles = particles, SimViscosity = Laminar(),
            SimDensityDiffusion = ZeroGravityLinearDensityDiffusion(),
            SimTimeStepping = SymplecticTimeStepping(), SimMotion = case.motion)
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    result.final_state.complete && result.termination_reason == :completed || error("Incomplete run")
    result.diagnostics.frozen_particles == 0 || error("Run froze particles")
    name = @sprintf("Re%03d_n%03d", case.reynolds, round(Int, 1 / case.dx))
    analysis = analyze_history(joinpath(save, "MovingSquare.vtkhdf"), case;
        coefficient_path = joinpath(results_dir, name * ".dat"),
        diagnostics_path = joinpath(results_dir, name * "_state.csv"))
    metadata = Dict{String,Any}(
        "reynolds" => case.reynolds, "dx" => case.dx, "duration" => case.duration,
        "output_interval" => case.interval, "rho0" => Float64(case.constants.ρ₀),
        "nu" => Float64(case.constants.ν₀), "sound_speed" => Float64(case.constants.c₀),
        "cfl" => Float64(case.constants.CFL), "density_diffusion" => Float64(case.constants.δᵩ),
        "h" => Float64(case.kernel.h), "wall_layers" => case.layers,
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

"""Piecewise-linear interpolation; extrapolation is deliberately rejected."""
function interpolate(times, values, time)
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
        times = Float64.(data.time)
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
