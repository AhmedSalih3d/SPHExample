# DualSPHysics 5.4 examples/main/04_ExternalForces, generated entirely in Julia.
# julia --project=gpu_version gpu_version/example/ExternalForces.jl [output_dir] [dx] [duration]
using SPHExampleGPU
using StaticArrays: SVector

"""
    external_forces_3d_geometry(; dx = 0.02, FloatType = Float32)

Generate a one-metre cubic container, a central fixed sphere of radius 0.1 m,
and the two lower water blocks from the DualSPHysics external-forces case.
The container uses three particle layers outside its inner faces. Both fluid
groups have the reference's upward acceleration ramp followed by oscillatory
angular acceleration about y. No CSV files or external installation are needed.
"""
function external_forces_3d_geometry(; dx = 0.02, FloatType = Float32)
    isfinite(dx) && 0 < dx <= 0.05 || throw(ArgumentError("dx must be finite and in (0, 0.05] to resolve the sphere"))
    T = FloatType
    V = SVector{3,Float64}
    walls, sphere, left, right = V[], V[], V[], V[]
    extent = ceil(Int, 0.5 / dx) + 2
    tolerance = 1e-8 * dx
    for k in -extent:extent, j in -extent:extent, i in -extent:extent
        p = V(i * dx, j * dx, k * dx)
        if maximum(abs, p) >= 0.5 - tolerance
            push!(walls, p)
        elseif sum(abs2, p) <= 0.1^2 + tolerance^2
            push!(sphere, p)
        elseif -0.482 - tolerance <= p[2] <= 0.482 + tolerance && -0.482 - tolerance <= p[3] <= -0.1607 + tolerance
            -0.482 - tolerance <= p[1] <= -0.1607 + tolerance && push!(left, p)
            0.1607 - tolerance <= p[1] <= 0.482 + tolerance && push!(right, p)
        end
    end
    water_level = maximum(last, left)
    depth = water_level - minimum(last, left)
    constants = SimulationConstants{T}(3; dx, g = 9.81, c₀ = 40sqrt(9.81depth),
        CFL = 0.25, α = 0.1, δᵩ = 0.1)
    geometry = [SPHGeometry{3,T}(points;
        Density = hydrostatic_density(points, constants; water_level), GroupMarker = marker,
        Type = marker <= 2 ? Fixed : Fluid) for (marker, points) in enumerate((walls, sphere, left, right))]
    times = [0.0, 1.0, 3.0, [3 + k / 10 for k in 1:20]...]
    linear = [SVector(0.0, 0.0, k == 2 ? 15.0 : 0.0) for k in eachindex(times)]
    angular = [SVector(0.0, k <= 3 || k == length(times) ? 0.0 : 20sinpi((k - 3) / 10), 0.0) for k in eachindex(times)]
    forces = Tuple(ExternalForce{3,T}(marker; times, linear_acceleration = linear,
        angular_acceleration = angular) for marker in (3, 4))
    return (; geometry, constants, forces)
end

"""
    run_external_forces_3d(; dx = 0.02, simulation_time = 5, ...)

Run the generated external-forces case with symplectic stepping and Wendland C2.
Returns a `SimulationResult`; the last two seconds exercise the angular terms.
"""
function run_external_forces_3d(; FloatType = Float32, dx = 0.02, simulation_time = 5.0,
        output_interval = 0.02, visualize = true, open_log_file = true, use_graph = true,
        save_location = joinpath(@__DIR__, "..", "output", "external_forces_3d"))
    isfinite(simulation_time) && simulation_time > 0 || throw(ArgumentError("simulation_time must be finite and positive"))
    isfinite(output_interval) && output_interval > 0 || throw(ArgumentError("output_interval must be finite and positive"))
    case = external_forces_3d_geometry(; dx, FloatType)
    T = FloatType
    mkpath(save_location)
    meta = SimulationMetaData{3,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = "ExternalForces3D", SaveLocation = save_location,
        SimulationTime = simulation_time, OutputTimes = output_interval,
        VisualizeInParaview = visualize, OpenLogFile = open_log_file,
        GPUUseGraph = use_graph, GPUCellSubdivision = 1, ExportGridCells = true)
    particles = AllocateDataStructures(case.geometry, meta)
    logger = SimulationLogger(save_location)
    try
        return RunSimulation(SimGeometry = case.geometry, SimMetaData = meta, SimParticles = particles,
            SimConstants = case.constants, SimKernel = SPHKernelInstance{3,T}(WendlandC2(); h = T(0.8sqrt(3) * dx)),
            SimLogger = logger, SimTimeStepping = SymplecticTimeStepping(),
            SimViscosity = ArtificialViscosity(), SimDensityDiffusion = LinearDensityDiffusion(),
            SimExternalForces = case.forces)
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end

run_external_forces_3d(;
    save_location = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "output", "external_forces_3d"),
    dx = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.02,
    simulation_time = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 5.0)
