# DualSPHysics 5.4 examples/main/05_SloshingTank, fixed-tank acceleration case.
# julia --project=gpu_version gpu_version/example/SloshingTank.jl [output_dir] [dx] [duration]
using SPHExampleGPU

"""
    sloshing_tank_geometry(; dx = 0.002, FloatType = Float32)

Generate a closed 0.9 m by 0.508 m tank and a 0.093 m deep water layer in
the reference's x/z plane, mapped to x/y. Three particle layers form the walls
at the default spacing. Geometry and hydrostatic densities are generated in
memory; only the measured excitation is read from the bundled data file.
"""
function sloshing_tank_geometry(; dx = 0.002, FloatType = Float32)
    isfinite(dx) && 0 < dx <= 0.02 || throw(ArgumentError("dx must be finite and in (0, 0.02] to resolve the water layer"))
    T = FloatType
    thickness = 2dx
    tank = (
        rectangle((-0.45 - thickness, -thickness), 0.9 + 2thickness, thickness),
        rectangle((-0.45 - thickness, 0.0), thickness, 0.508),
        rectangle((0.45, 0.0), thickness, 0.508),
        rectangle((-0.45 - thickness, 0.508), 0.9 + 2thickness, thickness),
    )
    water = rectangle((-0.45, 0.0), 0.9, 0.093)
    regions = [ParticleRegion("Tank", tank, Fixed), ParticleRegion("Water", water, Fluid)]
    boundary, fluid = sample_particles(regions, dx)
    # GenCase resolves automatic depth from the sampled fluid extrema (0.09 m at dx=0.002).
    water_level = maximum(last, fluid.positions)
    depth = water_level - minimum(last, fluid.positions)
    constants = SimulationConstants{T}(2; dx, g = 9.81, c₀ = 30sqrt(9.81 * depth),
        CFL = 0.2, α = 0.05, δᵩ = 0.1)
    geometry = [
        SPHGeometry{2,T}(boundary.positions; Density = constants.ρ₀, GroupMarker = 1, Type = Fixed),
        SPHGeometry{2,T}(fluid.positions;
            Density = hydrostatic_density(fluid.positions, constants; water_level),
            GroupMarker = 2, Type = Fluid),
    ]
    kernel = SPHKernelInstance{2,T}(WendlandC2(); h = T(0.91924sqrt(2) * dx))
    return (; geometry, constants, kernel, tank, water, water_level)
end

"""
    sloshing_tank_force(FloatType = Float32; acceleration_path, center = (0.45, 0))

Load the unresampled DualSPHysics acceleration history (0–8.35 s) for fluid
marker 2. Linear input already includes rotated gravity, so the force cancels
global gravity. Angular y in the reference becomes clockwise rotation in 2D.
The default center follows `CaseSloshingAcc_Def.xml`; its accompanying prose
instead describes the tank-bottom midpoint `(0, 0)`. Pass `center` explicitly
to study that alternative without silently changing the supplied case.
"""
function sloshing_tank_force(::Type{T} = Float32;
        acceleration_path = joinpath(@__DIR__, "data", "sloshing_tank", "CaseSloshingAccData.csv"),
        center = (0.45, 0.0)) where {T<:AbstractFloat}
    return ExternalForce{2,T}(2, acceleration_path; center, global_gravity = false)
end

"""
    run_sloshing_tank(; dx = 0.002, simulation_time = 8.35, ...)

Run the generated fixed-tank sloshing case using the full measured linear and
angular forcing, symplectic stepping, Wendland C2 and complex density diffusion.
Pressure at the reference's inner probe `(-0.4461, 0.093)` and three water-column
heights are saved with the particle output. Set `measurements = false` to omit
these probes. Pressure sampling uses the nearest fluid particle, rather than
the DualSPHysics MeasureTool interpolation. Returns a `SimulationResult`.
"""
function run_sloshing_tank(; FloatType = Float32, dx = 0.002, simulation_time = 8.35,
        output_interval = 0.01, visualize = true, open_log_file = true,
        use_graph = true, double_position = false, measurements = true,
        time_stepping::TimeSteppingMode = SymplecticTimeStepping(),
        acceleration_path = joinpath(@__DIR__, "data", "sloshing_tank", "CaseSloshingAccData.csv"),
        force_center = (0.45, 0.0),
        save_location = joinpath(@__DIR__, "..", "output", "sloshing_tank"))
    isfinite(simulation_time) && simulation_time > 0 || throw(ArgumentError("simulation_time must be finite and positive"))
    isfinite(output_interval) && output_interval > 0 || throw(ArgumentError("output_interval must be finite and positive"))
    T = FloatType
    force = sloshing_tank_force(T; acceleration_path, center = force_center)
    first(force.times) <= 0 && T(simulation_time) <= last(force.times) ||
        throw(ArgumentError("acceleration history must cover the complete requested simulation interval"))
    case = sloshing_tank_geometry(; dx, FloatType)
    probes = measurements ? MeasurementConfig(
        pressure_probes = [MeasurementProbe("left_wall_pressure", (-0.4461, 0.093))],
        water_column_probes = [WaterColumnProbe(name, (x, 0.0); radius = 2dx)
            for (name, x) in (("left_level", -0.44), ("center_level", 0.0), ("right_level", 0.44))],
    ) : nothing
    mkpath(save_location)
    meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = "SloshingTank", SaveLocation = save_location,
        SimulationTime = simulation_time, OutputTimes = output_interval,
        VisualizeInParaview = visualize, OpenLogFile = open_log_file,
        GPUUseGraph = use_graph, GPUDoublePosition = double_position,
        GPUCellSubdivision = 1, ExportGridCells = true)
    particles = AllocateDataStructures(case.geometry, meta)
    logger = SimulationLogger(save_location)
    try
        return RunSimulation(SimGeometry = case.geometry, SimMetaData = meta, SimParticles = particles,
            SimConstants = case.constants, SimKernel = case.kernel, SimLogger = logger,
            SimTimeStepping = time_stepping, SimViscosity = ArtificialViscosity(),
            SimDensityDiffusion = ComplexDensityDiffusion(), SimExternalForces = force,
            SimMeasurements = probes)
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end

# Including this file starts the simulation.
run_sloshing_tank(;
    save_location = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "output", "sloshing_tank"),
    dx = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.002,
    simulation_time = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 8.35)
