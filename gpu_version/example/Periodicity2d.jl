# DualSPHysics 5.4 examples/main/02_Periodicity, with x/z mapped to x/y.
# Run from the repository root:
# julia --project=gpu_version gpu_version/example/Periodicity2d.jl [output_dir] [dx] [duration]
using SPHExampleGPU
using StaticArrays: SVector

"""
    periodicity_2d_geometry(; dx = 0.002, FloatType = Float32)

Generate the floor, step, baffle and two water columns from CasePeriodicity_Def.xml.
The reference's XPeriodicIncZ = 0.3 raises particles crossing the right seam.
The periodic limits include half a spacing outside the boundary-particle extrema,
as in DualSPHysics. Coordinates and time are in SI units.
"""
function periodicity_2d_geometry(; dx = 0.002, FloatType = Float32)
    isfinite(dx) && 0 < dx <= 0.01 || throw(ArgumentError("dx must be finite and in (0, 0.01] to resolve the walls"))
    floor = polygon([(-0.01, 0.3), (0.0, 0.3), (0.0, 0.0), (0.45, 0.0),
        (0.45, 0.02), (0.48, 0.02), (0.48, 0.0), (0.8, 0.0),
        (0.8, -0.01), (-0.01, -0.01)])
    baffle = polygon([(0.25, 0.01), (0.30, 0.01), (0.30, 0.04),
        (0.26, 0.08), (0.26, 0.34), (0.25, 0.34)])
    # GenCase fillbox excludes its outer faces; walls own shared lattice nodes.
    regions = [ParticleRegion("Bound", (floor, baffle), Fixed),
        ParticleRegion("Fluid", (rectangle((0, 0), 0.25, 0.25),
            rectangle((0.2, 0), 0.6, 0.03)), Fluid)]
    sampled = sample_particles(regions, dx)
    fluid = sampled[2].positions
    # DualSPHysics computes sound speed from the deepest local water column.
    T = FloatType
    depth = maximum(last, filter(x -> x[1] < 0.25, fluid)) - minimum(last, filter(x -> x[1] < 0.25, fluid))
    constants = SimulationConstants{T}(2; dx = T(dx), α = T(0.2), δᵩ = T(0.1),
        g = T(9.81), c₀ = T(10sqrt(9.81depth)), CFL = T(0.2))
    left_level = maximum(last, filter(x -> x[1] < 0.25, fluid))
    right_level = maximum(last, filter(x -> x[1] > 0.30, fluid))
    densities = [hydrostatic_density([x], constants; water_level = x[1] < 0.25 ? left_level : right_level)[1] for x in fluid]
    geometry = [
        SPHGeometry{2, T}(sampled[1].positions; Density = constants.ρ₀, GroupMarker = 1, Type = Fixed),
        SPHGeometry{2, T}(fluid; Density = densities, GroupMarker = 2, Type = Fluid),
    ]
    periodicity = PeriodicBoundary{2}(1; min = -0.01 - dx / 2, max = 0.8 + dx / 2, shift = (0, 0.3))
    return (; geometry, constants, periodicity, sampled)
end

"""
    run_periodicity_2d(; dx = 0.002, simulation_time = 6, output_interval = 0.06, ...)

Run the shifted-periodic dam-break case using symplectic stepping, Wendland C2,
artificial viscosity and the solver's complex hydrostatic density diffusion. Returns a
`SimulationResult`. Generated particles are kept in memory; no input files or
DualSPHysics installation are required.
"""
function run_periodicity_2d(; FloatType = Float32, dx = 0.002, simulation_time = 6.0,
                            output_interval = 0.06, visualize = true, open_log_file = true,
                            use_graph = true, fixed_grid = true, double_position = false,
                            save_location = joinpath(@__DIR__, "..", "output", "periodicity_2d"))
    isfinite(simulation_time) && simulation_time > 0 || throw(ArgumentError("simulation_time must be finite and positive"))
    isfinite(output_interval) && output_interval > 0 || throw(ArgumentError("output_interval must be finite and positive"))
    case = periodicity_2d_geometry(; dx, FloatType)
    T = FloatType
    mkpath(save_location)
    meta = SimulationMetaData{2, T, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(
        SimulationName = "Periodicity2D", SaveLocation = save_location,
        SimulationTime = simulation_time, OutputTimes = output_interval,
        VisualizeInParaview = visualize, OpenLogFile = open_log_file,
        Periodicity = case.periodicity, GPUFixedGrid = fixed_grid,
        GPUGridExpansion = fixed_grid ? ((false, false), (true, true)) : ((false, false), (false, false)),
        GPUUseGraph = use_graph, GPUDoublePosition = double_position, GPUCellSubdivision = 1, ExportGridCells = true)
    particles = AllocateDataStructures(case.geometry, meta)
    logger = SimulationLogger(save_location)
    try
        return RunSimulation(SimGeometry = case.geometry, SimMetaData = meta, SimParticles = particles,
            SimConstants = case.constants, SimKernel = SPHKernelInstance{2, T}(WendlandC2(); h = T(0.866025sqrt(2) * dx)),
            SimLogger = logger, SimTimeStepping = SymplecticTimeStepping(),
            SimViscosity = ArtificialViscosity(), SimDensityDiffusion = ComplexDensityDiffusion())
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end

    run_periodicity_2d(;
        save_location = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "output", "periodicity_2d"),
        dx = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.002,
        simulation_time = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 6.0)
