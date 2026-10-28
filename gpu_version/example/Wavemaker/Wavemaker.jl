# DualSPHysics main/06_Wavemaker: sinusoidal piston and a sloping beach.
# From the repository root:
#   julia --project=gpu_version -t 2,0 gpu_version/example/Wavemaker/Wavemaker.jl [output_dir] [dx] [duration]
# From gpu_version/:
#   julia --project=. -t 2,0 example/Wavemaker/Wavemaker.jl
using SPHExampleGPU
using StaticArrays: SVector

include(joinpath(@__DIR__, "Geometry.jl"))
using .WavemakerGeometry

"""
    wavemaker_motion()

Prescribe the reference piston schedule for moving group 2: five seconds at
0.5 Hz, then alternating ten-second segments at 0.75 Hz and 0.5 Hz, stopping
at 100 seconds. Amplitude is 0.25 m (half the peak-to-peak stroke). The initial phase is
3π/2, so the piston starts at rest and moves into the tank.
"""
function wavemaker_motion()
    amplitude = 0.25
    first_frequency = 0.5
    second_frequency = 0.75

    segments = [SinusoidalSegment(5.0;
        amplitude = (amplitude, 0.0, 0.0),
        frequency = (first_frequency, 0.0, 0.0),
        phase = (4.71238898, 0.0, 0.0),
    )]
    for segment in 1:10
        frequency = isodd(segment) ? second_frequency : first_frequency
        push!(segments, SinusoidalSegment(10.0;
            amplitude = (amplitude, 0.0, 0.0),
            frequency = (frequency, 0.0, 0.0),
        ))
    end
    translation = TrajectoryWindow(MotionSequence(segments); end_time = 100.0)
    return PrescribedMotion(2; translation, center = (0.0, 0.0, 0.0))
end

"""
    run_wavemaker(; dx = 0.02, simulation_time = 10.0, output_interval = 0.1, ...)

Generate the wave tank and run its CUDA simulation. Geometry, particle CSVs,
an animated polygon preview and solver output are saved below `save_location`.
Use a new output directory for each run. Returns a `SimulationResult`.
"""
function run_wavemaker(; FloatType = Float32, dx = 0.02, simulation_time = 10.0,
        output_interval = 0.1,
        cache_fields = false, lanes = 1, boundary_forces = false,
        stationary = false,
        density_diffusion = FourtakasDensityDiffusion(),
        time_stepping = nothing, periodic_cell_coordinates = true, fast_math = true,
        visualize = true, open_log_file = true, use_graph = false,
        double_position = true, measurements = false,
        save_location = joinpath(@__DIR__, "..", "..", "output", "wavemaker"))
    isfinite(simulation_time) && 0 < simulation_time <= 100 ||
        throw(ArgumentError("simulation_time must be finite and in (0, 100]"))
    isfinite(output_interval) && output_interval > 0 ||
        throw(ArgumentError("output_interval must be finite and positive"))
    dimensions = 3
    T = FloatType

    # Geometry is kept in a separate, editable file.
    case = wavemaker_geometry(; dx)
    piston_motion = stationary ? PrescribedMotion(2; center = (0.0, 0.0, 0.0)) : wavemaker_motion()

    # Physical constants follow CaseWavemaker_Def.xml. Automatic sound speed
    # uses the sampled fluid depth, as in GenCase, rather than the polygon height.
    sampled_depth = case.water_level - minimum(last, case.positions.water)
    constants = SimulationConstants{T}(dimensions;
        dx = dx,
        ρ₀ = 1000.0,
        g = 9.81,
        c₀ = 20sqrt(9.81 * sampled_depth),
        CFL = 0.2,
        α = 0.02,
        δᵩ = 0.1,
    )

    fixed_boundary = SPHGeometry{dimensions,T}(;
        Particles = particle_struct_array(case.positions.fixed, constants.ρ₀),
        GroupMarker = 1,
        Type = Fixed,
    )
    piston = SPHGeometry{dimensions,T}(;
        Particles = particle_struct_array(case.positions.piston, constants.ρ₀),
        GroupMarker = 2,
        Type = Moving,
    )
    water = SPHGeometry{dimensions,T}(case.positions.water;
        Density = hydrostatic_density(case.positions.water, constants; water_level = case.water_level),
        GroupMarker = 3,
        Type = Fluid,
    )
    geometry = [fixed_boundary, piston, water]

    # The distributed reference uses DBC, without ghost-node correction.
    metadata = SimulationMetaData{
        dimensions,
        T,
        NoShifting,
        NoKernelOutput,
        NoMDBC,
        StoreLog,
    }(
        SimulationName = "Wavemaker",
        SaveLocation = save_location,
        SimulationTime = simulation_time,
        OutputTimes = output_interval,
        VisualizeInParaview = visualize,
        ExportSingleVTKHDF = true,
        ExportGridCells = false,
        OpenLogFile = open_log_file,
        Periodicity = case.periodicity,
        GPUUseGraph = use_graph,
        GPUDoublePosition = double_position,
        GPUPeriodicCellCoordinates = periodic_cell_coordinates,
        GPUCellSubdivision = 1,
        GPUCacheInteractionFields = cache_fields,
        GPULanesPerParticle = lanes,
        # The tank and piston are prescribed; their interaction forces are not
        # needed for rigid-body dynamics. Fluid forces are always evaluated.
        GPUBoundaryForces = boundary_forces,
        GPUClipTimeStepToOutput = false,
    )

    h = 1.5sqrt(3) * dx
    kernel = SPHKernelInstance{dimensions,T}(WendlandC2();
        h = T(h), αD = T(21 / (16pi * h^3)), η² = T((0.1h)^2))
    if time_stepping === nothing
        border = 0.05h
        xmin, xmax = extrema(first,case.positions.fixed)
        zmin, zmax = extrema(last,case.positions.fixed)
        time_stepping = DualSPHysicsSymplectic(;
            smoothing_length = h, sound_speed = 20sqrt(9.81 * sampled_depth),
            domain_min = (xmin - border, 0.0, zmin - border),
            domain_max = (xmax + border, 2.0, zmax + border + 0.5 * (zmax - zmin + 2border)),
            density_min = 700.0, density_max = 1300.0, fast_math)
    end
    viscosity = ArtificialViscosity()
    probes = measurements ? MeasurementConfig(
        water_column_probes = [WaterColumnProbe("WG0", (1.0, 1.0, 0.0); radius = 2dx)],
    ) : nothing

    mkpath(save_location)
    save_wavemaker_geometry(save_location, case, geometry)
    preview_times = collect(0.0:output_interval:simulation_time)
    last(preview_times) < simulation_time && push!(preview_times, Float64(simulation_time))
    SavePolygonMotionSequence(joinpath(save_location, "Wavemaker_Motion.vtkhdf"), case.surfaces;
        motions = (; piston = (point, time) -> motion_position(piston_motion, point, time)),
        times = preview_times,
    )

    particles = AllocateDataStructures(geometry, metadata)
    logger = SimulationLogger(save_location)
    try
        return RunSimulation(
            SimGeometry = geometry,
            SimMetaData = metadata,
            SimConstants = constants,
            SimLogger = logger,
            SimParticles = particles,
            SimKernel = kernel,
            SimViscosity = viscosity,
            SimDensityDiffusion = density_diffusion,
            SimTimeStepping = time_stepping,
            SimMotion = piston_motion,
            SimMeasurements = probes,
        )
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_wavemaker(;
        save_location = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "output", "wavemaker"),
        dx = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.02,
        simulation_time = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 10.0,
    )
end
