# DualSPHysics 5.4 examples/main/07_WavemakerFile, drawn and run with SPHExampleGPU.
# A piston follows the measured CIEMito paddle signal in a 2D flume; the waves
# shoal on a gentle beach and run up a vertical end wall. Reference x/z
# coordinates map to simulation x/y. Including or running this file starts the
# simulation. Requires an NVIDIA GPU.
#
#     julia --project=gpu_version -t 2,0 gpu_version/example/WavemakerFile2d.jl
using SPHExampleGPU

let
    Dimensions = 2
    FloatType  = Float32

    # Flume in metres: a piston, a flat bed, a 2 cm step at the beach toe and a
    # straight beach rising to the vertical end wall.
    dx          = 0.005
    water_depth = 0.31
    piston_x, piston_width, piston_height = 1.0, 0.03, 0.55
    bed_start   = 0.8
    beach_toe   = (4.0, 0.02)
    beach_crest = (6.045, 0.235)
    wall_top    = 0.5

    SimConstants = SimulationConstants{FloatType}(Dimensions;
        dx = dx,
        c₀ = 20 * sqrt(9.81 * water_depth),
    )

    # Draw the wetted surfaces. The tank wall grows outwards from the water, with
    # four particle layers to fill the kernel support (2h = 4dx) of the fluid.
    bed    = [(bed_start, 0.0), (beach_toe[1], 0.0), beach_toe, beach_crest, (beach_crest[1], wall_top)]
    tank   = polyline(bed; thickness = 3dx, side = :right)
    piston = rectangle((piston_x, dx), piston_width, piston_height - dx)  # slides one spacing above the bed
    water  = polygon([(piston_x + piston_width, 0.0), bed[2:4]...,
                      (beach_crest[1], water_depth), (piston_x + piston_width, water_depth)])

    # The tank is sampled along its own outline, so the beach has no staircase.
    regions = [
        ParticleRegion("Tank",   tank,   Fixed; sampling = :conforming),
        ParticleRegion("Piston", piston, Moving),
        ParticleRegion("Water",  water,  Fluid),
    ]
    TankParticles, PistonParticles, WaterParticles = sample_particles(regions, dx)

    FixedBoundary = SPHGeometry{Dimensions, FloatType}(TankParticles.positions;
        Density     = SimConstants.ρ₀,
        GroupMarker = 1,
        Type        = Fixed,
    )

    Piston = SPHGeometry{Dimensions, FloatType}(PistonParticles.positions;
        Density     = SimConstants.ρ₀,
        GroupMarker = 2,
        Type        = Moving,
    )

    Water = SPHGeometry{Dimensions, FloatType}(WaterParticles.positions;
        Density     = hydrostatic_density(WaterParticles.positions, SimConstants),
        GroupMarker = 3,
        Type        = Fluid,
    )

    SimulationGeometry = [FixedBoundary, Piston, Water]

    # The measured piston signal: time and x displacement, interpolated linearly.
    data_dir = joinpath(@__DIR__, "data", "wavemaker_file")
    signal = [parse.(Float64, split(line)) for line in eachline(joinpath(data_dir, "CaseWavemaker2D_Piston_Movement.dat"))]
    PistonMotion = PrescribedMotion(Piston.GroupMarker;
        translation = SampledTrajectory(first.(signal), [(row[2], 0.0) for row in signal]),
        center      = (0.0, 0.0),
    )

    # Water heights above y = 0 at the flume's wave gauges; subtract the initial
    # height to compare with the elevations in EXP_CaseWavemaker2D_CIEMito.txt.
    WaveGauges = MeasurementConfig(
        water_column_probes = [WaterColumnProbe("WG$i", (x, 0.0); radius = dx) for (i, x) in enumerate((1.38, 2.83, 4.12))],
    )

    SimMetaData = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        NoMDBC,
        StoreLog,
    }(
        SimulationName      = "WavemakerFile2D",
        SaveLocation        = joinpath(@__DIR__, "..", "output", "wavemaker_file_2d"),
        SimulationTime      = 10,
        OutputTimes         = 0.05,
        VisualizeInParaview = true,
        ExportSingleVTKHDF  = true,
        ExportGridCells     = true,
        OpenLogFile         = true,
        # Float32 positions cannot resolve the small steps of slow particles six metres from the origin.
        GPUDoublePosition   = true,
    )

    SimKernel           = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); dx = SimConstants.dx)
    SimViscosity        = ArtificialViscosity()
    SimDensityDiffusion = LinearDensityDiffusion()

    mkpath(SimMetaData.SaveLocation)

    SimLogger    = SimulationLogger(SimMetaData.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaData)

    CleanUpSimulationFolder(SimMetaData.SaveLocation)
    SavePolygonVTKHDF(joinpath(SimMetaData.SaveLocation, "WavemakerFile2D_Geometry.vtkhdf"), (; tank, piston, water))

    RunSimulation(
        SimGeometry         = SimulationGeometry,
        SimMetaData         = SimMetaData,
        SimConstants        = SimConstants,
        SimLogger           = SimLogger,
        SimParticles        = SimParticles,
        SimKernel           = SimKernel,
        SimViscosity        = SimViscosity,
        SimDensityDiffusion = SimDensityDiffusion,
        SimTimeStepping     = SymplecticTimeStepping(),
        SimMotion           = PistonMotion,
        SimMeasurements     = WaveGauges,
    )
end
