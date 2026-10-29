# DualSPHysics 5.4 examples/main/12_FloatingWaves (CaseFloatingWavesVal), drawn
# and run with SPHExampleGPU: a small 2D floating box in nonlinear waves made by
# a flap (video: https://youtu.be/nwbS8QcDnm4). The flap turns about a hinge at
# the bed, following the bundled angle signal, and the waves break on a gentle
# beach. The wave elevations and the box's heave, roll and sway can be compared
# with the experiment of Hadžić et al. (2005) in example/data/floating_waves.
# Reference x/z coordinates map to simulation x/y. Including or running this
# file starts the simulation. Requires an NVIDIA GPU.
#
#     julia --project=gpu_version -t 2,0 gpu_version/example/FloatingWavesFlap2d.jl
using SPHExampleGPU

let
    Dimensions = 2
    FloatType  = Float32

    # Flume in metres: a flap hinged at the origin, a flat bed and a 1:10 beach
    # from x = 4 m. The bed extends 0.5 m behind the flap, where the tank is dry.
    dx          = 0.002
    water_depth = 0.4
    flap_height = 0.6
    bed_start   = -0.5
    beach_toe   = (4.0, 0.0)
    beach_end   = (10.0, 0.6)
    box_corner, box_width, box_height = (2.06, 0.37), 0.1, 0.05
    box_relative_density = 0.68

    shoreline = beach_toe[1] + water_depth * (beach_end[1] - beach_toe[1]) / beach_end[2]

    # The tank wall grows outwards from the water, with four particle layers to
    # fill the kernel support. The flap is as thick, with its wet face on the
    # hinge, and starts one spacing above the bed.
    tank  = polyline([(bed_start, 0.0), beach_toe, beach_end]; thickness = 3dx, side = :right)
    flap  = rectangle((-3dx, dx), 3dx, flap_height - dx)
    box   = rectangle(box_corner, box_width, box_height)
    water = polygon([(0.0, 0.0), beach_toe, (shoreline, water_depth), (0.0, water_depth)])

    # The tank is sampled along its own outline, so the beach has no staircase.
    # The box claims its lattice points before the water, which surrounds it and,
    # like GenCase's fillbox, stops one spacing short of the flap, bed and surface.
    regions = [
        ParticleRegion("Tank",  tank,  Fixed; sampling = :conforming),
        ParticleRegion("Flap",  flap,  Moving),
        ParticleRegion("Box",   box,   Floating),
        ParticleRegion("Water", water, Fluid),
    ]
    TankParticles, FlapParticles, BoxParticles, WaterParticles = sample_particles(regions, dx)

    # As GenCase does, the sound speed uses the height of the sampled water column.
    water_bottom, water_level = extrema(last, WaterParticles.positions)
    SimConstants = SimulationConstants{FloatType}(Dimensions;
        dx = dx,
        c₀ = 15 * sqrt(9.81 * (water_level - water_bottom)),
    )
    density(particles) = hydrostatic_density(particles.positions, SimConstants; water_level)

    Tank = SPHGeometry{Dimensions, FloatType}(TankParticles.positions;
        Density     = density(TankParticles),
        GroupMarker = 1,
        Type        = Fixed,
    )

    Flap = SPHGeometry{Dimensions, FloatType}(FlapParticles.positions;
        Density     = density(FlapParticles),
        GroupMarker = 2,
        Type        = Moving,
    )

    # The body mass is RelativeWeight m₀ N, 0.68 times that of the water it replaces.
    Box = SPHGeometry{Dimensions, FloatType}(BoxParticles.positions;
        Density     = density(BoxParticles),
        GroupMarker = 3,
        Type        = Floating,
        Floating    = FloatingDetails{FloatType}(RelativeWeight = box_relative_density),
    )

    Water = SPHGeometry{Dimensions, FloatType}(WaterParticles.positions;
        Density     = density(WaterParticles),
        GroupMarker = 4,
        Type        = Fluid,
    )

    SimulationGeometry = [Tank, Flap, Box, Water]

    # The measured flap angle in degrees, interpolated linearly relative to its
    # first value. As in the reference, a positive angle turns the flap away
    # from the water, counterclockwise in the x-y plane.
    data_dir   = joinpath(@__DIR__, "data", "floating_waves")
    FlapMotion = PrescribedMotion(Flap.GroupMarker;
        rotation = load_rotation_trajectory(joinpath(data_dir, "CaseFloatingWavesVal_Flap.dat"); angle_units = :degrees),
        center   = (0.0, 0.0),
        axis     = (0.0, 0.0, 1.0),
    )

    # Water heights above the bed at the reference's elevation probes, in front
    # of and behind the box; subtract the initial height for the wave elevation.
    WaveGauges = MeasurementConfig(
        water_column_probes = [WaterColumnProbe("WG$i", (x, 0.0); radius = dx) for (i, x) in enumerate((1.16, 2.66))],
    )

    SimMetaData = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        NoMDBC,
        StoreLog,
    }(
        SimulationName      = "FloatingWavesFlap2D",
        SaveLocation        = joinpath(@__DIR__, "..", "output", "floating_waves_flap_2d"),
        SimulationTime      = 15,
        OutputTimes         = 0.05,
        VisualizeInParaview = true,
        ExportSingleVTKHDF  = true,
        ExportGridCells     = true,
        OpenLogFile         = true,
        # Float32 positions cannot resolve the small steps of slow particles eight metres from the origin.
        GPUDoublePosition   = true,
    )

    # h = 1.2 √2 dx, the reference's coefh = 1.2 in 2D.
    SimKernel           = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); h = FloatType(1.2 * sqrt(2) * dx))
    SimViscosity        = ArtificialViscosity()
    SimDensityDiffusion = LinearDensityDiffusion()

    mkpath(SimMetaData.SaveLocation)

    SimLogger    = SimulationLogger(SimMetaData.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaData)

    CleanUpSimulationFolder(SimMetaData.SaveLocation)
    SavePolygonVTKHDF(joinpath(SimMetaData.SaveLocation, "FloatingWavesFlap2D_Geometry.vtkhdf"), (; tank, flap, box, water))

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
        SimMotion           = FlapMotion,
        SimMeasurements     = WaveGauges,
    )
end
