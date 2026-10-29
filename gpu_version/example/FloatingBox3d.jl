# DualSPHysics examples/main/11_Floating (CaseFloating), drawn and run with
# SPHExampleGPU: a 3D floating box in a wave tank, periodic across the tank
# width, driven by a piston in sinusoidal motion, with δ-SPH density diffusion
# (video: https://youtu.be/hdgJJJ_5Mxo). Including or running this file starts
# the simulation. Requires an NVIDIA GPU.
#
#     julia --project=gpu_version -t 2,0 gpu_version/example/FloatingBox3d.jl
using SPHExampleGPU

let
    Dimensions = 3
    FloatType  = Float32

    # Tank in metres: x along the tank, y across it (periodic), z up.
    dx = 0.1
    tank_length, tank_width, tank_height = 16.0, 6.0, 6.0
    water_height = 4.0
    piston_x, piston_thickness, piston_height = 0.7, 0.3, 5.0
    box_corner, box_size, box_mass = (10.0, 2.0, 3.0), 2.0, 10_000.0

    # Particle rows lie at y = 0, dx, …, tank_width - dx. The periodic faces sit
    # half a spacing outside them, so no row is duplicated across the seam.
    y_min, y_max = -dx / 2, tank_width - dx / 2
    plan(x, extent) = rectangle((x, y_min), extent, tank_width)

    # The bed and end wall grow outwards from the water, with four particle
    # layers to fill the kernel support (2h ≈ 3.5dx). The piston slides one
    # spacing above the bed; nothing closes the tank behind it.
    bed    = prism(plan(0.0, tank_length + 3dx), -3dx, 0.0)
    wall   = prism(plan(tank_length, 3dx), -3dx, tank_height)
    piston = prism(plan(piston_x, piston_thickness), dx, piston_height)
    box    = prism(rectangle(box_corner[1:2], box_size, box_size), box_corner[3], box_corner[3] + box_size)
    water  = prism(plan(piston_x + piston_thickness, tank_length - piston_x - piston_thickness), 0.0, water_height)

    # Earlier regions claim their lattice points first, so the water surrounds the
    # half-submerged box. The water fills the open interior, stopping one spacing
    # short of the piston, bed, end wall and its 4 m top, like GenCase's fillbox.
    regions = [
        ParticleRegion("Tank",   (bed, wall), Fixed),
        ParticleRegion("Piston", piston,      Moving),
        ParticleRegion("Box",    box,         Floating),
        ParticleRegion("Water",  water,       Fluid),
    ]
    TankParticles, PistonParticles, BoxParticles, WaterParticles = sample_particles(regions, dx)

    # As GenCase does, the sound speed uses the height of the sampled water column.
    water_bottom, water_level = extrema(last, WaterParticles.positions)
    SimConstants = SimulationConstants{FloatType}(Dimensions;
        dx = dx,
        c₀ = 20 * sqrt(9.81 * (water_level - water_bottom)),
    )
    density(particles) = hydrostatic_density(particles.positions, SimConstants; water_level)

    Tank = SPHGeometry{Dimensions, FloatType}(TankParticles.positions;
        Density     = density(TankParticles),
        GroupMarker = 1,
        Type        = Fixed,
    )

    Piston = SPHGeometry{Dimensions, FloatType}(PistonParticles.positions;
        Density     = density(PistonParticles),
        GroupMarker = 2,
        Type        = Moving,
    )

    # The body mass is RelativeWeight m₀ N, so this gives the reference's 10 t box.
    Box = SPHGeometry{Dimensions, FloatType}(BoxParticles.positions;
        Density     = density(BoxParticles),
        GroupMarker = 3,
        Type        = Floating,
        Floating    = FloatingDetails{FloatType}(RelativeWeight = box_mass / (SimConstants.m₀ * length(BoxParticles.positions))),
    )

    Water = SPHGeometry{Dimensions, FloatType}(WaterParticles.positions;
        Density     = density(WaterParticles),
        GroupMarker = 4,
        Type        = Fluid,
    )

    SimulationGeometry = [Tank, Piston, Box, Water]

    # x = 0.5 sin(2π 0.3 t) for 50 s: the piston starts at full speed into the tank.
    PistonMotion = PrescribedMotion(Piston.GroupMarker;
        translation = MotionSequence([SinusoidalSegment(50.0; amplitude = (0.5, 0.0, 0.0), frequency = (0.3, 0.0, 0.0))]),
        center      = (0.0, 0.0, 0.0),
    )

    # Water heights above the bed between the piston and the box and behind the box.
    WaveGauges = MeasurementConfig(
        water_column_probes = [WaterColumnProbe("WG$i", (x, 1.0, 0.0); radius = dx) for (i, x) in enumerate((6.0, 14.0))],
    )

    SimMetaData = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        NoMDBC,
        StoreLog,
    }(
        SimulationName      = "FloatingBox3D",
        SaveLocation        = joinpath(@__DIR__, "..", "output", "floating_box_3d"),
        SimulationTime      = 10,
        OutputTimes         = 0.1,
        VisualizeInParaview = true,
        ExportSingleVTKHDF  = true,
        ExportGridCells     = false,
        OpenLogFile         = true,
        Periodicity         = PeriodicBoundary{Dimensions}(2; min = y_min, max = y_max),
        # Float32 positions cannot resolve the small steps of slow particles sixteen metres from the origin.
        GPUDoublePosition   = true,
    )

    # h = √3 dx, the reference's coefh = 1 in 3D.
    SimKernel           = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); h = FloatType(sqrt(3) * dx))
    SimViscosity        = ArtificialViscosity()
    SimDensityDiffusion = LinearDensityDiffusion()

    mkpath(SimMetaData.SaveLocation)

    SimLogger    = SimulationLogger(SimMetaData.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaData)

    CleanUpSimulationFolder(SimMetaData.SaveLocation)
    SavePolygonVTKHDF(joinpath(SimMetaData.SaveLocation, "FloatingBox3D_Geometry.vtkhdf"), (; bed, wall, piston, box, water))

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
