# DualSPHysics 5.4 examples/main/12_FloatingWaves (CaseFloatingWavesVal2), drawn
# and run with SPHExampleGPU: a 2D floating box in regular waves made by a piston
# (video: https://youtu.be/VnBVXMVNV7s). The piston follows second-order wave
# maker theory for 0.1 m high, 1.2 s waves in 0.4 m of water, and the box's
# surge, heave and roll can be compared with the experiment of Ren et al. (2015)
# in example/data/floating_waves. Reference x/z coordinates map to simulation
# x/y. Including or running this file starts the simulation. Requires an NVIDIA GPU.
#
#     julia --project=gpu_version -t 2,0 gpu_version/example/FloatingWavesPiston2d.jl
using SPHExampleGPU

let
    Dimensions = 2
    FloatType  = Float32

    # Flume in metres: a piston, a flat bed and a straight beach rising 1.8 m over
    # 7 m from x = 5 m. Walls end at 0.75 m, where the reference crops its domain.
    dx          = 0.005
    water_depth = 0.4
    piston_x    = 0.205  # face towards the water
    beach_toe   = (5.0, 0.0)
    beach_slope = 1.8 / 7.0
    tank_top    = 0.75
    box_corner, box_width, box_height = (1.85, 0.3), 0.3, 0.2
    box_relative_density = 0.5
    wave_height, wave_period = 0.1, 1.2

    beach_end   = (beach_toe[1] + tank_top / beach_slope, tank_top)
    shoreline   = beach_toe[1] + water_depth / beach_slope

    # The tank wall grows outwards from the water, with four particle layers to
    # fill the kernel support. The piston is as thick and slides one spacing above
    # the bed; nothing closes the dry tank behind it.
    tank   = polyline([(0.0, 0.0), beach_toe, beach_end]; thickness = 3dx, side = :right)
    piston = rectangle((piston_x - 3dx, dx), 3dx, tank_top - dx)
    box    = rectangle(box_corner, box_width, box_height)
    water  = polygon([(piston_x, 0.0), beach_toe, (shoreline, water_depth), (piston_x, water_depth)])

    # The tank is sampled along its own outline, so the beach has no staircase.
    # The box claims its lattice points before the water, which surrounds it and,
    # like GenCase's fillbox, stops one spacing short of the piston, bed and surface.
    regions = [
        ParticleRegion("Tank",   tank,   Fixed; sampling = :conforming),
        ParticleRegion("Piston", piston, Moving),
        ParticleRegion("Box",    box,    Floating),
        ParticleRegion("Water",  water,  Fluid),
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

    # The body mass is RelativeWeight m₀ N, half that of the water it replaces.
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

    SimulationGeometry = [Tank, Piston, Box, Water]

    # Second-order piston wave maker theory (Madsen 1971): the first harmonic's
    # stroke S₀ = H / m₁ comes from the linear transfer function, and a second
    # harmonic suppresses the spurious free waves. The wave number solves the
    # linear dispersion relation ω² = g k tanh(k d).
    g, d, H = 9.81, water_depth, wave_height
    ω = 2π / wave_period
    k = ω^2 / g
    for _ in 1:50
        k -= (g * k * tanh(k * d) - ω^2) / (g * tanh(k * d) + g * k * d * sech(k * d)^2)
    end
    m₁ = 2sinh(k * d)^2 / (sinh(k * d) * cosh(k * d) + k * d)
    a₁ = H / m₁ / 2
    a₂ = H^2 / (32d) * (3cosh(k * d) / sinh(k * d)^3 - 2 / m₁)

    # x = a₁ sin ωt + a₂ sin 2ωt, starting at full speed as in the reference (no
    # ramp). It is sampled with its exact velocity, so the cubic Hermite
    # interpolation follows it closely.
    simulation_time = 13.5
    times = range(0.0, simulation_time + wave_period; step = wave_period / 200)
    PistonMotion = PrescribedMotion(Piston.GroupMarker;
        translation = SampledTrajectory(times, [(a₁ * sin(ω * t) + a₂ * sin(2ω * t), 0.0) for t in times];
                                        velocities = [(a₁ * ω * cos(ω * t) + 2a₂ * ω * cos(2ω * t), 0.0) for t in times]),
        center      = (0.0, 0.0),
    )

    # Water height above the bed between the piston and the box; subtract the
    # initial height for the wave elevation.
    WaveGauges = MeasurementConfig(
        water_column_probes = [WaterColumnProbe("WG1", (1.0, 0.0); radius = dx)],
    )

    SimMetaData = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        NoMDBC,
        StoreLog,
    }(
        SimulationName      = "FloatingWavesPiston2D",
        SaveLocation        = joinpath(@__DIR__, "..", "output", "floating_waves_piston_2d"),
        SimulationTime      = simulation_time,
        OutputTimes         = 0.05,
        VisualizeInParaview = true,
        ExportSingleVTKHDF  = true,
        ExportGridCells     = true,
        OpenLogFile         = true,
        # Float32 positions cannot resolve the small steps of slow particles six metres from the origin.
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
    SavePolygonVTKHDF(joinpath(SimMetaData.SaveLocation, "FloatingWavesPiston2D_Geometry.vtkhdf"), (; tank, piston, box, water))

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
