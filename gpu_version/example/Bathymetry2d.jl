# DualSPHysics 5.4 examples/main/18_Bathymetry (CaseBathymetry2D), drawn and run
# with SPHExampleGPU: regular waves from a piston travel over a bathymetry made
# from surveyed XYZ points, then shoal and break on its beach. The bed is built
# here from the bundled points as the reference builds it: the points are moved,
# scaled and selected, sampled on a 2 m grid and extended, and the 2D case is a
# section through that grid. Reference x/z coordinates map to simulation x/y.
# Including or running this file starts the simulation. Requires an NVIDIA GPU.
#
#     julia --project=gpu_version -t 2,0 gpu_version/example/Bathymetry2d.jl
using SPHExampleGPU
using Meshes: Point, PointSet, DelaunayTesselation, tesselate, elements, topology, indices
using Random: Xoshiro

let
    Dimensions = 2
    FloatType  = Float32

    # Survey points (x, y, z) in metres, moved to the reference's frame with
    # heights scaled by 1.8, inside the area the reference selects.
    survey = map(Iterators.filter(!isempty ∘ strip, eachline(joinpath(@__DIR__, "data", "bathymetry", "Bathymetry.xyz")))) do line
        x, y, z = parse.(Float64, split(line))
        (x - 439998.30, y - 1148505.00, 1.8z)
    end
    filter!(((x, y, _),) -> 170 <= x <= 280 && 20 <= y <= 90, survey)

    # Heights between the scattered points are interpolated linearly on their
    # Delaunay triangulation. A fixed seed keeps the triangulation reproducible
    # where points are cocircular.
    triangulation = tesselate(PointSet([Point(x, y) for (x, y, _) in survey]), DelaunayTesselation(Xoshiro(0)))
    triangles = [indices(c) for c in elements(topology(triangulation))]
    function survey_height(x, y)
        for (i₁, i₂, i₃) in triangles
            (x₁, y₁, z₁), (x₂, y₂, z₂), (x₃, y₃, z₃) = survey[i₁], survey[i₂], survey[i₃]
            area = (y₂ - y₃) * (x₁ - x₃) + (x₃ - x₂) * (y₁ - y₃)
            w₁ = ((y₂ - y₃) * (x - x₃) + (x₃ - x₂) * (y - y₃)) / area
            w₂ = ((y₃ - y₁) * (x - x₃) + (x₁ - x₃) * (y - y₃)) / area
            w₃ = 1 - w₁ - w₂
            min(w₁, w₂, w₃) >= -1e-12 && return w₁ * z₁ + w₂ * z₂ + w₃ * z₃
        end
        throw(ArgumentError("($x, $y) lies outside the survey"))
    end

    # As in the reference, the survey is sampled on a 2 m grid over
    # 200 ≤ x ≤ 260 m, which is extended upstream by flat nodes at 190 and 182 m
    # and up to a beach crest at 280 m, then moved by (-185, -30, -21) m. The 2D
    # case is the grid row at y = 32 m after the move, along which the bed is
    # straight between the nodes.
    section_y = 32.0 + 30.0
    nodes = [(182.0, 15.0); (190.0, 15.0); [(x, survey_height(x, section_y)) for x in 200.0:2.0:260.0]; (280.0, 28.0)]
    bed   = [(x - 185, z - 21) for (x, z) in nodes]
    function bed_height(x)
        i = clamp(findlast(node -> first(node) <= x, bed), 1, length(bed) - 1)
        (x₁, z₁), (x₂, z₂) = bed[i], bed[i + 1]
        return z₁ + (x - x₁) * (z₂ - z₁) / (x₂ - x₁)
    end

    # Flume in metres. The still water level is at y = 0 and the bed lies 6 m
    # below it at the piston, whose wet particle column stands at x = 0.8 m.
    dx                = 0.2
    still_water_level = 0.0
    piston_face       = 0.8
    piston_top        = 6.0
    shore_node        = findfirst(node -> last(node) >= still_water_level, bed)
    (x_wet, z_wet), (x_dry, z_dry) = bed[shore_node - 1], bed[shore_node]
    shoreline         = x_wet + (still_water_level - z_wet) * (x_dry - x_wet) / (z_dry - z_wet)

    # The bed grows downwards from the water, with four particle layers to fill
    # the kernel support. The piston is as thick and slides one spacing above the
    # bed; nothing closes the dry flume behind it. Like GenCase's fillbox, the
    # water stops one spacing short of the piston, bed and surface.
    tank   = polyline(bed; thickness = 3dx, side = :right)
    piston = rectangle((piston_face - 3dx, bed_height(piston_face) + dx), 3dx, piston_top - bed_height(piston_face) - 2dx)
    water  = polygon([(piston_face, bed_height(piston_face));
                      [node for node in bed if piston_face < first(node) < shoreline];
                      (shoreline, still_water_level);
                      (piston_face, still_water_level)])

    # The tank is sampled along its own outline, so the bed has no staircase.
    regions = [
        ParticleRegion("Tank",   tank,   Fixed; sampling = :conforming),
        ParticleRegion("Piston", piston, Moving),
        ParticleRegion("Water",  water,  Fluid),
    ]
    TankParticles, PistonParticles, WaterParticles = sample_particles(regions, dx)

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

    Water = SPHGeometry{Dimensions, FloatType}(WaterParticles.positions;
        Density     = density(WaterParticles),
        GroupMarker = 3,
        Type        = Fluid,
    )

    SimulationGeometry = [Tank, Piston, Water]

    # Second-order piston wave maker theory (Madsen 1971) for the reference's
    # 1.5 m high, 4 s waves in its nominal depth of 5.5 m. The wave number solves
    # the linear dispersion relation ω² = g k tanh(k d). The first harmonic's
    # stroke H / m₁ is amplified by the reference's gain of 1.1, and a second
    # harmonic suppresses the spurious free waves.
    g, d, H, T = 9.81, 5.5, 1.5, 4.0
    gain = 1.1
    ω = 2π / T
    k = ω^2 / g
    for _ in 1:50
        k -= (g * k * tanh(k * d) - ω^2) / (g * tanh(k * d) + g * k * d * sech(k * d)^2)
    end
    m₁ = 2sinh(k * d)^2 / (sinh(k * d) * cosh(k * d) + k * d)
    a₁ = gain * H / m₁ / 2
    a₂ = H^2 / (32d) * (3cosh(k * d) / sinh(k * d)^3 - 2 / m₁)

    # Over the first period, both harmonics are scaled by the ramp
    # r(s) = 6s² - 8s³ + 3s⁴ with s = t / T, which rises from 0 to 1 with zero
    # slope at both ends; with it, the motion matches the one DualSPHysics writes
    # for this case. It is sampled with its exact velocity, so the cubic Hermite
    # interpolation follows it closely.
    ramp(t)  = (s = min(t / T, 1.0); s^2 * (6 - 8s + 3s^2))
    ramp′(t) = (s = min(t / T, 1.0); 12s * (1 - s)^2 / T)
    stroke(t)  = a₁ * sin(ω * t) + a₂ * sin(2ω * t)
    stroke′(t) = a₁ * ω * cos(ω * t) + 2a₂ * ω * cos(2ω * t)
    simulation_time = 30.0
    times = range(0.0, simulation_time + T; step = T / 200)
    PistonMotion = PrescribedMotion(Piston.GroupMarker;
        translation = SampledTrajectory(times, [(ramp(t) * stroke(t), 0.0) for t in times];
                                        velocities = [(ramp′(t) * stroke(t) + ramp(t) * stroke′(t), 0.0) for t in times]),
        center      = (0.0, 0.0),
    )

    # Water depths along the flume, from x = 6 m, where DualSPHysics writes the
    # theoretical elevation, to the shallows in front of the beach. Each gauge
    # stands on the bed; subtract its initial depth for the wave elevation.
    WaveGauges = MeasurementConfig(
        water_column_probes = [WaterColumnProbe("WG" * lpad(round(Int, x), 2, '0'), (x, bed_height(x)); radius = dx)
                               for x in [6.0; 10.0:5.0:75.0]],
    )

    SimMetaData = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        NoMDBC,
        StoreLog,
    }(
        SimulationName      = "Bathymetry2D",
        SaveLocation        = joinpath(@__DIR__, "..", "output", "bathymetry_2d"),
        SimulationTime      = simulation_time,
        OutputTimes         = 0.1,
        VisualizeInParaview = true,
        ExportSingleVTKHDF  = true,
        ExportGridCells     = true,
        OpenLogFile         = true,
        # Float32 positions cannot resolve the small steps of slow particles tens of metres from the origin.
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
    SavePolygonVTKHDF(joinpath(SimMetaData.SaveLocation, "Bathymetry2D_Geometry.vtkhdf"), (; tank, piston, water))

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
