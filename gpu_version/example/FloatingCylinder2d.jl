# GPU version of the DualSPHysics 2D falling cylinder (examples/main/11_Floating,
# CaseFloatingSphereVal2D). Requires an NVIDIA GPU with CUDA.
#
# A floating cylinder of radius 1 m and relative weight 1.2 starts half
# submerged in a 10 m wide, 14 m deep tank. It is held still for the first
# second (DualSPHysics `FtPause`) while the water settles, then sinks. The
# periodic sides of the DualSPHysics case are fixed walls here.
#
# Particles are generated in this file, then the GPU solver runs.
# Including or running this file starts the simulation:
#     julia --project=gpu_version gpu_version/example/FloatingCylinder2d.jl
#
# Besides the particle output, `<SimulationName>_Floating.csv` in the save
# location holds the centre, velocity, angle and angular velocity of the
# cylinder at every output time. The DualSPHysics validation data (Fekken 2004,
# Moyo and Greenhow 2000) give the sinking distance and velocity against
# t* = (t - 1 s) sqrt(g / R) for t* <= 8.
using SPHExampleGPU
using StaticArrays: SVector
using Meshes
using StructArrays

function run_floating_cylinder_2d()
    Dimensions = 2
    FloatType = Float32
    dx = 0.025

    # coefsound = 30 and hswl = 0.8 of the DualSPHysics case: c₀ = 30 sqrt(g hswl)
    SimConstantsFloating = SimulationConstants{FloatType}(
        dx = dx,
        ρ₀ = 1000,
        g = 9.81,
        c₀ = 30 * sqrt(9.81 * 0.8),
        δᵩ = 0.1,
        CFL = 0.2,
        ν₀ = 1e-6,
    )

    input_dir =
        normpath(joinpath(@__DIR__, "..", "input", "floating_cylinder_2d_generated"))
    csv(name) = joinpath(input_dir, "FloatingCylinder2D_Dp$(dx)_$(name).csv")

    # Sample particles here; the generator script supplies shape definitions.
    polygons = floating_cylinder_2d_shapes(; dx)
    regions = [
        ParticleRegion("Cylinder", polygons.cylinder, Floating; sampling = :conforming),
        ParticleRegion("Bound", polygons.tank, Fixed),
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    sampled = sample_particles(regions, SimConstantsFloating.dx)
    positions(name) = only(r.positions for r in sampled if r.name == name)
    water_level = maximum(last, positions("Fluid"))

    FixedBoundary = SPHGeometry{Dimensions, FloatType}(
        Particles = StructArray((
            Position = positions("Bound"),
            Density = hydrostatic_density(
                positions("Bound"),
                SimConstantsFloating;
                water_level,
            ),
        )),
        # CSVFile     = csv("Bound"),
        GroupMarker = 1,
        Type = Fixed,
    )
    Water = SPHGeometry{Dimensions, FloatType}(
        Particles = StructArray((
            Position = positions("Fluid"),
            Density = hydrostatic_density(
                positions("Fluid"),
                SimConstantsFloating;
                water_level,
            ),
        )),
        # CSVFile     = csv("Fluid"),
        GroupMarker = 2,
        Type = Fluid,
    )
    Cylinder = SPHGeometry{Dimensions, FloatType}(
        Particles = StructArray((
            Position = positions("Cylinder"),
            Density = hydrostatic_density(
                positions("Cylinder"),
                SimConstantsFloating;
                water_level,
            ),
        )),
        # CSVFile     = csv("Cylinder"),
        GroupMarker = 3,
        Type = Floating,
        Floating = FloatingDetails{FloatType}(RelativeWeight = 1.2, PauseTime = 1.0),
    )
    SimulationGeometry = [FixedBoundary, Water, Cylinder]

    SimMetaDataFloating = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        NoMDBC,
        StoreLog,
    }(
        SimulationName = "FloatingCylinder2D",
        SaveLocation = "C:/TestSimulations/FloatingCylinder2D_GPU",
        SimulationTime = 8.5,
        OutputTimes = 0.02,
        VisualizeInParaview = true,
        ExportSingleVTKHDF = true,
        ExportGridCells = true,
        OpenLogFile = true,
        # positions up to 16 m: integrate them in Float64, the rest in Float32
        GPUDoublePosition = true,
        # Cache inverse density per particle (4 extra device bytes with Float32).
        GPUCacheInteractionFields = true,
    )
    mkpath(SimMetaDataFloating.SaveLocation)

    SimLogger = SimulationLogger(SimMetaDataFloating.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaDataFloating)
    CleanUpSimulationFolder(SimMetaDataFloating.SaveLocation)

    # DualSPHysics coefh = 1.2: h = 1.2 sqrt(2) dx in 2D, kernel support 2h.
    SimKernel = SPHKernelInstance{Dimensions, FloatType}(
        WendlandC2();
        h = FloatType(1.2 * sqrt(2) * dx),
    )

    RunSimulation(
        SimGeometry = SimulationGeometry,
        SimMetaData = SimMetaDataFloating,
        SimConstants = SimConstantsFloating,
        SimKernel = SimKernel,
        SimLogger = SimLogger,
        SimParticles = SimParticles,
        SimViscosity = LaminarSPS(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        SimTimeStepping = SingleNeighborTimeStepping(),
    )
end

"""
    floating_cylinder_2d_shapes(; dx = 0.025, width = 10.0, depth = 14.0,
                                tank_height = 16.0, radius = 1.0)

The tank (walls three spacings thick outside the wetted surface), the water
and the cylinder, centred on the free surface in the middle of the tank. The
water reaches half a spacing above `depth`, so its top particle layer lies at
`depth` like the DualSPHysics case.
"""
function floating_cylinder_2d_shapes(;
    dx = 0.025,
    width = 10.0,
    depth = 14.0,
    tank_height = 16.0,
    radius = 1.0,
)
    half = width / 2
    tank = polyline(
        [(-half, tank_height), (-half, 0.0), (half, 0.0), (half, tank_height)];
        thickness = 3dx,
        side = :right,
    )
    water = rectangle((-half, 0.0), width, depth + dx / 2)
    cylinder = circle((0.0, depth), radius; segments = 256)
    return (; tank, water, cylinder)
end

"""
    floating_cylinder_simulation_constants(FloatType = Float64; dx = 0.025)

Constants of the DualSPHysics case: `c₀ = coefsound * sqrt(g * hswl)` with
`coefsound = 30` and `hswl = 0.8`, `ρ₀ = 1000`, `g = 9.81`, `CFL = 0.2`.
"""
floating_cylinder_simulation_constants(FloatType = Float64; dx = 0.025) =
    SimulationConstants{FloatType}(;
        dx,
        ρ₀ = 1000,
        g = 9.81,
        c₀ = 30 * sqrt(9.81 * 0.8),
        δᵩ = 0.1,
        CFL = 0.2,
        ν₀ = 1e-6,
    )

"""
    generate_floating_cylinder_2d(output_dir; dx = 0.025,
        SimConstants = floating_cylinder_simulation_constants(; dx))

Write `FloatingCylinder2D_Geometry.vtkhdf`, `FloatingCylinder2D_Dp<dx>_Particles.vtkhdf`
and the CSV files `FloatingCylinder2D_Dp<dx>_<Cylinder|Bound|Fluid>.csv` to
`output_dir`. Every particle below the top fluid layer gets the hydrostatic
density of its depth, the others `ρ₀`. Returns the sampled regions with their
densities.
"""
function generate_floating_cylinder_2d(
    output_dir;
    dx = 0.025,
    SimConstants = floating_cylinder_simulation_constants(; dx),
)
    dx > 0 || throw(ArgumentError("particle spacing dx must be positive"))
    shapes = floating_cylinder_2d_shapes(; dx)
    # The conforming cylinder comes first: the lattice keeps half a spacing from it.
    regions = [
        ParticleRegion("Cylinder", shapes.cylinder, Floating; sampling = :conforming),
        ParticleRegion("Bound", shapes.tank, Fixed),
        ParticleRegion("Fluid", shapes.water, Fluid),
    ]
    sampled = sample_particles(regions, dx)
    water_level = maximum(last, only(r for r in sampled if r.type == Fluid).positions)
    particles = map(sampled) do region
        density = hydrostatic_density(region.positions, SimConstants; water_level)
        (; region..., density)
    end

    mkpath(output_dir)
    SavePolygonVTKHDF(joinpath(output_dir, "FloatingCylinder2D_Geometry.vtkhdf"), shapes)
    prefix = joinpath(output_dir, "FloatingCylinder2D_Dp$(dx)")
    next_id = 0
    for region in particles
        next_id = write_particle_csv(
            "$(prefix)_$(region.name).csv",
            region.positions;
            density = region.density,
            first_id = next_id,
        )
    end

    (; ρ₀, c₀) = SimConstants
    positions = SVector{3, Float64}[]
    density   = Float64[]
    types     = Int8[]
    markers   = Int[]
    for (k, region) in enumerate(particles)
        append!(positions, to_3d(region.positions))
        append!(density, region.density)
        append!(types, fill(Int8(region.type), length(region.positions)))
        append!(markers, fill(k, length(region.positions)))
    end
    pressure = EquationOfStateGamma7.(density, c₀, ρ₀)
    SaveVTKHDF(
        "$(prefix)_Particles.vtkhdf",
        positions,
        ["Density", "Pressure", "Type", "GroupMarker"],
        density,
        pressure,
        types,
        markers,
    )
    return particles
end

# Including this file starts the simulation.
run_floating_cylinder_2d()
