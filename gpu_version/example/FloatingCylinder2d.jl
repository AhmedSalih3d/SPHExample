# GPU version of the DualSPHysics 2D falling cylinder (examples/main/11_Floating,
# CaseFloatingSphereVal2D). Requires an NVIDIA GPU with CUDA.
#
# A floating cylinder of radius 1 m and relative weight 1.2 starts half
# submerged in a 10 m wide, 14 m deep tank. It is held still for the first
# second (DualSPHysics `FtPause`) while the water settles, then sinks. The
# periodic sides of the DualSPHysics case are fixed walls here.
#
# Particle generation runs below. Optional CSV export and simulation:
#     julia --project=gpu_version gpu_version/example/GenerateFloatingCylinder2D.jl
#     julia --project=gpu_version gpu_version/example/FloatingCylinder2d.jl
#
# Besides the particle output, `<SimulationName>_Floating.csv` in the save
# location holds the centre, velocity, angle and angular velocity of the
# cylinder at every output time. The DualSPHysics validation data (Fekken 2004,
# Moyo and Greenhow 2000) give the sinking distance and velocity against
# t* = (t - 1 s) sqrt(g / R) for t* <= 8.
using SPHExampleGPU
using StructArrays
include(joinpath(@__DIR__, "GenerateFloatingCylinder2D.jl"))

let
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
        OpenLogFile = true,
        # positions up to 16 m: integrate them in Float64, the rest in Float32
        GPUDoublePosition = true,
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
        SimTimeStepping = SymplecticTimeStepping(),
    )
end
