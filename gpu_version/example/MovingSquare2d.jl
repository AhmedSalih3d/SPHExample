# GPU version of example/MovingSquare2d.jl. Requires an NVIDIA GPU with CUDA.
#
# Run from the repository root with:
#     julia --project=gpu_version gpu_version/example/MovingSquare2d.jl
#
# Particle generation runs below; optional CSV export (no GPU required):
#     julia --project=gpu_version gpu_version/example/GenerateMovingSquare2D.jl
#
# FloatType = Float32 is usually 2-4x faster than Float64 on consumer and
# laptop GPUs (their double precision throughput is low); Float64 reproduces
# the CPU results to round-off.
import StaticArrays: SVector
using SPHExampleGPU
using StructArrays
include(joinpath(@__DIR__, "GenerateMovingSquare2D.jl"))

let
    Dimensions = 2
    FloatType = Float32

    # ViscoBoundFactor should be 1, but need to understand how to implement it
    SimConstantsMovingSquare = SimulationConstants{FloatType}(
        dx = 0.02,
        c₀ = 28,
        δᵩ = 0.1,
        g = 0,
        Cb = 112000,
        α = 1e-6,
        CFL = 0.2,
    )

    SimMetaDataMovingSquare = SimulationMetaData{
        Dimensions,
        FloatType,
        PlanarShifting,
        NoKernelOutput,
        NoMDBC,
        StoreLog,
    }(
        SimulationName = "MovingSquare2D",
        SaveLocation = "C:/TestSimulations/MovingSquare2D_GPU",
        SimulationTime = 2.5,
        OutputTimes = 0.01,
        VisualizeInParaview = true,
        ExportSingleVTKHDF = true,
        OpenLogFile = true,
    )
    moving_square_input_dir =
        normpath(joinpath(@__DIR__, "..", "input", "moving_square_2d_generated"))

    # Sample particles here; the generator script supplies shape definitions.
    polygons = moving_square_2d_polygons()
    regions = [
        ParticleRegion("Fixed", polygons.tank, Fixed),
        ParticleRegion("Square", polygons.square, Moving),
        ParticleRegion("Fluid", polygons.water, Fluid),
    ]
    sampled = sample_particles(regions, SimConstantsMovingSquare.dx)
    positions(name) = only(r.positions for r in sampled if r.name == name)

    FixedBoundary = SPHGeometry{Dimensions, FloatType}(
        Particles = particle_struct_array(positions("Fixed"), SimConstantsMovingSquare.ρ₀),
        # CSVFile     = joinpath(moving_square_input_dir,
        # "MovingSquare2D_Dp$(SimConstantsMovingSquare.dx)_Fixed.csv"),
        GroupMarker = 1,
        Type = Fixed,
        Motion = nothing,
    )

    Water = SPHGeometry{Dimensions, FloatType}(
        Particles = particle_struct_array(positions("Fluid"), SimConstantsMovingSquare.ρ₀),
        # CSVFile     = joinpath(moving_square_input_dir,
        # "MovingSquare2D_Dp$(SimConstantsMovingSquare.dx)_Fluid.csv"),
        GroupMarker = 2,
        Type = Fluid,
        Motion = nothing,
    )

    MovingSquare = SPHGeometry{Dimensions, FloatType}(
        Particles = particle_struct_array(positions("Square"), SimConstantsMovingSquare.ρ₀),
        # CSVFile     = joinpath(moving_square_input_dir,
        # "MovingSquare2D_Dp$(SimConstantsMovingSquare.dx)_Square.csv"),
        GroupMarker = 3,
        Type = Moving,
        Motion = MotionDetails{Dimensions, FloatType}(
            Velocity = 2.8,
            StartTime = 0.0,
            Duration = 3.0,
            Direction = SVector{Dimensions, FloatType}(1.0, 0.0),  # 2D direction vector with Float64 type
        ),
    )

    SimulationGeometry = [FixedBoundary; Water; MovingSquare]

    # Collect SPHGeometry instances into a vector
    SimulationGeometry = [FixedBoundary, Water, MovingSquare]
    # If save directory is not already made, make it
    if !isdir(SimMetaDataMovingSquare.SaveLocation)
        mkpath(SimMetaDataMovingSquare.SaveLocation)
    end

    SimLogger = SimulationLogger(SimMetaDataMovingSquare.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaDataMovingSquare)

    SimKernel = SPHKernelInstance{Dimensions, FloatType}(
        WendlandC2();
        dx = SimConstantsMovingSquare.dx,
        k = FloatType(sqrt(2)),
    )

    CleanUpSimulationFolder(SimMetaDataMovingSquare.SaveLocation)

    motion_preview = joinpath(SimMetaDataMovingSquare.SaveLocation, "MovingSquare2D_Motion.vtkhdf")
    SavePolygonMotionSequence(
        motion_preview,
        (; polygons.tank, polygons.square);
        motions = (; square = MovingSquare.Motion),
        times = 0.0:0.1:3.0,
    )
    OpenParaviewFile(motion_preview)

    RunSimulation(
        SimGeometry = SimulationGeometry,
        SimMetaData = SimMetaDataMovingSquare,
        SimConstants = SimConstantsMovingSquare,
        SimLogger = SimLogger,
        SimParticles = SimParticles,
        SimKernel = SimKernel,
        SimViscosity = LaminarSPS(),
        SimDensityDiffusion = ZeroGravityLinearDensityDiffusion(),
        SimTimeStepping = SymplecticTimeStepping(),
    )
end
