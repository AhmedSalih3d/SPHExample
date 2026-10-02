# GPU version of example/MovingSquare2d.jl. Requires an NVIDIA GPU with CUDA.
#
# Run from the repository root with:
#     julia --project=gpu_version gpu_version/example/MovingSquare2d.jl
#
# Generate the input particles first (no GPU required):
#     julia --project=gpu_version gpu_version/example/GenerateMovingSquare2D.jl
#
# FloatType = Float32 is usually 2-4x faster than Float64 on consumer and
# laptop GPUs (their double precision throughput is low); Float64 reproduces
# the CPU results to round-off.
import StaticArrays: SVector
using SPHExampleGPU

let
    Dimensions = 2
    FloatType  = Float32

    # ViscoBoundFactor should be 1, but need to understand how to implement it
    SimConstantsMovingSquare = SimulationConstants{FloatType}(dx=0.02,
        c₀=28, 
        δᵩ = 0.1,
        g  = 0,
        Cb = 112000,
        α  = 1e-6,
        CFL=0.2
    )

    SimMetaDataMovingSquare  = SimulationMetaData{Dimensions,FloatType,PlanarShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName="MovingSquare2D", 
        SaveLocation="C:/TestSimulations/MovingSquare2D_GPU",
        SimulationTime=2.5,
        OutputTimes=0.01,
        VisualizeInParaview=true,
        ExportSingleVTKHDF=true,
        OpenLogFile=true
    )
    moving_square_input_dir = normpath(joinpath(@__DIR__, "..", "input",
                                                "moving_square_2d_generated"))
    FixedBoundary = SPHGeometry{Dimensions, FloatType}(
        CSVFile     = joinpath(moving_square_input_dir,
            "MovingSquare2D_Dp$(SimConstantsMovingSquare.dx)_Fixed.csv"),
        GroupMarker = 1,
        Type        = Fixed,
        Motion      = nothing
    )
    
    Water = SPHGeometry{Dimensions, FloatType}(
        CSVFile     = joinpath(moving_square_input_dir,
            "MovingSquare2D_Dp$(SimConstantsMovingSquare.dx)_Fluid.csv"),
        GroupMarker = 2,
        Type        = Fluid,
        Motion      = nothing
    )
    
    MovingSquare = SPHGeometry{Dimensions, FloatType}(
        CSVFile     = joinpath(moving_square_input_dir,
            "MovingSquare2D_Dp$(SimConstantsMovingSquare.dx)_Square.csv"),
        GroupMarker = 3,
        Type        = Moving,
        Motion      = MotionDetails{Dimensions, FloatType}(
            Velocity  = 2.8,
            StartTime = 0.0,
            Duration  = 3.0,
            Direction = SVector{Dimensions, FloatType}(1.0, 0.0)  # 2D direction vector with Float64 type
        )
    )

    SimulationGeometry = [FixedBoundary;Water;MovingSquare]

    
    # Collect SPHGeometry instances into a vector
    SimulationGeometry = [FixedBoundary, Water, MovingSquare]
    # If save directory is not already made, make it
    if !isdir(SimMetaDataMovingSquare.SaveLocation)
        mkpath(SimMetaDataMovingSquare.SaveLocation)
    end

    SimLogger = SimulationLogger(SimMetaDataMovingSquare.SaveLocation; to_console=true)
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaDataMovingSquare)

    SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); dx = SimConstantsMovingSquare.dx, k  = FloatType(sqrt(2)))

    CleanUpSimulationFolder(SimMetaDataMovingSquare.SaveLocation)

    RunSimulation(
        SimGeometry         = SimulationGeometry,
        SimMetaData         = SimMetaDataMovingSquare,
        SimConstants        = SimConstantsMovingSquare,
        SimLogger           = SimLogger,
        SimParticles        = SimParticles,
        SimKernel           = SimKernel,
        SimViscosity        = LaminarSPS(),
        SimDensityDiffusion = ZeroGravityLinearDensityDiffusion(),
        SimTimeStepping     = SymplecticTimeStepping()
    )
end
