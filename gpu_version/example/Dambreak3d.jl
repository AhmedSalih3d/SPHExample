# GPU version of example/Dambreak3d.jl. Requires an NVIDIA GPU with CUDA.
#
# Run from the repository root with:
#     julia --project=gpu_version gpu_version/example/Dambreak3d.jl
#
# FloatType = Float32 is usually 2-4x faster than Float64 on consumer and
# laptop GPUs (their double precision throughput is low); Float64 reproduces
# the CPU results to round-off.
using SPHExampleGPU

let
    Dimensions = 3
    FloatType  = Float32

    # --- SPH constants ---
    dx = 0.0085
    SimConstantsDambreak3D = SimulationConstants{FloatType}(
        dx  = dx,
        c₀  = 33.14,
        α   = 0.1,
        m₀  = 1000 * dx^3,
        CFL = 0.2
    )

    # --- Geometry ---
    FixedBoundary = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/dam_break_3d/DamBreak3d_Dp$(dx)_Bound.csv",
        GroupMarker = 1,
        Type        = Fixed,
        Motion      = nothing
    )
    Water = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/dam_break_3d/DamBreak3d_Dp$(dx)_Fluid.csv",
        GroupMarker = 2,
        Type        = Fluid,
        Motion      = nothing
    )
    SimulationGeometry = [FixedBoundary; Water]

    # --- Allocate particles ---
    SimParticles = AllocateDataStructures(SimulationGeometry)

    # --- Simulation metadata & logging ---
    SimMetaDataDambreak3D = SimulationMetaData{Dimensions,FloatType}(
        SimulationName         = "DamBreak3D_Test",
        SaveLocation           = "C:/TestSimulations/DamBreak3D_GPU",
        SimulationTime         = 1.6,
        OutputTimes            = 0.01,
        VisualizeInParaview    = true,
        ExportSingleVTKHDF     = true,
        ExportGridCells        = true,
        OpenLogFile            = true,
        FlagOutputKernelValues = false,
        FlagLog                = true
    )

    if !isdir(SimMetaDataDambreak3D.SaveLocation)
        mkpath(SimMetaDataDambreak3D.SaveLocation)
    end

    SimLogger = SimulationLogger(SimMetaDataDambreak3D.SaveLocation; to_console=true)

    @warn("""
    3D mode enabled but lightly tested.
    With only one boundary layer, fluid may leak when the column relaxes.
    """)

    CleanUpSimulationFolder(SimMetaDataDambreak3D.SaveLocation)

    # --- Kernel, viscosity & diffusion ---
    SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); h   = FloatType(1 * sqrt(3 * dx^2)))
    SimViscosity        = ArtificialViscosity()
    SimDensityDiffusion = LinearDensityDiffusion()

    # --- Run simulation ---
    RunSimulation(
        SimGeometry        = SimulationGeometry,
        SimMetaData        = SimMetaDataDambreak3D,
        SimConstants       = SimConstantsDambreak3D,
        SimKernel          = SimKernel,
        SimLogger          = SimLogger,
        SimParticles       = SimParticles,
        SimViscosity       = SimViscosity,
        SimDensityDiffusion= SimDensityDiffusion
    )
end
