# GPU version of the Duckling case with SPH measurements. Requires an NVIDIA
# GPU with CUDA and the input files under input/case_duckling_mdbc.
#
# Run from the repository root with:
#     julia --project=gpu_version gpu_version/example/DucklingMeasurements.jl
#
# Pressure, velocity, water-column and free-surface data are written as
# blocks next to the particles in the combined VTKHDF output (a
# MultiBlockDataSet; see the `Measurements` node of the block hierarchy in
# ParaView).
using SPHExampleGPU

let
    Dimensions = 3
    FloatType = Float32
    InputDirectory = joinpath(@__DIR__, "..", "..", "input", "case_duckling_mdbc")

    SimConstantsDuckling = SimulationConstants{FloatType}(
        dx = 0.01,
        c₀ = 23.43842998154953,
        δᵩ = 0.1,
        CFL = 0.2,
        α = 0.02,
        m₀ = 0.001,
    )

    FixedBoundary = SPHGeometry{Dimensions, FloatType}(
        CSVFile = joinpath(
            InputDirectory,
            "CaseDuckling_Dp$(SimConstantsDuckling.dx)_Bound_MDBC.csv",
        ),
        GroupMarker = 1,
        Type = Fixed,
    )

    Water = SPHGeometry{Dimensions, FloatType}(
        CSVFile = joinpath(
            InputDirectory,
            "CaseDuckling_Dp$(SimConstantsDuckling.dx)_Fluid_MDBC.csv",
        ),
        GroupMarker = 2,
        Type = Fluid,
    )

    SimulationGeometry = [FixedBoundary; Water]

    SimMeasurements = MeasurementConfig(
        pressure_probes = [
            MeasurementProbe("front_pressure", (0.15, 0.25, 0.10)),
            MeasurementProbe("wake_pressure", (0.35, 0.25, 0.10)),
        ],
        velocity_probes = [
            MeasurementProbe("front_velocity", (0.15, 0.25, 0.10)),
            MeasurementProbe("wake_velocity", (0.35, 0.25, 0.10)),
        ],
        water_column_probes = [
            WaterColumnProbe("left_column", (0.10, 0.25, 0.0); radius = 0.05),
            WaterColumnProbe("center_column", (0.25, 0.25, 0.0)),
            WaterColumnProbe("right_column", (0.40, 0.25, 0.0); radius = 0.05),
        ],
        free_surface = FreeSurfaceDomain(
            (0.0, 0.0, 0.0), (0.5, 0.5, 0.0), 0.05,
        ),
        vertical_axis = 3,
        sample_every = 5,
    )

    SimMetaDataDuckling = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        SimpleMDBC,
        StoreLog,
    }(
        SimulationName = "DucklingMeasurements",
        SaveLocation = joinpath(@__DIR__, "..", "output", "DucklingMeasurements"),
        SimulationTime = 1.0,
        OutputTimes = 0.02,
        VisualizeInParaview = true,
        ExportSingleVTKHDF = true,
        ExportGridCells = false,
        OpenLogFile = false,
    )

    SimKernel = SPHKernelInstance{Dimensions, FloatType}(
        WendlandC2();
        dx = SimConstantsDuckling.dx,
        k = FloatType(1.5),
    )
    SimViscosity = ArtificialViscosity()
    SimDensityDiffusion = LinearDensityDiffusion()
    SimTimeStepping = SymplecticTimeStepping()

    mkpath(SimMetaDataDuckling.SaveLocation)
    SimLogger = SimulationLogger(SimMetaDataDuckling.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaDataDuckling)

    CleanUpSimulationFolder(SimMetaDataDuckling.SaveLocation)

    RunSimulation(
        SimGeometry = SimulationGeometry,
        SimMetaData = SimMetaDataDuckling,
        SimConstants = SimConstantsDuckling,
        SimLogger = SimLogger,
        SimParticles = SimParticles,
        SimKernel = SimKernel,
        SimViscosity = SimViscosity,
        SimDensityDiffusion = SimDensityDiffusion,
        SimTimeStepping = SimTimeStepping,
        ParticleNormalsPath = joinpath(
            InputDirectory,
            "CaseDuckling_Dp$(SimConstantsDuckling.dx)_GhostNodes.csv",
        ),
        SimMeasurements = SimMeasurements,
    )

    return SimParticles
end
