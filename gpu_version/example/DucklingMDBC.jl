# GPU version of example/DucklingMDBC.jl. Requires an NVIDIA GPU with CUDA.
#
# Run from the repository root with:
#     julia --project=gpu_version gpu_version/example/DucklingMDBC.jl
#
# FloatType = Float32 is usually 2-4x faster than Float64 on consumer and
# laptop GPUs (their double precision throughput is low); Float64 reproduces
# the CPU results to round-off.
using SPHExampleGPU

let
    Dimensions = 3
    FloatType  = Float32
    
    SimConstantsWedge = SimulationConstants{FloatType}(dx=0.01,c₀=23.43842998154953, δᵩ = 0.1, CFL=0.2, α=0.02, m₀=0.001)

    # Assuming SimConstantsWedge is defined somewhere else with the field `dx`
    FixedBoundary = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/case_duckling_mdbc/CaseDuckling_Dp$(SimConstantsWedge.dx)_Bound_MDBC.csv",
        GroupMarker = 1,
        Type        = Fixed,   # Using the enum value Fixed
        Motion      = nothing
    )

    Water = Geometry{Dimensions, FloatType}(
        CSVFile     = "./input/case_duckling_mdbc/CaseDuckling_Dp$(SimConstantsWedge.dx)_Fluid_MDBC.csv",
        GroupMarker = 2,
        Type        = Fluid,   # Using the enum value Fluid
        Motion      = nothing
    )

    SimulationGeometry = [FixedBoundary;Water]
    
    # Load in particles
    SimParticles = AllocateDataStructures(SimulationGeometry)

    SimMetaDataWedge  = SimulationMetaData{Dimensions,FloatType}(
        SimulationName="CaseDuckling", 
        SaveLocation="C:/TestSimulations/Duckling_GPU",
        SimulationTime=1,
        OutputTimes=0.02,
        VisualizeInParaview=true,
        ExportSingleVTKHDF=true,
        ExportGridCells= true,
        OpenLogFile=true,
        FlagOutputKernelValues=false,
        FlagLog=true,
        FlagShifting=false,
        FlagMDBCSimple=true,
    )

    SimKernel           = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); dx = SimConstantsWedge.dx, k = FloatType(1.5))
    SimViscosity        = ArtificialViscosity()
    SimDensityDiffusion = LinearDensityDiffusion()

    mkpath(SimMetaDataWedge.SaveLocation)

    SimLogger = SimulationLogger(SimMetaDataWedge.SaveLocation)

    CleanUpSimulationFolder(SimMetaDataWedge.SaveLocation)

    
    RunSimulation(
        SimGeometry         = SimulationGeometry,
        SimMetaData         = SimMetaDataWedge,
        SimConstants        = SimConstantsWedge,
        SimLogger           = SimLogger,
        SimParticles        = SimParticles,
        SimKernel           = SimKernel,
        SimViscosity        = SimViscosity,
        SimDensityDiffusion = SimDensityDiffusion,
        ParticleNormalsPath = "./input/case_duckling_mdbc/CaseDuckling_Dp$(SimConstantsWedge.dx)_GhostNodes.csv"
    )

    return SimParticles
end



