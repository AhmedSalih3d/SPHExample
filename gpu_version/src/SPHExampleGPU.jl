module SPHExampleGPU

    # CUDA port of SPHExample. The public API mirrors the CPU package so that the
    # example scripts only need `using SPHExampleGPU` instead of `using SPHExample`.
    # Particle data is loaded on the host (CSV), uploaded to the GPU once and
    # only copied back for output. All time stepping runs on the GPU.

    using CUDA

    # Include submodules in dependency order
    submodules = [
        "AuxiliaryFunctions.jl",
        "SPHKernels.jl",
        "SPHViscosityModels.jl",
        "ProduceHDFVTK.jl",
        "TimeStepping.jl",
        "SimulationEquations.jl",
        "SimulationGeometry.jl",
        "SimulationMetaDataConfiguration.jl",
        "SimulationConstantsConfiguration.jl",
        "SimulationLoggerConfiguration.jl",
        "PreProcess.jl",
        "OpenExternalPrograms.jl",
        "SPHDensityDiffusionModels.jl",
        "GPUReductions.jl",
        "GPUCellGrid.jl",
        "GPUKernels.jl",
        "SPHCellList.jl",
        "CustomPrettyPrinting.jl",
    ]
    foreach(include, submodules)

    # Re-export desired functions from each submodule
    using .AuxiliaryFunctions
    export ResetArrays!, to_3d, CloseHDFVTKManually, CleanUpSimulationFolder

    using .SPHKernels
    export SPHKernel, SPHKernelInstance, WendlandC2, CubicSpline, Wᵢⱼ, ∇Wᵢⱼ, tensile_correction

    using .SPHViscosityModels
    export SPHViscosity, ZeroViscosity, ArtificialViscosity, Laminar, LaminarSPS, compute_viscosity

    using .SPHDensityDiffusionModels
    export SPHDensityDiffusion, ZeroDensityDiffusion, ZeroGravityLinearDensityDiffusion, LinearDensityDiffusion, ZeroGravityComplexDensityDiffusion, ComplexDensityDiffusion, compute_density_diffusion

    using .SimulationGeometry
    export ParticleType, Fixed, Fluid, Moving, Geometry, MotionDetails

    using .PreProcess
    export AllocateDataStructures, AllocateSupportDataStructures, LoadBoundaryNormals

    using .ProduceHDFVTK
    export SaveVTKHDF, GenerateGeometryStructure, GenerateStepStructure, AppendVTKHDFData, SaveCellGridVTKHDF, AppendVTKHDFGridData, SetupVTKOutput

    using .TimeStepping: Δt
    export Δt

    using .SimulationEquations
    export EquationOfState, EquationOfStateGamma7, Pressure!, DensityEpsi!, LimitDensityAtBoundary!, ConstructGravitySVector, InverseHydrostaticEquationOfState, Estimate7thRoot

    using .SimulationLoggerConfiguration
    export SimulationLogger, generate_format_string, InitializeLogger, LogSimulationDetails, LogStep, LogFinal

    using .SimulationMetaDataConfiguration
    export SimulationMetaData

    using .SimulationConstantsConfiguration
    export SimulationConstants

    using .GPUReductions
    export ReductionWorkspace, reduce_svector

    using .GPUCellGrid
    export CellGrid, CellListWorkspace, update_cell_list!, unique_cells_host, map_floor

    using .GPUKernels
    export launch_interactions!, launch_mdbc!, launch_motion!, launch_half_step!, launch_final_step!, choose_lanes

    using .SPHCellList
    export GPUParticles, GPUSupportArrays, MotionArrays, upload_particles, download_particles!,
           RunSimulation, SimulationLoop, StepReduction

    using .OpenExternalPrograms
    export AutoOpenLogFile, AutoOpenParaview

end
