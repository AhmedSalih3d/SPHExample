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
        "GPUStepState.jl",
        "GPUCellGrid.jl",
        "GPUKernels.jl",
        "SPHCellList.jl",
        "ParticleGenerator.jl",
        "CustomPrettyPrinting.jl",
    ]
    foreach(include, submodules)

    # Re-export desired functions from each submodule
    using .AuxiliaryFunctions
    export to_3d, CloseHDFVTKManually, CleanUpSimulationFolder

    using .SPHKernels
    export SPHKernel, SPHKernelInstance, WendlandC2, CubicSpline, Wᵢⱼ, ∇Wᵢⱼ, tensile_correction

    using .SPHViscosityModels
    export SPHViscosity, ZeroViscosity, ArtificialViscosity, Laminar, LaminarSPS, compute_viscosity

    using .SPHDensityDiffusionModels
    export SPHDensityDiffusion, ZeroDensityDiffusion, ZeroGravityLinearDensityDiffusion, LinearDensityDiffusion, ZeroGravityComplexDensityDiffusion, ComplexDensityDiffusion, compute_density_diffusion

    using .SimulationGeometry
    export ParticleType, Fixed, Fluid, Moving, SPHGeometry, MotionDetails,
           GravityFactorValue, MotionLimiterValue

    using .PreProcess
    export AllocateDataStructures, AllocateSupportDataStructures, LoadBoundaryNormals

    using .ProduceHDFVTK
    export SaveVTKHDF, SavePolygonVTKHDF, GenerateGeometryStructure,
           GenerateStepStructure, AppendVTKHDFData, SaveCellGridVTKHDF,
           AppendVTKHDFGridData, SetupVTKOutput, GridGeometryBuffers,
           fill_grid_geometry!, GridFrameWriter, append_grid_frame!,
           PolyDataFrameWriter, append_frame!, flush_frames!, frames_written,
           frames_pending, buffered_frames, MAX_BUFFERED_FRAMES

    using .TimeStepping: Δt
    export Δt

    using .SimulationEquations
    export EquationOfState, EquationOfStateGamma7, Pressure!, DensityEpsi!, LimitDensityAtBoundary!, ConstructGravitySVector, InverseHydrostaticEquationOfState, Estimate7thRoot

    using .SimulationLoggerConfiguration
    export SimulationLogger, generate_format_string, InitializeLogger, LogSimulationDetails, LogStep, step_log_line, log_line, LogFinal

    using .SimulationMetaDataConfiguration
    export SimulationMetaData, ShiftingMode, NoShifting, PlanarShifting,
           KernelOutputMode, NoKernelOutput, StoreKernelOutput,
           MDBCMode, NoMDBC, SimpleMDBC,
           LogMode, NoLog, StoreLog,
           TimeSteppingMode, SymplecticTimeStepping, SingleNeighborTimeStepping,
           OUTPUT_VARIABLES, DEFAULT_OUTPUT_VARIABLES, resolve_output_variables!, position_float_type

    using .SimulationConstantsConfiguration
    export SimulationConstants

    using .GPUReductions
    export ReductionWorkspace, reduce_svector

    using .GPUStepState
    export StepState, HostStep, readback!

    using .GPUCellGrid
    export CellGrid, CellListWorkspace, update_cell_list!, compact_nonzero!, unique_cells_host, map_floor,
           PosCell, CellRow, cell_rows, cell_size, pos_cell, pair_vector

    using .GPUKernels
    export launch_interactions!, launch_mdbc!, launch_motion!, launch_half_step!, launch_final_step!,
           launch_finish!, launch_commit!, launch_pos_cells!, choose_lanes

    using .SPHCellList
    export GPUParticles, GPUSupportArrays, MotionArrays, upload_particles, download_particles!,
           RunSimulation, SimulationLoop, position_type, uses_pos_cells

    using .ParticleGenerator
    export ParticleRegion, sample_particles, write_particle_csv

    using .OpenExternalPrograms
    export AutoOpenLogFile, AutoOpenParaview

    include("PrecompileWorkload.jl")

end
