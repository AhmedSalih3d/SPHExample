module SPHExample

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
    export AllocateDataStructures, AllocateSupportDataStructures, AllocateThreadedArrays, LoadBoundaryNormals

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

    using .SPHCellList
    export ConstructStencil, ExtractCells!, UpdateNeighbors!, NeighborLoop!, ComputeInteractions!, RunSimulation

    using .OpenExternalPrograms
    export AutoOpenLogFile, AutoOpenParaview

end

