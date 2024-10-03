module SPHExample

    include("AuxillaryFunctions.jl");          
    include("ProduceHDFVTK.jl")    
    include("TimeStepping.jl");       
    include("SimulationEquations.jl");
    include("SimulationGeometry.jl")
    include("SimulationMetaDataConfiguration.jl");
    include("SimulationConstantsConfiguration.jl");
    include("SimulationLoggerConfiguration.jl");
    include("PreProcess.jl");
    include("OpenExternalPrograms.jl")
    include("SPHCellList.jl") #Must be last

    # Re-export desired functions from each submodule
    using .AuxillaryFunctions
    export ResetArrays!, to_3d, CloseHDFVTKManually, CleanUpSimulationFolder

    using .SimulationGeometry
    export ParticleType, Fixed, Fluid, Moving, Geometry, MotionDetails

    using .PreProcess
    export AllocateDataStructures, LoadBoundaryNormals

    using .ProduceHDFVTK
    export SaveVTKHDF

    using .TimeStepping: Δt
    export Δt

    using .SimulationEquations
    export EquationOfState, EquationOfStateGamma7, Pressure!, DensityEpsi!, LimitDensityAtBoundary!, ConstructGravitySVector, InverseHydrostaticEquationOfState

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

