module SPHExample

    include("AuxillaryFunctions.jl");        
    include("PostProcess.jl");    
    include("ProduceVTP.jl")    
    include("TimeStepping.jl");       
    include("SimulationEquations.jl");
    include("SimulationMetaDataConfiguration.jl");
    include("SimulationConstantsConfiguration.jl");
    include("SimulationLoggerConfiguration.jl");
    include("PreProcess.jl");
    include("SPHCellList.jl")
    
    # Re-export desired functions from each submodule
    using .AuxillaryFunctions
    export ResetArrays!, to_3d

    using .PreProcess
    export LoadParticlesFromCSV_StaticArrays, AllocateDataStructures, LoadBoundaryNormals

    using .PostProcess
    export create_vtp_file, OutputVTP

    using .ProduceVTP
    export ExportVTP, ConvertHDFtoVTP, SaveHDF5!, HDFtoVTP, OpenForWriteH5

    using .TimeStepping: Δt
    export Δt

    using .SimulationEquations
    export EquationOfState, EquationOfStateGamma7, Pressure!, DensityEpsi!, LimitDensityAtBoundary!, ConstructGravitySVector, InverseHydrostaticEquationOfState

    using .SimulationLoggerConfiguration
    export SimulationLogger, generate_format_string, InitializeLogger, LogStep, LogFinal

    using .SimulationMetaDataConfiguration
    export SimulationMetaData

    using .SimulationConstantsConfiguration
    export SimulationConstants

    using .SPHCellList
    export ConstructStencil, ExtractCells!, UpdateNeighbors!, NeighborLoop!, ComputeInteractions!

end

