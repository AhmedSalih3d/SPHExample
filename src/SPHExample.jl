module SPHExample

    include("AuxillaryFunctions.jl");        
    include("PostProcess.jl");    
    include("ProduceVTP.jl")    
    include("TimeStepping.jl");       
    include("SimulationEquations.jl");
    include("SimulationMetaDataConfiguration.jl");
    include("SimulationConstantsConfiguration.jl");
    include("SimulationDataArrays.jl")
    include("PreProcess.jl");
    include("SPHCellList.jl")
    
    # Re-export desired functions from each submodule
    using .AuxillaryFunctions
    export RearrangeVector!

    using .PreProcess
    export LoadParticlesFromCSV, LoadParticlesFromCSV_StaticArrays, LoadBoundaryNormals

    using .PostProcess
    export create_vtp_file, OutputVTP

    using .ProduceVTP
    export ExportVTP

    using .TimeStepping: Δt
    export Δt

    using .SimulationEquations
    export EquationOfState, EquationOfStateGamma7, Pressure!, DensityEpsi!, LimitDensityAtBoundary!, ConstructGravitySVector, InverseHydrostaticEquationOfState

    using .SimulationMetaDataConfiguration
    export SimulationMetaData

    using .SimulationConstantsConfiguration
    export SimulationConstants

    using .SimulationDataArrays
    export ResetArrays!, ResizeBuffers!

    using .SPHCellList
    export ConstructStencil, ExtractCells!, UpdateNeighbors!, NeighborLoop!, ComputeInteractions!
end

