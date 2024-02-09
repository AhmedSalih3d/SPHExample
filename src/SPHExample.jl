module SPHExample

    include("AuxillaryFunctions.jl"); 
    include("PreProcess.jl");         
    include("PostProcess.jl");        
    include("TimeStepping.jl");       
    include("SimulationEquations.jl");
    include("SimulationMetaDataConfiguration.jl");
    include("SimulationConstantsConfiguration.jl");
    include("SimulationDataArrays.jl")
    
    # Re-export desired functions from each submodule
    using .PreProcess
    export LoadParticlesFromCSV

    using .PostProcess
    export create_vtp_file, OutputVTP

    using .TimeStepping: Δt
    export Δt

    using .SimulationEquations
    export Wᵢⱼ, ∑ⱼWᵢⱼ!, Optim∇ᵢWᵢⱼ, ∑ⱼ∇ᵢWᵢⱼ!, Pressure, ∂Πᵢⱼ∂t!, ∂ρᵢ∂tDDT!, ∂vᵢ∂t!, DensityEpsi!, LimitDensityAtBoundary!, updatexᵢⱼ!

    using .SimulationMetaDataConfiguration
    export SimulationMetaData

    using .SimulationConstantsConfiguration
    export SimulationConstants

    using .SimulationDataArrays
    export ResetArrays!, ResizeBuffers!
end

