module SPHExample

    include("AuxillaryFunctions.jl");        
    include("PostProcess.jl");        
    include("TimeStepping.jl");       
    include("SimulationEquations.jl");
    include("SimulationMetaDataConfiguration.jl");
    include("SimulationConstantsConfiguration.jl");
    include("SimulationDataArrays.jl")
    include("PreProcess.jl");  
    
    # Re-export desired functions from each submodule
    using .PreProcess
    export LoadParticlesFromCSV, LoadParticlesFromCSV_StaticArrays, LoadBoundaryNormals

    using .PostProcess
    export create_vtp_file, OutputVTP

    using .TimeStepping: Δt
    export Δt

    using .SimulationEquations
    export Wᵢⱼ, ∑ⱼWᵢⱼ!, Optim∇ᵢWᵢⱼ, ∑ⱼWᵢⱼ!∑ⱼ∇ᵢWᵢⱼ!, EquationOfState, Pressure!, ∂Πᵢⱼ∂t!, ∂ρᵢ∂t!, ∂ρᵢ∂tDDT!, ∂vᵢ∂t!, DensityEpsi!, LimitDensityAtBoundary!, updatexᵢⱼ!, ArtificialViscosityMomentumEquation!, DimensionalData, Optim∇ᵢWᵢⱼ

    using .SimulationMetaDataConfiguration
    export SimulationMetaData

    using .SimulationConstantsConfiguration
    export SimulationConstants

    using .SimulationDataArrays
    export ResetArrays!, ResizeBuffers!
end

