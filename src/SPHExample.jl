module SPHExample

    include("AuxillaryFunctions.jl"); 
    include("PreProcess.jl");         
    include("PostProcess.jl");        
    include("TimeStepping.jl");       
    include("SimulationEquations.jl");
    
    # Re-export desired functions from each submodule
    using .PreProcess: LoadParticlesFromCSV
    export LoadParticlesFromCSV

    using .PostProcess: create_vtp_file
    export create_vtp_file

    using .TimeStepping: Δt
    export Δt

    using .SimulationEquations: Wᵢⱼ, ∑ⱼWᵢⱼ, Optim∇ᵢWᵢⱼ, ∑ⱼ∇ᵢWᵢⱼ!, Pressure, ∂Πᵢⱼ∂t!, ∂ρᵢ∂tDDT!, ∂vᵢ∂t!, updatexᵢⱼ!, resizebuffers!
    export Wᵢⱼ, ∑ⱼWᵢⱼ, Optim∇ᵢWᵢⱼ, ∑ⱼ∇ᵢWᵢⱼ!, Pressure, ∂Πᵢⱼ∂t!, ∂ρᵢ∂tDDT!, ∂vᵢ∂t!, updatexᵢⱼ!, resizebuffers!

end

