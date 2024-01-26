module SPHExample

    include("AuxillaryFunctions.jl"); 
    include("PreProcess.jl");         
    include("PostProcess.jl");        
    include("TimeStepping.jl");       
    include("SimulationEquations.jl");
    
    # Re-export desired functions from each submodule
    using .PreProcess: LoadParticlesFromCSV
    export LoadParticlesFromCSV

end

