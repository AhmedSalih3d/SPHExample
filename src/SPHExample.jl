module SPHExample

    include("AuxillaryFunctions.jl"); 
    include("PreProcess.jl");
    using .PreProcess       
    include("PostProcess.jl");        
    include("TimeStepping.jl");       
    include("SimulationEquations.jl");
    
    

end

