module SPHExample

    include("AuxiliaryFunctions.jl");
    include("SPHKernels.jl")
    include("SPHViscosityModels.jl")      
    include("ProduceHDFVTK.jl")    
    include("TimeStepping.jl");       
    include("SimulationEquations.jl");
    include("SimulationGeometry.jl")
    include("SimulationMetaDataConfiguration.jl");
    include("SimulationConstantsConfiguration.jl");
    include("SimulationLoggerConfiguration.jl");
    include("PreProcess.jl");
    include("OpenExternalPrograms.jl")
    include("SPHDensityDiffusionModels.jl")  
    include("SPHCellList.jl") #Must be last    

    using Reexport

    # Re-export desired functions from each submodule
    @reexport using .AuxiliaryFunctions

    @reexport using .SPHKernels

    @reexport using .SPHViscosityModels

    @reexport using .SPHDensityDiffusionModels

    @reexport using .SimulationGeometry

    @reexport using .PreProcess

    @reexport using .ProduceHDFVTK

    @reexport using .TimeStepping

    @reexport using .SimulationEquations

    @reexport using .SimulationLoggerConfiguration

    @reexport using .SimulationMetaDataConfiguration

    @reexport using .SimulationConstantsConfiguration

    @reexport using .SPHCellList

    @reexport using .OpenExternalPrograms


end

