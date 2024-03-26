module SPHExample

    include("AuxillaryFunctions.jl");        
    include("PostProcess.jl");    
    include("ProduceVTP.jl")    
    include("TimeStepping.jl");       
    include("SimulationEquations.jl");
    include("SimulationMetaDataConfiguration.jl");
    include("SimulationConstantsConfiguration.jl");
    include("PreProcess.jl");
    include("SPHCellList.jl")
    
    # Re-export desired functions from each submodule
    using .AuxillaryFunctions
    export RearrangeVector!, ResetArrays!, to_3d

    using .PreProcess
    export LoadParticlesFromCSV_StaticArrays, LoadBoundaryNormals

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

    using .SPHCellList
    export ConstructStencil, ExtractCells!, UpdateNeighbors!, NeighborLoop!, ComputeInteractions!

    # Temp
    using Parameters, StaticArrays
    @with_kw struct Particle{D,T}
        Cell::CartesianIndex{D}
        Position::SVector{D,T}
        Acceleration::SVector{D,T}
        Velocity::SVector{D,T} 
        Density::T
        GravityFactor::T
        MotionLimiter::T
        ID::Int64
    end

    export Particle
end

