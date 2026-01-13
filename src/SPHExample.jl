module SPHExample

    include("AuxiliaryFunctions.jl");
    include("SPHKernels.jl")
    include("SPHViscosityModels.jl")      
    include("ProduceHDFVTK.jl")    
    include("SimulationGeometry.jl")
    include("SimulationMetaDataConfiguration.jl");
    include("SimulationEquations.jl");
    include("TimeStepping.jl");       
    include("SimulationConstantsConfiguration.jl");
    include("SimulationLoggerConfiguration.jl");
    include("PreProcess.jl");
    include("OpenExternalPrograms.jl")
    include("SPHDensityDiffusionModels.jl")  
    include("SPHNeighborList.jl")
    include("SPHCellList.jl") #Must be last    

    # Re-export desired functions from each submodule
    using .AuxiliaryFunctions
    export ResetArrays!, to_3d, CloseHDFVTKManually, CleanUpSimulationFolder

    using .SPHKernels
    export SPHKernel, SPHKernelInstance, WendlandC2, CubicSpline, Wᵢⱼ, ∇Wᵢⱼ, tensile_correction

    using .SPHViscosityModels
    export SPHViscosity, ZeroViscosity, ArtificialViscosity, Laminar, LaminarSPS, compute_viscosity

    using .SPHDensityDiffusionModels
    export SPHDensityDiffusion, ZeroDensityDiffusion, ZeroGravityLinearDensityDiffusion, LinearDensityDiffusion, ZeroGravityComplexDensityDiffusion, ComplexDensityDiffusion, compute_density_diffusion
 
    using .SimulationGeometry
    export ParticleType, Fixed, Fluid, Moving, Geometry, MotionDetails

    using .PreProcess
    export AllocateDataStructures, AllocateSupportDataStructures, AllocateThreadedArrays, LoadBoundaryNormals

    using .ProduceHDFVTK
    export SaveVTKHDF, GenerateGeometryStructure, GenerateStepStructure, AppendVTKHDFData, SaveCellGridVTKHDF, AppendVTKHDFGridData, SetupVTKOutput

    using .TimeStepping: Δt
    export Δt

    using .SimulationEquations
    export EquationOfState, EquationOfStateGamma7, Pressure!, DensityEpsi!, LimitDensityAtBoundary!, ConstructGravitySVector, InverseHydrostaticEquationOfState, Estimate7thRoot

    using .SimulationLoggerConfiguration
    export SimulationLogger, generate_format_string, InitializeLogger, LogSimulationDetails, LogStep, LogFinal

    using .SimulationMetaDataConfiguration
    export SimulationMetaData, ShiftingMode, NoShifting, PlanarShifting,
           KernelOutputMode, NoKernelOutput, StoreKernelOutput,
           MDBCMode, NoMDBC, SimpleMDBC,
           LogMode, NoLog, StoreLog

    using .SimulationConstantsConfiguration
    export SimulationConstants

    using .SPHNeighborList
    export ConstructStencil, ExtractCells!, UpdateNeighbors!, BuildNeighborCellLists!, ComputeCellParticleCounts, ComputeCellNeighborCounts

    using .SPHCellList
    export NeighborLoop!, ComputeInteractions!, RunSimulation

    using .OpenExternalPrograms
    export AutoOpenLogFile, AutoOpenParaview

    using PrecompileTools
    using StaticArrays

    @setup_workload begin
        KernelInstance = SPHKernelInstance{2, Float64}(WendlandC2(); dx=0.02)
        Constants = SimulationConstants()
        Position = SVector(0.0, 0.0)
        Density = [Constants.ρ₀, Constants.ρ₀]
        Pressure = similar(Density)
        MotionLimiter = [1.0, 0.0]
        q = 0.5

        @compile_workload begin
            Wᵢⱼ(KernelInstance, q)
            ∇Wᵢⱼ(KernelInstance, q, Position)
            tensile_correction(KernelInstance, 0.0, 1.0, 0.0, 1.0, q, 0.02)
            EquationOfStateGamma7(Constants.ρ₀, Constants.c₀, Constants.ρ₀)
            ConstructGravitySVector(Position, Constants.g)
            Pressure!(Pressure, Density, Constants)
            LimitDensityAtBoundary!(Density, Constants.ρ₀, MotionLimiter)
        end
    end

end
