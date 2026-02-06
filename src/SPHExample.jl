module SPHExample

    include("AuxiliaryFunctions.jl");
    include("SPHKernels.jl")
    include("SPHViscosityModels.jl")      
    include("SimulationGeometry.jl")
    include("ProduceHDFVTK.jl")    
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
    export ParticleType, Fixed, Fluid, Moving, FixedMoving, Geometry, MotionDetails, MotionPositionFactorValue

    using .PreProcess
    export AllocateDataStructures, AllocateSupportDataStructures, AllocateThreadedArrays, LoadBoundaryNormals

    using .ProduceHDFVTK
    export SaveVTKHDF, GenerateGeometryStructure, GenerateStepStructure, AppendVTKHDFData, SaveCellGridVTKHDF, AppendVTKHDFGridData, SetupVTKOutput

    using .TimeStepping: Δt, FinalizeTimeStep, UpdateTimeStepBuffers!, HalfTimeStep, FullTimeStep
    export Δt

    using .SimulationEquations
    export EquationOfState, EquationOfStateGamma7, Pressure!, DensityEpsi!, LimitDensityAtBoundary!, ConstructGravitySVector, InverseHydrostaticEquationOfState, Estimate7thRoot

    using .SimulationLoggerConfiguration
    export SimulationLogger, generate_format_string, InitializeLogger, LogSimulationDetails, LogStep, LogFinal

    using .SimulationMetaDataConfiguration
    export SimulationMetaData, ShiftingMode, NoShifting, PlanarShifting,
           KernelOutputMode, NoKernelOutput, StoreKernelOutput,
           MDBCMode, NoMDBC, SimpleMDBC,
           LogMode, NoLog, StoreLog,
           TimeSteppingMode, SymplecticTimeStepping, SingleNeighborTimeStepping

    using .SimulationConstantsConfiguration
    export SimulationConstants

    using .SPHNeighborList
    export ConstructStencil, ExtractCells!, UpdateNeighbors!, BuildNeighborCellLists!, ComputeCellParticleCounts, ComputeCellNeighborCounts

    using .SPHCellList
    export NeighborLoop!, ComputeInteractions!, RunSimulation

    using .OpenExternalPrograms
    export AutoOpenLogFile, AutoOpenParaview

    using PrecompileTools
    using LinearAlgebra
    using StaticArrays
    using StructArrays

    @setup_workload begin
        KernelInstance = SPHKernelInstance{2, Float64}(WendlandC2(); dx=0.02)
        CubicKernelInstance = SPHKernelInstance{2, Float64}(CubicSpline{Float64}(); dx=0.02)
        Constants = SimulationConstants(ρ₀ = 1000.0, dx = 0.02, γ = 7.0)
        Position = [SVector(0.0, 0.0), SVector(0.02, 0.0), SVector(0.04, 0.0)]
        Velocity = [SVector(0.0, 0.0), SVector(0.1, 0.0), SVector(-0.05, 0.0)]
        Acceleration = [SVector(0.0, 0.0), SVector(0.0, -9.81), SVector(0.0, -9.81)]
        Density = fill(Constants.ρ₀, 3)
        Pressure = zeros(Float64, 3)
        GravityFactor = [-1.0, 1.0, -1.0]
        MotionLimiter = [1.0, 0.0, 1.0]
        Types = [Fluid, Moving, Fluid]
        GroupMarker = [UInt(1), UInt(1), UInt(2)]
        Cells = fill(CartesianIndex{2}(0, 0), 3)
        Kernel = zeros(Float64, 3)
        KernelGradient = zeros(eltype(Position), 3)
        GhostPoints = zeros(eltype(Position), 3)
        GhostNormals = zeros(eltype(Position), 3)
        SimParticles = StructArray((;
            Cells = Cells,
            Position = Position,
            Acceleration = Acceleration,
            Velocity = Velocity,
            Density = Density,
            Pressure = Pressure,
            GravityFactor = GravityFactor,
            MotionLimiter = MotionLimiter,
            Type = Types,
            GroupMarker = GroupMarker,
            Kernel = Kernel,
            KernelGradient = KernelGradient,
            GhostPoints = GhostPoints,
            GhostNormals = GhostNormals,
        ))
        Xij = Position[1] - Position[2]
        Vij = Velocity[1] - Velocity[2]
        DistanceSquared = dot(Xij, Xij)
        Q = sqrt(DistanceSquared) * KernelInstance.h⁻¹
        GradientWij = ∇Wᵢⱼ(KernelInstance, Q, Xij)
        CubicGradientWij = ∇Wᵢⱼ(CubicKernelInstance, Q, Xij)
        SimMetaData = SimulationMetaData{2, Float64, NoShifting, StoreKernelOutput, SimpleMDBC, NoLog}(SimulationName = "Precompile", SaveLocation = ".")
        DensityRate, VelocityNext, PositionNext, DensityNext, ShiftGradient, ShiftDivergence = AllocateSupportDataStructures(SimMetaData, Position)
        HalfStep = 0.5
        FullStep = 0.01
        MaxVisc = zeros(Float64, 3)
        MinDtForce = fill(Inf, 3)

        @compile_workload begin
            Wᵢⱼ(KernelInstance, Q)
            ∇Wᵢⱼ(KernelInstance, Q, Xij)
            tensile_correction(KernelInstance, 0.0, 1.0, 0.0, 1.0, Q, 0.02)
            Wᵢⱼ(CubicKernelInstance, Q)
            ∇Wᵢⱼ(CubicKernelInstance, Q, Xij)
            tensile_correction(CubicKernelInstance, 0.0, 1.0, 0.0, 1.0, Q, 0.02)
            EquationOfStateGamma7(Constants.ρ₀, Constants.c₀, Constants.ρ₀)
            ConstructGravitySVector(Xij, Constants.g)
            Pressure!(Pressure, Density, Constants)
            LimitDensityAtBoundary!(Density, Constants.ρ₀, MotionLimiter)
            compute_viscosity(ZeroViscosity(), KernelInstance, Constants, SimParticles, Xij, Vij, GradientWij, DistanceSquared, 1, 2)
            compute_viscosity(ArtificialViscosity(), KernelInstance, Constants, SimParticles, Xij, Vij, GradientWij, DistanceSquared, 1, 2)
            compute_viscosity(Laminar(), KernelInstance, Constants, SimParticles, Xij, Vij, GradientWij, DistanceSquared, 1, 2)
            compute_viscosity(LaminarSPS(), KernelInstance, Constants, SimParticles, Xij, Vij, GradientWij, DistanceSquared, 1, 2)
            compute_density_diffusion(ZeroDensityDiffusion(), KernelInstance, Constants, SimParticles, Xij, GradientWij, DistanceSquared, 1, 2, MotionLimiter)
            compute_density_diffusion(LinearDensityDiffusion(), KernelInstance, Constants, SimParticles, Xij, GradientWij, DistanceSquared, 1, 2, MotionLimiter)
            compute_density_diffusion(ComplexDensityDiffusion(), KernelInstance, Constants, SimParticles, Xij, GradientWij, DistanceSquared, 1, 2, MotionLimiter)
            compute_density_diffusion(LinearDensityDiffusion(), CubicKernelInstance, Constants, SimParticles, Xij, CubicGradientWij, DistanceSquared, 1, 2, MotionLimiter)
            HalfTimeStep(SimMetaData, Constants, SimParticles, PositionNext, VelocityNext, DensityNext, DensityRate, HalfStep)
            FullTimeStep(SimMetaData, KernelInstance, Constants, SimParticles, ShiftGradient, ShiftDivergence, FullStep)
            UpdateTimeStepBuffers!(MaxVisc, MinDtForce, 1, Position[1], Velocity[1], Acceleration[1], KernelInstance)
            UpdateTimeStepBuffers!(MaxVisc, MinDtForce, 2, Position[2], Velocity[2], Acceleration[2], KernelInstance)
            FinalizeTimeStep(MaxVisc, MinDtForce, Constants, KernelInstance)
            Δt(Position, Velocity, Acceleration, Constants, KernelInstance)
        end
    end

end
