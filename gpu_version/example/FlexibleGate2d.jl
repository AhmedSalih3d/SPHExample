# DualSPHysics 5.4 chrono/03_FlexibleGate, native GPU elastic hinges.
using SPHExampleGPU
include(joinpath(@__DIR__,"data","chrono_reference.jl"))

let
    Dimensions = 2
    FloatType = Float32
    dx = 0.01
    simulation_time = 2.0
    output_interval = 0.01
    save_location = joinpath(@__DIR__,"..","output","flexible_gate_2d")
    visualize = false
    open_log_file = false
    use_graph = true
    iterations = 32
    time_stepping = SymplecticTimeStepping()

    dx == 0.01 || throw(ArgumentError("the bundled GenCase fixture requires dx = 0.01"))
    fixture = joinpath(@__DIR__,"data","flexible_gate")
    SimulationGeometry = reference_geometry(fixture,Val(Dimensions),FloatType;
        dx,origin = (-0.1,0.,-0.1),weights = Dict(g => 0.8 for g in 11:15))
    # Five links extend from the free lower tip to the fixed upper hinge.
    joints = Tuple(HingeJoint(10+b,b == 5 ? 0 : 11+b;Anchor = (0.5,0,0.2b),Axis = (0,1,0),
        Stiffness = 750,Damping = 50) for b in 1:5)
    SimRigidBodyDynamics = RigidBodyDynamics(Joints = joints,Iterations = iterations)
    SimConstants = SimulationConstants{FloatType}(Dimensions;dx,g = 9.81,c₀ = sqrt(7*330737.14286/1000),α = 0.02,CFL = 0.2)
    SimKernel = SPHKernelInstance{Dimensions,FloatType}(WendlandC2();h = FloatType(sqrt(2)*dx),η² = FloatType((0.1sqrt(2)*dx)^2))
    SimViscosity = ArtificialViscosity(boundary_factor = 0)
    SimDensityDiffusion = FourtakasDensityDiffusion()
    SimMotion = nothing
    SimMeasurements = nothing
    SimMetaData = SimulationMetaData{Dimensions,FloatType,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = "FlexibleGate2D",SaveLocation = save_location,SimulationTime = simulation_time,
        OutputTimes = output_interval,VisualizeInParaview = visualize,OpenLogFile = open_log_file,
        ExportSingleVTKHDF = true,GPUDoublePosition = true,GPUBoundaryForces = false,
        GPUUseGraph = use_graph,GPUMaxStepsPerSync = 64,GPUClipTimeStepToOutput = true)
    save_surfaces(directory,geometry) = save_reference_surfaces(directory,fixture,geometry,"FlexibleGate2D";
        force_offset = 0.16,force_offset_vector = (0.18,0,0),fluid_alpha = 2dx)

    mkpath(SimMetaData.SaveLocation)
    SimLogger = SimulationLogger(SimMetaData.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry,SimMetaData)
    try
        RunSimulation(SimGeometry = SimulationGeometry,SimMetaData = SimMetaData,SimConstants = SimConstants,
            SimParticles = SimParticles,SimKernel = SimKernel,SimLogger = SimLogger,SimViscosity = SimViscosity,
            SimDensityDiffusion = SimDensityDiffusion,SimTimeStepping = time_stepping,
            SimRigidBodyDynamics = SimRigidBodyDynamics,SimMotion,SimMeasurements)
        save_surfaces(SimMetaData.SaveLocation,SimulationGeometry)
    finally
        isopen(SimLogger.LoggerIo) && close(SimLogger.LoggerIo)
    end
end
