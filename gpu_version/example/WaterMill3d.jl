# DualSPHysics chrono/08_WaterMill: five fluid-driven bodies and damped hinges.
using SPHExampleGPU
include(joinpath(@__DIR__,"data","chrono_reference.jl"))

let
    Dimensions = 3
    FloatType = Float32
    dx = 0.025
    simulation_time = 5.0
    output_interval = 0.01
    save_location = joinpath(@__DIR__,"..","output","water_mill_3d")
    visualize = false
    open_log_file = false
    use_graph = true
    iterations = 32
    time_stepping = SymplecticTimeStepping()

    dx == 0.025 || throw(ArgumentError("the reference WaterMill fixture requires dx = 0.025"))
    fixture = joinpath(@__DIR__,"data","water_mill")
    SimulationGeometry = reference_geometry(fixture,Val(Dimensions),FloatType;dx,origin = (0.,0.,-0.3),
        weights = Dict(g => 0.04 for g in 11:15))
    joints = (
        HingeJoint(11;Anchor = (5.5,0,3.),Axis = (0,1,0),Damping = 500),
        HingeJoint(11,12;Anchor = (5.5,0,1.1),Axis = (0,1,0),Damping = 100),
        HingeJoint(11,13;Anchor = (5.5,0,4.8),Axis = (0,1,0),Damping = 100),
        HingeJoint(11,14;Anchor = (7.35,0,3.),Axis = (0,1,0),Damping = 100),
        HingeJoint(11,15;Anchor = (3.65,0,3.),Axis = (0,1,0),Damping = 100),
    )
    SimRigidBodyDynamics = RigidBodyDynamics(Joints = joints,Iterations = iterations)
    SimConstants = SimulationConstants{FloatType}(Dimensions;dx,g = 9.81,c₀ = sqrt(7*3363821.7332/1000),α = 0.01,CFL = 0.2)
    SimKernel = SPHKernelInstance{Dimensions,FloatType}(WendlandC2();h = FloatType(1.5sqrt(3)*dx),η² = FloatType((0.1*1.5sqrt(3)*dx)^2))
    SimViscosity = ArtificialViscosity(boundary_factor = 1)
    SimDensityDiffusion = FourtakasDensityDiffusion()
    SimMotion = nothing
    SimMeasurements = nothing
    SimMetaData = SimulationMetaData{Dimensions,FloatType,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = "WaterMill3D",SaveLocation = save_location,SimulationTime = simulation_time,OutputTimes = output_interval,
        VisualizeInParaview = visualize,OpenLogFile = open_log_file,ExportSingleVTKHDF = true,GPUDoublePosition = true,
        GPUBoundaryForces = false,GPUUseGraph = use_graph,GPUMaxStepsPerSync = 64,GPUClipTimeStepToOutput = true)
    save_surfaces(directory,geometry;write_geometry = true) = save_reference_surfaces(directory,fixture,geometry,"WaterMill3D";
        force_offset = 0.6,force_offset_vector = (0,0,1.),speed_limit = 8.,write_geometry)

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
