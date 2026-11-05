# DualSPHysics chrono/05_OWSC: wave-driven flap with an elastic damped hinge.
using SPHExampleGPU
include(joinpath(@__DIR__,"data","chrono_reference.jl"))
include(joinpath(@__DIR__,"data","regular_piston.jl"))

let
    Dimensions = 3
    FloatType = Float32
    dx = 0.05
    simulation_time = 10.0
    output_interval = 0.02
    save_location = joinpath(@__DIR__,"..","output","owsc_3d")
    visualize = false
    open_log_file = false
    use_graph = true
    iterations = 32
    time_stepping = SymplecticTimeStepping()

    dx == 0.05 || throw(ArgumentError("the bundled OWSC fixture requires dx = 0.05"))
    fixture = joinpath(@__DIR__,"data","owsc")
    flap = FloatingDetails{FloatType}(RelativeWeight = 0.461,Center = (0.,0.,0.33),Inertia = (27.6,14.8,13.1))
    SimulationGeometry = reference_geometry(fixture,Val(Dimensions),FloatType;dx,origin = (-10.,-1.,-0.2),
        weights = Dict(15 => 0.461),details = Dict(15 => flap))
    SimRigidBodyDynamics = RigidBodyDynamics(Joints = (
        HingeJoint(15;Anchor = (0,0,0),Axis = (0,1,0),Stiffness = 15,Damping = 100),),Iterations = iterations)
    SimConstants = SimulationConstants{FloatType}(Dimensions;dx,g = 9.81,c₀ = sqrt(7*378385.71429/1000),α = 0.01,CFL = 0.2)
    SimKernel = SPHKernelInstance{Dimensions,FloatType}(WendlandC2();h = FloatType(1.2sqrt(3)*dx),η² = FloatType((0.1*1.2sqrt(3)*dx)^2))
    SimViscosity = ArtificialViscosity(boundary_factor = 0)
    SimDensityDiffusion = FourtakasDensityDiffusion()
    SimMotion,piston_state = regular_piston_motion(20,Val(Dimensions);height = 0.15,period = 2.,depth = 1.3,
        duration = simulation_time,ramp_periods = 1)
    SimMeasurements = nothing
    SimMetaData = SimulationMetaData{Dimensions,FloatType,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = "OWSC3D",SaveLocation = save_location,SimulationTime = simulation_time,OutputTimes = output_interval,
        VisualizeInParaview = visualize,OpenLogFile = open_log_file,ExportSingleVTKHDF = true,GPUDoublePosition = true,
        GPUBoundaryForces = false,GPUUseGraph = use_graph,GPUMaxStepsPerSync = 64,GPUClipTimeStepToOutput = true)
    save_surfaces(directory,geometry;write_geometry = true) = save_reference_surfaces(directory,fixture,geometry,"OWSC3D";
        moving = Dict(20 => (point,time) -> point+SVector(piston_state(time)[1],0,0)),force_offset = 0.5,
        force_offset_vector = (0,0,1.6),speed_limit = 0.5,write_geometry)

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
