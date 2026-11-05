# DualSPHysics chrono/11_CurrentWheelPulley: genuine open flow and native shafts.
using SPHExampleGPU
include(joinpath(@__DIR__,"data","chrono_reference.jl"))
include(joinpath(@__DIR__,"data","current_wheel","output.jl"))

let
    variant = :wheel # :wheel or :pulley
    variant in (:wheel,:pulley) || throw(ArgumentError("choose :wheel or :pulley"))
    name = variant == :wheel ? "CurrentWheel2D" : "CurrentWheelPulley2D"
    Dimensions = 2
    FloatType = Float32
    dx = 0.0075
    simulation_time = 4.0
    output_interval = 0.01
    save_location = joinpath(@__DIR__,"..","output",lowercase(name))
    visualize = false
    open_log_file = false
    use_graph = true
    cache_interaction_fields = false
    iterations = 32
    time_stepping = SymplecticTimeStepping()

    dx == 0.0075 || throw(ArgumentError("the current-wheel fixture requires dx = 0.0075"))
    fixture = joinpath(@__DIR__,"data","current_wheel",string(variant))
    weights = variant == :wheel ? Dict(60 => 1.5) : Dict(60 => 0.5,61 => 0.5)
    reference_geometry_data = reference_geometry(fixture,Val(Dimensions),FloatType;dx,origin = (-0.1,0.,-0.5),weights)
    # GenCase snaps the inlet seed line to x = -0.0025 m, while the outlet is x = 2 m.
    SimOpenBoundaries = OpenChannel{FloatType}(Inlet = -0.0025,Outlet = 2.,WaterLevel = 0.3,
        Velocity = 1.,Spacing = dx,Duration = simulation_time,InletLayers = 5,OutletLayers = 8)
    SimulationGeometry = open_channel_geometry(reference_geometry_data,SimOpenBoundaries)
    joints = variant == :wheel ? (HingeJoint(60;Anchor = (1,0,0.6),Axis = (0,1,0)),) : (
        HingeJoint(60;Anchor = (1,0,0.6),Axis = (0,1,0)),
        HingeJoint(61;Anchor = (1.0025,0,1.2025),Axis = (0,1,0)),
        PulleyJoint(60,61;Ratio = 0.3/0.05),
    )
    SimRigidBodyDynamics = RigidBodyDynamics(Joints = joints,Iterations = iterations)
    SimConstants = SimulationConstants{FloatType}(Dimensions;dx,g = 9.81,c₀ = sqrt(7*560571.42857/1000),α = 0.05,CFL = 0.2)
    SimKernel = SPHKernelInstance{Dimensions,FloatType}(WendlandC2();h = FloatType(sqrt(2)*dx),η² = FloatType((0.1sqrt(2)*dx)^2))
    SimViscosity = ArtificialViscosity(boundary_factor = 0)
    SimDensityDiffusion = FourtakasDensityDiffusion()
    SimMotion = nothing
    SimMeasurements = nothing
    SimMetaData = SimulationMetaData{Dimensions,FloatType,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = name,SaveLocation = save_location,SimulationTime = simulation_time,OutputTimes = output_interval,
        VisualizeInParaview = visualize,OpenLogFile = open_log_file,ExportSingleVTKHDF = true,GPUDoublePosition = true,
        GPUBoundaryForces = false,GPUCacheInteractionFields = cache_interaction_fields,GPUUseGraph = use_graph,GPUMaxStepsPerSync = 64,GPUClipTimeStepToOutput = true)
    save_surfaces(directory,geometry;write_geometry = true) = save_current_wheel_surfaces(directory,fixture,geometry,name,variant;write_geometry,dx)

    mkpath(SimMetaData.SaveLocation)
    SimLogger = SimulationLogger(SimMetaData.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry,SimMetaData)
    try
        RunSimulation(SimGeometry = SimulationGeometry,SimMetaData = SimMetaData,SimConstants = SimConstants,
            SimParticles = SimParticles,SimKernel = SimKernel,SimLogger = SimLogger,SimViscosity = SimViscosity,
            SimDensityDiffusion = SimDensityDiffusion,SimTimeStepping = time_stepping,
            SimRigidBodyDynamics = SimRigidBodyDynamics,SimMotion,SimMeasurements,SimOpenBoundaries)
        save_surfaces(SimMetaData.SaveLocation,SimulationGeometry)
    finally
        isopen(SimLogger.LoggerIo) && close(SimLogger.LoggerIo)
    end
end
