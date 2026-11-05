# DualSPHysics chrono/12_ExternalVelocity: native GPU motors and rigid response.
using SPHExampleGPU
include(joinpath(@__DIR__,"data","chrono_reference.jl"))
include(joinpath(@__DIR__,"data","external_velocity","output.jl"))

let
    variant = :boxes # :boxes, :breaking_wall, :many_blocks, :hauling, :ship
    configurations = (
        boxes = (;name = "ExternalVelocity3D",D = 3,dx = 0.05,origin = (-0.5,-0.5,-0.5),duration = 2.5,sound = 60.),
        breaking_wall = (;name = "BreakingWall3D",D = 3,dx = 0.02,origin = (-1.,-4.,-1.),duration = 4.,sound = 100.),
        many_blocks = (;name = "BreakingWallMany3D",D = 3,dx = 0.02,origin = (-1.,-4.,-1.),duration = 4.,sound = 100.),
        hauling = (;name = "Hauling2D",D = 2,dx = 0.02,origin = (-1.,0.,-1.),duration = 4.,sound = 160.),
        ship = (;name = "Ship2D",D = 2,dx = 0.005,origin = (-1.,0.,-1.),duration = 10.,sound = 20sqrt(2*9.81*0.3)),
    )
    hasproperty(configurations,variant) || throw(ArgumentError("choose :boxes, :breaking_wall, :many_blocks, :hauling or :ship"))
    configuration = getproperty(configurations,variant)
    Dimensions = configuration.D
    FloatType = Float32
    dx = configuration.dx
    simulation_time = configuration.duration
    output_interval = 0.01
    save_location = joinpath(@__DIR__,"..","output",lowercase(configuration.name))
    visualize = false
    open_log_file = false
    use_graph = true
    iterations = 64
    max_timestep = 0.001
    time_stepping = SymplecticTimeStepping()

    dx == configuration.dx || throw(ArgumentError("regenerate the reference fixture before changing dx"))
    fixture = joinpath(@__DIR__,"data","external_velocity",string(variant))
    weights = variant == :boxes ? Dict(63 => 1.2,64 => 1.2) : variant == :hauling ? Dict(60 => 2.,61 => 0.5) :
        variant == :ship ? Dict(23 => 0.5,24 => 0.5) : Dict(20 => 30.,(g => 1. for g in 61:(variant == :breaking_wall ? 87 : 172))...)
    SimulationGeometry = reference_geometry(fixture,Val(Dimensions),FloatType;dx,origin = configuration.origin,weights)
    velocities = if variant == :boxes
        linear = VelocitySeries([0,.05,.2,.2,1.8,1.8,1.9,1.9,1.91],
            [(0,0,0),(4,0,4),(3,0,3),nothing,(0,0,0),(0,0,8),(0,0,8),(0,0,0),nothing])
        angular = VelocitySeries([0,1.9],[nothing,(0,-10,0)])
        (RigidVelocity(63;Linear = linear,Angular = angular),
            RigidVelocity(64;Linear = reference_velocity_file(joinpath(fixture,"CaseExternalVelocityFile_linvel.csv")),
                Angular = reference_velocity_file(joinpath(fixture,"CaseExternalVelocityFile_angvel.csv"))))
    elseif variant in (:breaking_wall,:many_blocks)
        (RigidVelocity(20;Linear = VelocitySeries([0,.5,.5,.6,.6,2.02,2.02,2.2,2.2],
            [(0,0,1),(0,0,1),(0,0,0),(0,0,0),(-5,0,.1),(-5,0,.1),(0,0,0),(0,0,0),nothing])),)
    elseif variant == :hauling
        # Native counterclockwise rotation is minus world-Y angular velocity.
        (RigidVelocity(60;Angular = VelocitySeries([0,.1],[0.,-5.])),)
    else
        (RigidVelocity(24;Angular = VelocitySeries([0,2.],[0.,4.])),)
    end
    springs = variant == :hauling ? (BodySpring(61,60;Attachment = (0.6,0.4),Anchor = (1.2,0.4),
        Stiffness = 4000,Damping = 0,RestLength = 1.),) : ()
    joints = variant == :ship ? (HingeJoint(23,24;Anchor = (1.755,0,0.145),Axis = (0,1,0)),) : ()
    contacts = variant == :ship ? nothing : RigidContacts(ParticleDiameter = dx,Margin = 0.25dx,
        Friction = 0.75,FixedFriction = 0.45,Restitution = 0.5,FixedRestitution = 0.8,
        Materials = variant in (:breaking_wall,:many_blocks) ? (20 => RigidMaterial(Friction = 0.45,Restitution = 0.8),) : ())
    SimRigidBodyDynamics = RigidBodyDynamics(Joints = joints,Contacts = contacts,Velocities = velocities,Springs = springs,
        Iterations = iterations,MaxTimeStep = variant == :ship ? nothing : max_timestep)
    SimConstants = SimulationConstants{FloatType}(Dimensions;dx,g = 9.81,c₀ = configuration.sound,α = variant == :ship ? 0.05 : 0.01,CFL = 0.2)
    smoothing_length = sqrt(Dimensions)*dx
    SimKernel = SPHKernelInstance{Dimensions,FloatType}(WendlandC2();h = FloatType(smoothing_length),η² = FloatType((0.1smoothing_length)^2))
    SimViscosity = ArtificialViscosity(boundary_factor = 1)
    SimDensityDiffusion = variant in (:hauling,:ship) ? FourtakasDensityDiffusion() : ZeroDensityDiffusion()
    SimMotion = nothing
    SimMeasurements = nothing
    SimMetaData = SimulationMetaData{Dimensions,FloatType,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = configuration.name,SaveLocation = save_location,SimulationTime = simulation_time,OutputTimes = output_interval,
        VisualizeInParaview = visualize,OpenLogFile = open_log_file,ExportSingleVTKHDF = true,GPUDoublePosition = true,
        GPUBoundaryForces = false,GPUUseGraph = use_graph,GPUMaxStepsPerSync = 64,GPUClipTimeStepToOutput = true)
    save_surfaces(directory,geometry;write_geometry = true) = save_external_velocity_surfaces(directory,fixture,geometry,configuration.name,variant;write_geometry)

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
