# DualSPHysics chrono/10_PointAbsorber: direct/native heave, spring and Coulomb PTO.
using SPHExampleGPU
include(joinpath(@__DIR__,"data","chrono_reference.jl"))
include(joinpath(@__DIR__,"data","regular_piston.jl"))

let
    variant = :direct # :direct, :point_line, :spring, :coulomb
    variant in (:direct,:point_line,:spring,:coulomb) || throw(ArgumentError("unknown point absorber variant"))
    names = (;direct = "PointAbsorber2D",point_line = "PointAbsorberLine2D",spring = "PointAbsorberSpring2D",coulomb = "PointAbsorberCoulomb2D")
    name = getproperty(names,variant)
    Dimensions = 2
    FloatType = Float32
    dx = 0.01
    simulation_time = 10.0
    output_interval = 0.05
    save_location = joinpath(@__DIR__,"..","output",lowercase(name))
    visualize = false
    open_log_file = false
    use_graph = true
    iterations = 32
    time_stepping = SymplecticTimeStepping()

    dx == 0.01 || throw(ArgumentError("the bundled point absorber fixture requires dx = 0.01"))
    fixture = joinpath(@__DIR__,"data","point_absorber")
    constraints = variant == :point_line ? nothing : FloatingConstraints{Dimensions,FloatType}(
        TranslationLocks = (true,false),RotationLocks = (true,))
    links = if variant == :spring
        (LinearSpring{Dimensions,FloatType}(Anchor = (2,0),Attachment = (2,0.325),Stiffness = 100,Damping = 500,RestLength = 0.325),)
    elseif variant == :coulomb
        (CoulombDamper{Dimensions,FloatType}(Anchor = (2,0),Attachment = (2,0.325),Force = 10,RestLength = 0.325),)
    else
        ()
    end
    body = FloatingDetails{FloatType}(RelativeWeight = 0.5,Constraints = constraints,Springs = links)
    SimulationGeometry = reference_geometry(fixture,Val(Dimensions),FloatType;dx,origin = (-0.2,0.,-0.05),
        weights = Dict(60 => 0.5),details = Dict(60 => body))
    # The reference point-line permits only world-Z rotation, which is outside
    # this physical X-Z plane. Its planar motion is therefore pure heave.
    SimRigidBodyDynamics = variant == :point_line ? RigidBodyDynamics(Joints = (
        PointLineJoint(60;Point = (2,0,0.4),Direction = (0,0,1),LockRotation = true),),Iterations = iterations) : nothing
    SimConstants = SimulationConstants{FloatType}(Dimensions;dx,g = 9.81,c₀ = sqrt(7*213017.14286/1000),α = 0.01,CFL = 0.2)
    SimKernel = SPHKernelInstance{Dimensions,FloatType}(WendlandC2();h = FloatType(1.2sqrt(2)*dx),η² = FloatType((0.1*1.2sqrt(2)*dx)^2))
    SimViscosity = ArtificialViscosity(boundary_factor = 0)
    SimDensityDiffusion = FourtakasDensityDiffusion()
    SimMotion,piston_state = regular_piston_motion(20,Val(Dimensions);height = 0.1,period = 1.2,depth = 0.4,
        duration = simulation_time,ramp_periods = 1)
    SimMeasurements = MeasurementConfig(water_column_probes = [WaterColumnProbe("WG1",(1.,0.);radius = 2dx)])
    SimMetaData = SimulationMetaData{Dimensions,FloatType,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = name,SaveLocation = save_location,SimulationTime = simulation_time,OutputTimes = output_interval,
        VisualizeInParaview = visualize,OpenLogFile = open_log_file,ExportSingleVTKHDF = true,GPUDoublePosition = true,
        GPUBoundaryForces = false,GPUUseGraph = use_graph,GPUMaxStepsPerSync = 64,GPUClipTimeStepToOutput = true)
    save_surfaces(directory,geometry;write_geometry = true) = save_reference_surfaces(directory,fixture,geometry,name;
        moving = Dict(20 => (point,time) -> point+SVector(piston_state(time)[1],0,0)),force_offset = 0.3,
        fluid_alpha = 2dx,speed_limit = 1.,detail_view = true,
        direct_heave_weight = variant == :point_line ? nothing : Float64(FloatType(0.5)*SimConstants.m₀*FloatType(527)*SimConstants.g),write_geometry)

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
