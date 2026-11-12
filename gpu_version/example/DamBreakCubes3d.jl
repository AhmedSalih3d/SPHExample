# DualSPHysics chrono/07_DamBreakCubes: one cube, three cubes or eighteen blocks.
using SPHExampleGPU
include(joinpath(@__DIR__,"data","chrono_reference.jl"))

let
    variant = :one_cube # :one_cube, :three_cubes, :blocks
    configurations = (
        one_cube = (;fixture = "one_cube",name = "DamBreak1Cube3D",dx = 0.0125,origin = (-2.05,-0.05,-0.05),duration = 2.,B = 217221.42857),
        three_cubes = (;fixture = "three_cubes",name = "DamBreak3Cubes3D",dx = 0.0125,origin = (-2.05,-0.05,-0.05),duration = 2.,B = 266271.42857),
        blocks = (;fixture = "blocks",name = "DamBreakBlocks3D",dx = 0.01,origin = (-0.05,-0.05,-0.05),duration = 1.5,B = 560571.42857),
    )
    hasproperty(configurations,variant) || throw(ArgumentError("choose :one_cube, :three_cubes or :blocks"))
    configuration = getproperty(configurations,variant)
    Dimensions = 3
    FloatType = Float32
    dx = configuration.dx
    simulation_time = configuration.duration
    output_interval = 0.01
    save_location = joinpath(@__DIR__,"..","output",lowercase(configuration.name))
    visualize = false
    open_log_file = false
    use_graph = true
    iterations = 32
    time_stepping = SymplecticTimeStepping()

    dx == configuration.dx || throw(ArgumentError("regenerate the GenCase fixture before changing dx"))
    fixture = joinpath(@__DIR__,"data","dam_break_cubes",configuration.fixture)
    groups = variant == :blocks ? (61:78) : variant == :three_cubes ? (61:63) : (61:61)
    weight = variant == :blocks ? 1.25 : 2.7/(2197*1000dx^3)
    details = variant == :blocks ? Dict(g => FloatingDetails{FloatType}(RelativeWeight = weight,
        InteractionMass = 1.25*1000dx^3) for g in groups) : Dict()
    SimulationGeometry = reference_geometry(fixture,Val(Dimensions),FloatType;dx,origin = configuration.origin,
        weights = Dict(g => weight for g in groups),details)
    # The reference uses Chrono CAD contacts. Native contacts use the exact
    # sampled body/tank geometry; this is a different contact discretization.
    SimRigidBodyDynamics = RigidBodyDynamics(Contacts = RigidContacts(ParticleDiameter = dx,Margin = 0.5dx,
        Friction = variant == :blocks ? 0.7 : 0.15,FixedFriction = 0.35,
        Restitution = 0.6,FixedRestitution = variant == :blocks ? 0.65 : 0.8,FixedGroups = (10,)),Iterations = iterations)
    SimConstants = SimulationConstants{FloatType}(Dimensions;dx,g = 9.81,c₀ = sqrt(7configuration.B/1000),α = 0.05,CFL = 0.2)
    SimKernel = SPHKernelInstance{Dimensions,FloatType}(WendlandC2();h = FloatType(sqrt(3)*dx),η² = FloatType((0.1sqrt(3)*dx)^2))
    SimViscosity = ArtificialViscosity(boundary_factor = variant == :blocks ? 1 : 0)
    SimDensityDiffusion = FourtakasDensityDiffusion()
    SimMotion = nothing
    SimMeasurements = nothing
    SimMetaData = SimulationMetaData{Dimensions,FloatType,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = configuration.name,SaveLocation = save_location,SimulationTime = simulation_time,OutputTimes = output_interval,
        VisualizeInParaview = visualize,OpenLogFile = open_log_file,ExportSingleVTKHDF = true,GPUDoublePosition = true,
        GPUBoundaryForces = false,GPUUseGraph = use_graph,GPUMaxStepsPerSync = 64,GPUClipTimeStepToOutput = true)
    save_surfaces(directory,geometry;write_geometry = true) = save_reference_surfaces(directory,fixture,geometry,configuration.name;
        force_offset = variant == :blocks ? 0.08 : 0.2,force_offset_vector = (0,0,variant == :blocks ? 0.25 : 0.4),write_geometry)

    mkpath(SimMetaData.SaveLocation)
    SimLogger = SimulationLogger(SimMetaData.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry,SimMetaData)
    try
        RunSimulation(SimGeometry = SimulationGeometry,SimMetaData = SimMetaData,SimConstants = SimConstants,
            SimParticles = SimParticles,SimKernel = SimKernel,SimLogger = SimLogger,SimViscosity = SimViscosity,
            SimDensityDiffusion = SimDensityDiffusion,SimTimeStepping = time_stepping,
            SimRigidBodyDynamics = SimRigidBodyDynamics,SimMotion = SimMotion,SimMeasurements = SimMeasurements)
        save_surfaces(SimMetaData.SaveLocation,SimulationGeometry)
    finally
        isopen(SimLogger.LoggerIo) && close(SimLogger.LoggerIo)
    end
end
