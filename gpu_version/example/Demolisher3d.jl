# DualSPHysics 5.4 examples/chrono/01_Pendulum/CaseDemolisher.
# A four-arm chain carries a wrecking ball into sixteen movable blocks.
# Including or running this file starts the native CUDA simulation.
#
#     julia --project=gpu_version -t 2,0 gpu_version/example/Demolisher3d.jl
using SPHExampleGPU
using StaticArrays: SVector
include(joinpath(@__DIR__, "data", "demolisher", "output.jl"))

let
    Dimensions = 3
    FloatType  = Float32

    dx              = 0.05
    simulation_time = 3.0
    output_interval = 0.01
    save_location   = joinpath(@__DIR__, "..", "output", "demolisher_3d")
    visualize       = true
    open_log_file   = true
    use_graph       = true
    iterations      = 32
    max_timestep    = 0.001
    time_stepping   = SymplecticTimeStepping()

    dx == 0.05 || throw(ArgumentError("the reference GenCase sampling uses dx = 0.05 m; regenerate it before changing resolution"))
    isfinite(simulation_time) && simulation_time > 0 && isfinite(output_interval) && output_interval > 0 ||
        throw(ArgumentError("simulation_time and output_interval must be finite and positive"))

    SimConstants = SimulationConstants{FloatType}(Dimensions;
        dx = dx, ρ₀ = 1000, g = 9.81, c₀ = 100, α = 0.01, CFL = 0.2)

    # The open container occupies [0,11] × [2,8] × [0,3]. Shared edges
    # occur once, giving 67,541 fixed nodes on the five reference faces.
    domain_points = [SVector(i*dx, 2+j*dx, 0.) for j in 0:120 for i in 0:220]
    append!(domain_points, [SVector(i*dx, y, k*dx) for k in 1:60 for y in (2.,8.) for i in 0:220])
    append!(domain_points, [SVector(x, 2+j*dx, k*dx) for k in 1:60 for x in (0.,11.) for j in 1:119])
    Domain = SPHGeometry{Dimensions,FloatType}(domain_points;
        Density = SimConstants.ρ₀, GroupMarker = 11, Type = Fixed)

    # GenCase's mask=1 cylinders sample an eight-node ring at this spacing.
    # Later shapes own shared rings; the sphere replaces the last cap.
    ring = [(j,k) for k in -1:1 for j in -1:1 if j != 0 || k != 0]
    arm_points = [[SVector(4+arm+i*dx, 5+j*dx, 5.2+k*dx)
        for i in 0:19 for (j,k) in ring] for arm in 1:4]
    for arm in 1:3
        push!(arm_points[arm], SVector(5.0 + arm, 5.0, 5.2))
    end
    arms = [SPHGeometry{Dimensions,FloatType}(points;
        Density = SimConstants.ρ₀, GroupMarker = arm, Type = Floating,
        Floating = FloatingDetails{FloatType}(RelativeWeight = 1.2)) for (arm,points) in enumerate(arm_points)]
    Arm1, Arm2, Arm3, Arm4 = arms

    # This integer rule reproduces the reference's 1,839 sphere samples.
    ball_points = [SVector(9.35+i*dx, 5+j*dx, 5.2+k*dx)
        for k in -7:7 for j in -7:7 for i in -7:7 if i*i+j*j+k*k <= 57]
    Ball = SPHGeometry{Dimensions,FloatType}(ball_points;
        Density = SimConstants.ρ₀, GroupMarker = 5, Type = Floating,
        Floating = FloatingDetails{FloatType}(RelativeWeight = 2.5))

    block_origins = vcat([SVector(4.7,3.7+0.55*j,0.05) for j in 0:4],
        [SVector(4.7,3.45+0.55*j,0.6) for j in 0:5],
        [SVector(4.7,3.7+0.55*j,1.15) for j in 0:4])
    blocks = [SPHGeometry{Dimensions,FloatType}(
        [origin + dx*SVector(i,j,k) for k in 0:10 for j in 0:10 for i in 0:10];
        Density = SimConstants.ρ₀, GroupMarker = 51+block, Type = Floating,
        Floating = FloatingDetails{FloatType}(RelativeWeight = 1.2)) for (block,origin) in enumerate(block_origins)]
    SimulationGeometry = [Domain, arms..., Ball, blocks...]

    joints = (
        BallJoint(Arm1.GroupMarker; Anchor = (5,5,5.2)),
        BallJoint(Arm1.GroupMarker, Arm2.GroupMarker; Anchor = (6,5,5.2)),
        BallJoint(Arm2.GroupMarker, Arm3.GroupMarker; Anchor = (7,5,5.2)),
        BallJoint(Arm3.GroupMarker, Arm4.GroupMarker; Anchor = (8,5,5.2)),
        BallJoint(Arm4.GroupMarker, Ball.GroupMarker; Anchor = (9,5,5.2)),
    )
    steel = RigidMaterial(Friction = 0.45, Restitution = 0.8)
    wood  = RigidMaterial(Friction = 0.75, Restitution = 0.5)
    # The reference gives collision meshes to the ball, blocks and domain,
    # but not to the arms. Keep their analytical joints active. Particle
    # contacts use a full-dp distance so sampled walls remain closed.
    contacts = RigidContacts(ParticleDiameter = dx, Margin = dx/2, FixedGroups = (Domain.GroupMarker,),
        FloatingGroups = (Ball.GroupMarker, (b.GroupMarker for b in blocks)...),
        Friction = wood.friction, Restitution = wood.restitution,
        FixedFriction = steel.friction, FixedRestitution = steel.restitution,
        Materials = (Ball.GroupMarker => steel,))
    SimRigidBodyDynamics = RigidBodyDynamics(Joints = joints, Contacts = contacts,
        Iterations = iterations, MaxTimeStep = max_timestep)

    SimMetaData = SimulationMetaData{
        Dimensions, FloatType, NoShifting, NoKernelOutput, NoMDBC, StoreLog,
    }(
        SimulationName = "Demolisher3D", SaveLocation = save_location,
        SimulationTime = simulation_time, OutputTimes = output_interval,
        VisualizeInParaview = false, OpenLogFile = open_log_file,
        ExportSingleVTKHDF = true, GPUDoublePosition = true,
        GPUBoundaryForces = false, GPUUseGraph = use_graph,
        GPUMaxStepsPerSync = 64, GPUCellSubdivision = 1,
        GPUClipTimeStepToOutput = true,
    )
    SimKernel           = SPHKernelInstance{Dimensions,FloatType}(WendlandC2(); h = FloatType(sqrt(3)*dx))
    SimViscosity        = ArtificialViscosity()
    SimDensityDiffusion = ZeroDensityDiffusion()

    mkpath(SimMetaData.SaveLocation)
    SimLogger    = SimulationLogger(SimMetaData.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaData)

    try
        RunSimulation(
            SimGeometry = SimulationGeometry, SimMetaData = SimMetaData,
            SimConstants = SimConstants, SimLogger = SimLogger,
            SimParticles = SimParticles, SimKernel = SimKernel,
            SimViscosity = SimViscosity, SimDensityDiffusion = SimDensityDiffusion,
            SimTimeStepping = time_stepping, SimRigidBodyDynamics = SimRigidBodyDynamics,
        )
        save_demolisher_surfaces(SimMetaData.SaveLocation, SimulationGeometry)
        SimMetaData.VisualizeInParaview = visualize
        open_demolisher_output(SimMetaData)
    finally
        isopen(SimLogger.LoggerIo) && close(SimLogger.LoggerIo)
    end
end
