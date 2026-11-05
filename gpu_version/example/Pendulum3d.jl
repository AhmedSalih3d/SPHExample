# DualSPHysics 5.4 examples/chrono/01_Pendulum/CasePendulum: four rigid arms
# with spherical joints, solved entirely on CUDA. No Chrono is required.
# Including or running this file starts the simulation.
#
#     julia --project=gpu_version -t 2,0 gpu_version/example/Pendulum3d.jl
using SPHExampleGPU
using StaticArrays: SVector
include(joinpath(@__DIR__, "data", "pendulum", "output.jl"))

let
    Dimensions = 3
    FloatType  = Float64

    dx              = 0.05
    simulation_time = 6.0
    output_interval = 0.01
    save_location   = joinpath(@__DIR__, "..", "output", "pendulum_3d")
    visualize       = true
    open_log_file   = true
    use_graph       = true
    iterations      = 48
    max_timestep    = 0.001
    time_stepping   = SymplecticTimeStepping()

    dx == 0.05 || throw(ArgumentError("this reference lattice uses dx = 0.05 m; regenerate the cylinder sampling before changing resolution"))
    isfinite(simulation_time) && simulation_time > 0 && isfinite(output_interval) && output_interval > 0 ||
        throw(ArgumentError("simulation_time and output_interval must be finite and positive"))

    SimConstants = SimulationConstants{FloatType}(Dimensions;
        dx = dx, ρ₀ = 1000, g = 9.81, c₀ = 100, α = 0.01, CFL = 0.2)

    # GenCase clips the box to pointmin=(4,0.5,-0.1), pointmax=(6,9.5,7.5).
    # Only its bottom is sampled; the upper x/y bounds are exclusive.
    floor = [SVector(4 + i*dx, 0.5 + j*dx, 0.0) for j in 0:179 for i in 0:39]
    Domain = SPHGeometry{Dimensions,FloatType}(floor;
        Density = SimConstants.ρ₀, GroupMarker = 6, Type = Fixed)

    # The radius=dp, mask=1 reference cylinder has an eight-point shell and
    # a top cap. A later arm owns the shared end ring, while the cap centre
    # stays with the preceding arm: 161,161,161,169 particles.
    ring = [(i,k) for k in -1:1 for i in -1:1 if i != 0 || k != 0]
    arm_positions = [[SVector(5 + i*dx, 5 + (arm-1) + j*dx, 5 + k*dx)
        for j in 0:(arm == 4 ? 20 : 19) for (i,k) in ring] for arm in 1:4]
    for arm in 1:4
        push!(arm_positions[arm], SVector(5., 5. + arm, 5.))
    end
    arms = [SPHGeometry{Dimensions,FloatType}(points;
        Density = SimConstants.ρ₀, GroupMarker = arm, Type = Floating,
        Floating = FloatingDetails{FloatType}(RelativeWeight = 1.0)) for (arm,points) in enumerate(arm_positions)]
    Arm1, Arm2, Arm3, Arm4 = arms
    SimulationGeometry = [Domain, Arm1, Arm2, Arm3, Arm4]

    # The fixed pivot is an analytical constraint, not a particle support.
    # All four reference links are spherical, with zero stiffness/damping.
    joints = (
        BallJoint(Arm1.GroupMarker; Anchor = (5,5,5)),
        BallJoint(Arm1.GroupMarker, Arm2.GroupMarker; Anchor = (5,6,5)),
        BallJoint(Arm2.GroupMarker, Arm3.GroupMarker; Anchor = (5,7,5)),
        BallJoint(Arm3.GroupMarker, Arm4.GroupMarker; Anchor = (5,8,5)),
    )
    # RigidAlgorithm=0: no contacts, fluid, periodicity or joint damping.
    SimRigidBodyDynamics = RigidBodyDynamics(Joints = joints, Contacts = nothing,
        Iterations = iterations, MaxTimeStep = max_timestep)

    SimMetaData = SimulationMetaData{
        Dimensions, FloatType, NoShifting, NoKernelOutput, NoMDBC, StoreLog,
    }(
        SimulationName = "Pendulum3D", SaveLocation = save_location,
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
            SimTimeStepping = time_stepping,
            SimRigidBodyDynamics = SimRigidBodyDynamics,
        )
        save_pendulum_surfaces(SimMetaData.SaveLocation, SimulationGeometry)
        SimMetaData.VisualizeInParaview = visualize
        open_pendulum_output(SimMetaData)
    finally
        isopen(SimLogger.LoggerIo) && close(SimLogger.LoggerIo)
    end
end
