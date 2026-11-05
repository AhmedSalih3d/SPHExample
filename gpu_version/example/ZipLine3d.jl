# DualSPHysics 5.4 examples/chrono/06_ZipLine, using native CUDA joints and
# particle contacts. No Chrono library or installation is needed to run.
# Including or running this file starts the simulation. Requires an NVIDIA GPU.
#
#     julia --project=gpu_version -t 2,0 gpu_version/example/ZipLine3d.jl
using SPHExampleGPU
using StaticArrays: SVector
using CSV
include(joinpath(@__DIR__, "data", "zipline", "output.jl"))

let
    Dimensions = 3
    FloatType  = Float32

    dx              = 0.04
    simulation_time = 3.5
    output_interval = 0.02
    save_location   = joinpath(@__DIR__, "..", "output", "zipline_3d")
    visualize       = true
    open_log_file   = true
    use_graph       = true
    iterations      = 32
    max_timestep    = 0.002

    dx == 0.04 || throw(ArgumentError("the bundled GenCase lattice uses dx = 0.04 m; regenerate it before changing resolution"))
    isfinite(simulation_time) && simulation_time > 0 && isfinite(output_interval) && output_interval > 0 ||
        throw(ArgumentError("simulation_time and output_interval must be finite and positive"))

    # GenCase's 40 mm lattice is anchored at the definition's pointmin. The
    # cable and CAD ragdoll use bundled integer lattice coordinates. Only the
    # two sampled container faces lie inside this example's particle domain.
    # At the reference spacing the case has 32,142 particles and no fluid.
    origin = SVector(-0.5, -1.5, -10.0)
    lattice = CSV.File(joinpath(@__DIR__, "data", "zipline", "lattice.csv"))
    node_positions = [Tuple{Int,Int,Int}[] for _ in 1:12]
    for row in lattice
        push!(node_positions[row.group], (row.i, row.j, row.k))
    end
    occupied = Set(vcat(node_positions...))
    floor = [(i, j, 0) for j in 0:74 for i in 0:288]
    end_wall = [(288, j, k) for k in 1:125 for j in 0:74]
    append!(node_positions[2], filter(node -> !(node in occupied), vcat(floor, end_wall)))
    positions = [[origin + dx * SVector(node) for node in nodes] for nodes in node_positions]

    SimConstants = SimulationConstants{FloatType}(Dimensions;
        dx  = dx,
        ρ₀  = 1000,
        g   = 9.81,
        c₀  = 20sqrt(9.81 * 0.8),
        α   = 0.01,
        CFL = 0.2,
    )

    Cable = SPHGeometry{Dimensions, FloatType}(positions[1];
        Density = SimConstants.ρ₀, GroupMarker = 1, Type = Fixed)
    Domain = SPHGeometry{Dimensions, FloatType}(positions[2];
        Density = SimConstants.ρ₀, GroupMarker = 2, Type = Fixed)

    # Reference rhopbody = 1 kg/m³, so relative weight is 1/1000. Every part
    # has its own mass, full inertia tensor, quaternion and velocity on CUDA.
    parts = [SPHGeometry{Dimensions, FloatType}(positions[marker];
        Density = SimConstants.ρ₀, GroupMarker = marker, Type = Floating,
        Floating = FloatingDetails{FloatType}(RelativeWeight = 0.001)) for marker in 3:12]
    Head, Body, ArmLeft, ArmRight, ForearmLeft, ForearmRight, LegLeft, LegRight, CalfLeft, CalfRight = parts
    SimulationGeometry = [Cable, Domain, parts...]

    # All attachment points and axes are the original world coordinates.
    # Four hinges constrain elbows/knees; five ball joints join the remaining
    # parts. The left forearm's attachment slides freely along the fixed line.
    joints = (
        HingeJoint(ArmLeft.GroupMarker, ForearmLeft.GroupMarker;
            Anchor = (0, -0.18, -0.381), Axis = (1, 0, 0)),
        HingeJoint(ArmRight.GroupMarker, ForearmRight.GroupMarker;
            Anchor = (0, 0.18, -0.381), Axis = (1, 0, 0)),
        HingeJoint(LegLeft.GroupMarker, CalfLeft.GroupMarker;
            Anchor = (0, -0.055, -0.837), Axis = (0, 1, 0)),
        HingeJoint(LegRight.GroupMarker, CalfRight.GroupMarker;
            Anchor = (0, 0.055, -0.837), Axis = (0, 1, 0)),
        BallJoint(Head.GroupMarker, Body.GroupMarker; Anchor = (0, 0, -0.125)),
        BallJoint(Body.GroupMarker, ArmLeft.GroupMarker; Anchor = (0, -0.136, -0.17)),
        BallJoint(Body.GroupMarker, ArmRight.GroupMarker; Anchor = (0, 0.136, -0.17)),
        BallJoint(Body.GroupMarker, LegLeft.GroupMarker; Anchor = (0, -0.055, -0.532)),
        BallJoint(Body.GroupMarker, LegRight.GroupMarker; Anchor = (0, 0.055, -0.532)),
        PointLineJoint(ForearmLeft.GroupMarker; Point = (0, -0.2, -0.6), Direction = (11, 0, -8.4)),
    )
    # Match the material coefficients. Keep a full-dp particle contact distance:
    # the reference's half-dp mesh overlap allowance cannot be used to shrink
    # sphere contacts without opening gaps in a sampled wall. The cable
    # is a visual guide, as in the reference's point-line constraint; only the
    # container participates in fixed-surface contacts. Same-body pairs never
    # collide. Native contacts use the particle surfaces, while Chrono uses CAD.
    contacts = RigidContacts(
        ParticleDiameter = dx, Slop = 0, FixedGroups = (Domain.GroupMarker,),
        Friction = 0.75, FixedFriction = 0.45, Restitution = 0.5, FixedRestitution = 0.8)
    SimRigidBodyDynamics = RigidBodyDynamics(Joints = joints, Contacts = contacts,
        Iterations = iterations, MaxTimeStep = max_timestep)

    SimMetaData = SimulationMetaData{
        Dimensions, FloatType, NoShifting, NoKernelOutput, NoMDBC, StoreLog,
    }(
        SimulationName      = "ZipLine3D",
        SaveLocation        = save_location,
        SimulationTime      = simulation_time,
        OutputTimes         = output_interval,
        VisualizeInParaview = false, # open after appending the CAD motion block
        ExportSingleVTKHDF  = true,
        OpenLogFile         = open_log_file,
        GPUDoublePosition   = true,
        GPUBoundaryForces   = false,
        GPUUseGraph         = use_graph,
        GPUMaxStepsPerSync  = 64,
        GPUCellSubdivision  = 1,
        GPUClipTimeStepToOutput = true,
    )

    SimKernel           = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); h = FloatType(1.3sqrt(3) * dx))
    SimViscosity        = ArtificialViscosity()
    SimDensityDiffusion = ZeroDensityDiffusion()

    mkpath(SimMetaData.SaveLocation)
    SimLogger    = SimulationLogger(SimMetaData.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaData)

    try
        RunSimulation(
            SimGeometry         = SimulationGeometry,
            SimMetaData         = SimMetaData,
            SimConstants        = SimConstants,
            SimLogger           = SimLogger,
            SimParticles        = SimParticles,
            SimKernel           = SimKernel,
            SimViscosity        = SimViscosity,
            SimDensityDiffusion = SimDensityDiffusion,
            SimTimeStepping     = SymplecticTimeStepping(),
            SimRigidBodyDynamics = SimRigidBodyDynamics,
        )
        save_zipline_surfaces(SimMetaData.SaveLocation, SimulationGeometry)
        SimMetaData.VisualizeInParaview = visualize
        open_zipline_output(SimMetaData)
    finally
        isopen(SimLogger.LoggerIo) && close(SimLogger.LoggerIo)
    end
end
