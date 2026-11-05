# DualSPHysics 5.4 examples/chrono/04_Pelamis, with native CUDA joints.
# Reference world (x,z) coordinates become simulation (x,y) coordinates.
# Including or running this file starts the simulation.
#
#     julia --project=gpu_version -t 2,0 gpu_version/example/Pelamis2d.jl
using SPHExampleGPU
using StaticArrays: SVector
using CSV
include(joinpath(@__DIR__,"data","pelamis","output.jl"))

let
    Dimensions = 2
    FloatType  = Float32

    dx              = 0.025
    simulation_time = 10.0
    output_interval = 0.05
    save_location   = joinpath(@__DIR__,"..","output","pelamis_2d")
    visualize       = true
    open_log_file   = true
    use_graph       = true
    iterations      = 32
    time_stepping   = SymplecticTimeStepping()
    wave_height     = 0.2
    wave_period     = 1.5
    water_depth     = 1.5

    dx == 0.025 || throw(ArgumentError("the reference GenCase lattice uses dx = 0.025 m; regenerate it before changing resolution"))
    isfinite(simulation_time) && simulation_time > 0 && isfinite(output_interval) && output_interval > 0 ||
        throw(ArgumentError("simulation_time and output_interval must be finite and positive"))

    # Preserve GenCase's clipped STL fill, shared-end ownership, bed sampling
    # and initial fluid density. No STL reader or DualSPHysics installation is needed.
    groups = (1,10,11,12,13,20,21)
    points = Dict(g => SVector{2,Float64}[] for g in groups)
    densities = Dict(g => FloatType[] for g in groups)
    for row in CSV.File(joinpath(@__DIR__,"data","pelamis","lattice.csv"))
        push!(points[row.group],SVector(2.8+dx*row.i,-0.1+dx*row.k))
        push!(densities[row.group],FloatType(row.density))
    end
    sound_speed = 20sqrt(9.81*1.45) # sampled fluid column: z = 0.025 to 1.475
    SimConstants = SimulationConstants{FloatType}(Dimensions;
        dx = dx,ρ₀ = 1000,g = 9.81,c₀ = sound_speed,α = 0.01,CFL = 0.2)
    Beach = SPHGeometry{Dimensions,FloatType}(points[21];Density = densities[21],GroupMarker = 21,Type = Fixed)
    Piston = SPHGeometry{Dimensions,FloatType}(points[20];Density = densities[20],GroupMarker = 20,Type = Moving)
    segments = [SPHGeometry{Dimensions,FloatType}(points[g];Density = densities[g],GroupMarker = g,Type = Floating,
        Floating = FloatingDetails{FloatType}(RelativeWeight = 0.5)) for g in 10:13]
    Segment1,Segment2,Segment3,Segment4 = segments
    Water = SPHGeometry{Dimensions,FloatType}(points[1];Density = densities[1],GroupMarker = 1,Type = Fluid)
    SimulationGeometry = [Beach,Piston,segments...,Water]

    # The first attachment heaves on a fixed vertical line. The other links
    # permit relative pitch. Coordinates and hinge axes are the reference world values.
    joints = (
        PointLineJoint(Segment1.GroupMarker;Point = (5.8,0,1.5),Direction = (0,0,1)),
        HingeJoint(Segment1.GroupMarker,Segment2.GroupMarker;Anchor = (7.15,0,1.5),Axis = (0,1,0)),
        HingeJoint(Segment2.GroupMarker,Segment3.GroupMarker;Anchor = (8.45,0,1.5),Axis = (0,1,0)),
        HingeJoint(Segment3.GroupMarker,Segment4.GroupMarker;Anchor = (9.75,0,1.5),Axis = (0,1,0)),
    )
    SimRigidBodyDynamics = RigidBodyDynamics(Joints = joints,Contacts = nothing,Iterations = iterations)

    # Second-order regular piston waves, with phase zero and no ramp as in
    # the reference. Solve the dispersion relation once during setup.
    omega = 2pi/wave_period
    wave_number = omega^2/9.81
    for _ in 1:50
        kd = wave_number*water_depth
        wave_number -= (9.81*wave_number*tanh(kd)-omega^2)/(9.81*tanh(kd)+9.81*wave_number*water_depth*sech(kd)^2)
    end
    kd = wave_number*water_depth
    transfer = 2sinh(kd)^2/(sinh(kd)*cosh(kd)+kd)
    a1 = wave_height/(2transfer)
    a2 = wave_height^2/(32water_depth)*(3cosh(kd)/sinh(kd)^3-2/transfer)
    piston_state(t) = (a1*sin(omega*t)+a2*sin(2omega*t),a1*omega*cos(omega*t)+2a2*omega*cos(2omega*t))
    motion_times = collect(0.0:wave_period/200:simulation_time+wave_period)
    motion_states = piston_state.(motion_times)
    PistonMotion = PrescribedMotion(Piston.GroupMarker;
        translation = SampledTrajectory(motion_times,[(s[1],0.) for s in motion_states];velocities = [(s[2],0.) for s in motion_states]),
        center = (0.,0.))
    WaveGauges = MeasurementConfig(water_column_probes = [
        WaterColumnProbe("WG4",(4.,0.);radius = 2dx),WaterColumnProbe("WG12",(12.,0.3);radius = 2dx)])

    SimMetaData = SimulationMetaData{Dimensions,FloatType,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = "Pelamis2D",SaveLocation = save_location,SimulationTime = simulation_time,
        OutputTimes = output_interval,VisualizeInParaview = false,OpenLogFile = open_log_file,
        ExportSingleVTKHDF = true,GPUDoublePosition = true,GPUBoundaryForces = false,
        GPUUseGraph = use_graph,GPUMaxStepsPerSync = 64,GPUCellSubdivision = 1,GPUClipTimeStepToOutput = true)
    SimKernel = SPHKernelInstance{Dimensions,FloatType}(WendlandC2();h = FloatType(1.1sqrt(2)*dx),η² = FloatType((0.1*1.1sqrt(2)*dx)^2))
    SimViscosity = ArtificialViscosity(boundary_factor = 0)
    SimDensityDiffusion = FourtakasDensityDiffusion()

    mkpath(SimMetaData.SaveLocation)
    SimLogger = SimulationLogger(SimMetaData.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry,SimMetaData)
    try
        RunSimulation(SimGeometry = SimulationGeometry,SimMetaData = SimMetaData,SimConstants = SimConstants,
            SimParticles = SimParticles,SimKernel = SimKernel,SimLogger = SimLogger,SimViscosity = SimViscosity,
            SimDensityDiffusion = SimDensityDiffusion,SimTimeStepping = time_stepping,
            SimRigidBodyDynamics = SimRigidBodyDynamics,SimMotion = PistonMotion,SimMeasurements = WaveGauges)
        save_pelamis_surfaces(SimMetaData.SaveLocation,SimulationGeometry;piston_state)
        SimMetaData.VisualizeInParaview = visualize
        open_pelamis_output(SimMetaData)
    finally
        isopen(SimLogger.LoggerIo) && close(SimLogger.LoggerIo)
    end
end
