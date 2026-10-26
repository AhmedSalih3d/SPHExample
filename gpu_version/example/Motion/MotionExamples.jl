module MotionExamples

using SPHExampleGPU
using StaticArrays

export motion_example, run_motion_example

"""
    motion_example(number; dimensions = 2, FloatType = Float32, dx = 0.025)

Build a small, entirely Julia-generated kinematic demonstration of MOTION01–09.
The blocks are prescribed `Moving` particles, with no fluid or gravity. These
examples demonstrate motion features, rather than reproducing upstream XML
geometry or fluid results. Positions use (x,z) in 2D and (x,y,z) in 3D.
"""
function motion_example(number; dimensions = 2, FloatType = Float32, dx = 0.025)
    number in 1:9 || throw(ArgumentError("motion example number must be in 1:9"))
    dimensions in (2,3) || throw(ArgumentError("dimensions must be 2 or 3"))
    isfinite(dx) && 0 < dx <= 0.05 || throw(ArgumentError("dx must be finite and in (0,0.05]"))
    D, T = dimensions, FloatType
    point(x,z=0) = SVector{D,Float64}(ntuple(d -> d==1 ? x : d==D ? z : 0.,D))
    center(g) = point(0.3*(g-1),0.15*(g-1))
    axis = D==2 ? (0,0,1) : (1,2,3)
    angle(v,a=0) = MotionSequence([KinematicSegment(4;velocity=v,acceleration=a)])
    linear(v,a=zero(v)) = MotionSequence([KinematicSegment(4;velocity=v,acceleration=a)])
    sinusoid(amplitude,frequency,phase=zero(amplitude)) = MotionSequence([SinusoidalSegment(4;amplitude,frequency,phase)])
    motions = if number == 1
        translation = MotionSequence([KinematicSegment(1;velocity=point(0.1)),
            KinematicSegment(1;velocity=point(0)),KinematicSegment(1;velocity=point(-0.1)),
            KinematicSegment(1;velocity=point(0))])
        (PrescribedMotion(1;center=center(1),translation),)
    elseif number == 2
        translation = MotionSequence([KinematicSegment(1;velocity=v) for v in
            (point(0.08),point(0,0.08),point(-0.08),point(0,-0.08))])
        (PrescribedMotion(1;center=center(1),translation),)
    elseif number == 3
        (PrescribedMotion(1;center=center(1),axis,translation=linear(point(0.02)),rotation=angle(0.35)),
            PrescribedMotion(2;parent=1,center=center(2),axis,rotation=angle(-0.6)),
            PrescribedMotion(3;parent=2,center=center(3),translation=sinusoid(point(0.04),point(0.5))))
    elseif number == 4
        translation = MotionSequence([KinematicSegment(2;velocity=point(0.05),acceleration=point(0.025)),
            KinematicSegment(2;velocity=point(0.1),acceleration=point(-0.025))])
        (PrescribedMotion(1;center=center(1),translation),)
    elseif number == 5
        (PrescribedMotion(1;center=center(1),axis,rotation=angle(pi/2)),)
    elseif number == 6
        angles = angle(0.2,0.3)
        (PrescribedMotion(1;center=center(1),axis,rotation=angles),
            PrescribedMotion(2;center=center(2),translation=CircularTrajectory(angles;
                reference=center(2),center=center(2)-point(0.15),axis)))
    elseif number == 7
        angles = sinusoid(pi/4,0.5,pi/6)
        (PrescribedMotion(1;center=center(1),translation=sinusoid(point(0.1,0.05),point(0.5,1))),
            PrescribedMotion(2;center=center(2),axis,rotation=angles),
            PrescribedMotion(3;center=center(3),translation=CircularTrajectory(angles;
                reference=center(3),center=center(3)-point(0.15),axis)))
    elseif number == 8
        translation = load_translation_trajectory(joinpath(@__DIR__,"data","translation.csv");
            value_columns=D==2 ? (2,4) : (2,3,4))
        (PrescribedMotion(1;center=center(1),translation),)
    else
        degrees = load_rotation_trajectory(joinpath(@__DIR__,"data","rotation_degrees.csv");angle_units=:degrees)
        radians = load_rotation_trajectory(joinpath(@__DIR__,"data","rotation_radians.csv");angle_units=:radians)
        (PrescribedMotion(1;center=center(1),axis,rotation=degrees),
            PrescribedMotion(2;center=center(2),axis,rotation=radians))
    end
    radius = max(1,floor(Int,0.05/dx))
    geometry = [SPHGeometry{D,T}([center(g) + point(i*dx,j*dx)
            for i in -radius:radius for j in -radius:radius];
            Density=1000,GroupMarker=g,Type=Moving) for g in 1:length(motions)]
    return (;geometry,motion=MotionSystem(motions),duration=4.0,
        constants=SimulationConstants{T}(D;dx,c₀=10,g=0))
end

"""
    run_motion_example(number; dimensions = 2, simulation_time = 4, ...)

Run a motion demonstration and write VTKHDF particle output. Set `visualize`
and `open_log_file` to false for unattended runs. `time_stepping` accepts both
supported integrators; `use_graph` enables CUDA graph replay.
"""
function run_motion_example(number; dimensions = 2, FloatType = Float32, dx = 0.025,
        simulation_time = 4.0, output_interval = 0.05, visualize = true,
        open_log_file = true, use_graph = true,
        time_stepping = SymplecticTimeStepping(),
        save_location = joinpath(@__DIR__,"..","..","output","Motion",lpad(number,2,'0')))
    isfinite(simulation_time) && simulation_time > 0 || throw(ArgumentError("simulation_time must be finite and positive"))
    isfinite(output_interval) && output_interval > 0 || throw(ArgumentError("output_interval must be finite and positive"))
    case = motion_example(number;dimensions,FloatType,dx)
    D,T = dimensions,FloatType
    mkpath(save_location)
    metadata = SimulationMetaData{D,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName="Motion"*lpad(number,2,'0'),SaveLocation=save_location,
        SimulationTime=simulation_time,OutputTimes=output_interval,
        GPUUseGraph=use_graph,GPUDoublePosition=true,GPUClipTimeStepToOutput=true,
        GPUCellSubdivision=1,ExportGridCells=false,
        VisualizeInParaview=visualize,OpenLogFile=open_log_file)
    particles = AllocateDataStructures(case.geometry,metadata)
    logger = SimulationLogger(save_location)
    try
        return RunSimulation(SimGeometry=case.geometry,SimMetaData=metadata,SimParticles=particles,
            SimConstants=case.constants,SimKernel=SPHKernelInstance{D,T}(WendlandC2();dx=T(dx)),
            SimLogger=logger,SimTimeStepping=time_stepping,SimViscosity=ZeroViscosity(),
            SimDensityDiffusion=ZeroDensityDiffusion(),SimMotion=case.motion)
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end

end
