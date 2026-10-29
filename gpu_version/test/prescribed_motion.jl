module PrescribedMotionTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra, HDF5

const PM = SPHExampleGPU.PrescribedMotions
const ROT = SPHExampleGPU.PrescribedRotation
const STAGE = SPHExampleGPU.MotionStageSampling
const STEP = SPHExampleGPU.GPUStepState

include(joinpath(@__DIR__,"reference","dual_motion_cases.jl"))

@testset "independent DualSPHysics motion-engine position fixtures" begin
    for line in Iterators.drop(eachline(joinpath(@__DIR__,"reference","dual_motion_reference.csv")),1)
        fields = split(line,',')
        name, time = fields[1], parse(Float64,fields[2])
        expected = SVector{3,Float64}(parse.(Float64,fields[3:5]))
        motion = dual_motion_case(name)
        @test motion_position(motion,2,(2,0,1),time) ≈ expected atol = 2e-14
        if time > 0
            increment = SVector{3,Float64}(parse.(Float64,fields[6:8]))
            previous = motion_position(motion,2,(2,0,1),time-0.125)
            @test (motion_position(motion,2,(2,0,1),time)-previous)/0.125 ≈ increment atol = 2e-13
        end
    end
    # The analytical rate is intentional. An increment-averaged velocity is a
    # different boundary condition, even when positions match at every sample.
    motion = dual_motion_case("05")
    increment = (motion_position(motion,2,(2,0,1),0.125)-SVector(2.,0.,1.))/0.125
    @test norm(motion_velocity(motion,2,(2,0,1),0.125)-increment) > 0.01
end

@testset "MOTION01/02 pauses and ordered segments" begin
    path = MotionSequence([KinematicSegment(1; velocity = (2,0)),
        KinematicSegment(2; velocity = (0,0)), KinematicSegment(1; velocity = (0,-1))]; start_time = 1)
    for (t, x, v) in ((0, (0,0), (0,0)), (1.5, (1,0), (2,0)),
            (2, (2,0), (0,0)), (3, (2,0), (0,0)), (4, (2,0), (0,-1)),
            (4.5, (2,-0.5), (0,-1)), (5, (2,-1), (0,0)), (10, (2,-1), (0,0)))
        @test trajectory_state(path, t) == (SVector{2,Float64}(x), SVector{2,Float64}(v), SVector(0.,0.))
    end
    @test_throws ArgumentError KinematicSegment(0; velocity = 1)
    @test_throws ArgumentError KinematicSegment(Inf; velocity = 1)
    @test_throws ArgumentError KinematicSegment(1; velocity = NaN)
    @test_throws ArgumentError MotionSequence([])
    @test_throws ArgumentError MotionSequence([KinematicSegment(1; velocity = 1)]; start_time = -1)
    @test_throws ArgumentError MotionSequence([KinematicSegment(1; velocity = (1,0))]; initial_value = 1)
end

@testset "MOTION04 acceleration and cubic interpolation" begin
    path = MotionSequence([KinematicSegment(2; velocity = (1,0), acceleration = (2,-4)),
        KinematicSegment(1; velocity = (5,-8), acceleration = (0,0))])
    @test trajectory_state(path, 0.5) == (SVector(0.75,-0.5), SVector(2.,-2.), SVector(2.,-4.))
    @test trajectory_state(path, 2) == (SVector(6.,-8.), SVector(5.,-8.), SVector(0.,0.))
    @test trajectory_state(path, 3) == (SVector(11.,-16.), SVector(0.,0.), SVector(0.,0.))
    # A quadratic must be reproduced exactly, including its derivatives.
    cubic = SampledTrajectory([0,2], [(0,0), (6,-8)]; velocities = [(1,0), (5,-8)])
    for t in (0, 0.25, 1, 1.75)
        @test trajectory_state(cubic, t) == (SVector(t+t*t,-2t*t), SVector(1+2t,-4t), SVector(2.,-4.))
    end
    @test trajectory_state(cubic, 2) == (SVector(6.,-8.), SVector(0.,0.), SVector(0.,0.))
    @test_throws ArgumentError SampledTrajectory([0,0], [0,1])
    @test_throws ArgumentError SampledTrajectory([0,NaN], [0,1])
    @test_throws ArgumentError SampledTrajectory([-1,0], [0,1])
    @test_throws ArgumentError SampledTrajectory([0,1], [0,Inf])
    @test_throws DimensionMismatch SampledTrajectory([0,1], [0])
    @test_throws DimensionMismatch SampledTrajectory([0,1], [0,1]; velocities = [0])
    @test_throws ArgumentError SampledTrajectory([0,1e-300], [0,0]; velocities = [1,1])
    @test_throws ArgumentError SinusoidalSegment(10; amplitude=1,frequency=1e308)
end

@testset "MOTION03 nested transforms, rates and validation" begin
    angle = MotionSequence([KinematicSegment(3; velocity = pi/2)])
    parent = PrescribedMotion(10; group_marker = nothing, rotation = angle)
    child = PrescribedMotion(11; parent = 10,
        translation = MotionSequence([KinematicSegment(3; velocity = (1,0))]))
    grandchild = PrescribedMotion(12; parent = 11, center = (1,0), rotation = angle)
    system = MotionSystem((grandchild, child, parent))
    @test map(n -> n.id, system.motions) == (10,11,12)
    @test motion_position(system, 11, (1,0), 1) ≈ SVector(0.,2.) atol = 3e-15
    @test motion_velocity(system, 11, (1,0), 1) ≈ SVector(-pi,1.) atol = 3e-15
    @test motion_acceleration(system, 11, (1,0), 1) ≈ SVector(-pi,-pi*pi/2) atol = 3e-15
    @test motion_position(system, 12, (2,0), 1) ≈ SVector(-1.,2.) atol = 3e-15
    for id in (11,12), t in (0.2,0.7,1.4)
        x = SVector(2.,0.3)
        delta = 1e-5
        fd = (motion_position(system,id,x,t+delta) - motion_position(system,id,x,t-delta))/(2delta)
        fa = (motion_velocity(system,id,x,t+delta) - motion_velocity(system,id,x,t-delta))/(2delta)
        @test motion_velocity(system,id,x,t) ≈ fd rtol = 2e-9 atol = 1e-9
        @test motion_acceleration(system,id,x,t) ≈ fa rtol = 2e-9 atol = 1e-9
    end
    @test_throws ArgumentError MotionSystem((parent,parent))
    @test_throws ArgumentError MotionSystem((PrescribedMotion(1; parent=2),))
    @test_throws ArgumentError MotionSystem((PrescribedMotion(1; parent=2), PrescribedMotion(2; parent=1)))
    @test_throws ArgumentError MotionSystem((PrescribedMotion(1; parent=1),))
    @test_throws ArgumentError MotionSystem((PrescribedMotion(1), PrescribedMotion(2; group_marker=1)))
    @test_throws DimensionMismatch MotionSystem((PrescribedMotion(1), PrescribedMotion(2; center=(0,0,0))))
    @test_throws ArgumentError motion_state(system, 99, 0)
    @test_throws ArgumentError PrescribedMotion(1; axis = (1,0,0))
    @test_throws ArgumentError PrescribedMotion(1; axis = (0,0,0))
    @test_throws DimensionMismatch PrescribedMotion(1; center=(0,0,0), translation=child.translation)
end

@testset "MOTION05/06 rotation and circular translation in 2D/3D" begin
    angles = MotionSequence([KinematicSegment(3; velocity = 0, acceleration = pi)])
    rotation = PrescribedMotion(1; rotation = angles, center = (1,0))
    @test motion_position(rotation, (2,0), 1) ≈ SVector(1.,1.) atol = 1e-15
    @test motion_velocity(rotation, (2,0), 1) ≈ SVector(-pi,0.) atol = 1e-15
    @test motion_acceleration(rotation, (2,0), 1) ≈ SVector(-pi,-pi*pi) atol = 2e-15
    circular = CircularTrajectory(angles; reference=(2,0), center=(1,0))
    translation = PrescribedMotion(1; translation=circular)
    @test motion_position(translation,(2,0),1) ≈ SVector(1.,1.) atol = 1e-15
    @test motion_position(translation,(3,0),1) ≈ SVector(2.,1.) atol = 1e-15
    @test motion_velocity(translation,(3,0),1) ≈ SVector(-pi,0.) atol = 1e-15
    for axis in ((1,0,0),(0,1,0),(0,0,1),(1,2,3))
        m = PrescribedMotion(2; rotation=angles, center=(1,2,3), axis)
        pose = motion_state(m,0.7)
        @test pose.rotation' * pose.rotation ≈ I atol = 1e-15
        @test det(pose.rotation) ≈ 1 atol = 1e-15
        @test motion_position(m,(1,2,3),1) ≈ SVector(1.,2.,3.) atol = 1e-15
        p, t, delta = SVector(2.,4.,1.), 0.7, 1e-5
        @test motion_velocity(m,p,t) ≈ (motion_position(m,p,t+delta)-motion_position(m,p,t-delta))/(2delta) rtol = 2e-9
    end
    m = PrescribedMotion(1; rotation=angles, center=(0,0,0), axis=(0,1,0))
    @test motion_position(m,(1,0,0),1) ≈ SVector(0.,0.,-1.) atol = 1e-15
end

@testset "MOTION07 sinusoidal translation, rotation and circular translation" begin
    phase = pi/6
    path = MotionSequence([SinusoidalSegment(2; amplitude=(2,1), frequency=(0.5,1), phase=(phase,0))])
    @test first(trajectory_state(path,0)) == SVector(0.,0.)
    p,v,a = trajectory_state(path,0.25)
    @test p ≈ SVector(2*(sin(pi/4+phase)-sin(phase)),1.) atol = 2e-15
    @test v ≈ SVector(2pi*cos(pi/4+phase),0.) atol = 2e-15
    @test a ≈ SVector(-2pi*pi*sin(pi/4+phase),-4pi*pi) atol = 1e-14
    angles = MotionSequence([SinusoidalSegment(2; amplitude=pi/2, frequency=1)])
    rot = PrescribedMotion(1; rotation=angles)
    cir = PrescribedMotion(2; translation=CircularTrajectory(angles; reference=(1,0)))
    @test motion_position(rot,(1,0),0.25) ≈ SVector(0.,1.) atol = 1e-15
    @test motion_position(cir,(2,0),0.25) ≈ SVector(1.,1.) atol = 1e-15
    @test_throws ArgumentError SinusoidalSegment(1; amplitude=1,frequency=-1)
end

@testset "MOTION08/09 file data, relative origins, units and validation" begin
    mktempdir() do dir
        path = joinpath(dir,"translation.csv")
        write(path,"# time;x;unused;z\n10;5;0;2\n11;7;0;1\n13;7;0;5 # endpoint\n")
        tr = load_translation_trajectory(path; value_columns=(2,4),start_time=2)
        @test trajectory_state(tr,1) == (SVector(0.,0.),SVector(0.,0.),SVector(0.,0.))
        @test trajectory_state(tr,2.5) == (SVector(1.,-0.5),SVector(2.,-1.),SVector(0.,0.))
        @test trajectory_state(tr,3) == (SVector(2.,-1.),SVector(0.,2.),SVector(0.,0.))
        @test trajectory_state(tr,5) == (SVector(2.,3.),SVector(0.,0.),SVector(0.,0.))
        absolute = load_translation_trajectory(path; value_columns=(2,4),relative=false)
        @test first(trajectory_state(absolute,0)) == SVector(5.,2.)
        rotpath = joinpath(dir,"rotation.dat")
        write(rotpath,"time,angle\n10,30\n11,120\n12,390\n")
        rot = load_rotation_trajectory(rotpath;angle_units=:degrees,skip_rows=1)
        @test all(isapprox.(trajectory_state(rot,0.5), (pi/4,pi/2,0.)))
        @test all(isapprox.(trajectory_state(rot,2), (2pi,0.,0.)))
        @test_throws ArgumentError load_rotation_trajectory(rotpath)
        @test_throws ArgumentError load_rotation_trajectory(rotpath;angle_units=:turns)
        @test_throws ArgumentError load_translation_trajectory(path;value_columns=(1,2))
        @test_throws ArgumentError load_translation_trajectory(path;value_columns=(2,9))
        @test_throws ArgumentError load_translation_trajectory(path;skip_rows=-1)
        for bad in ("0 1\n0 2\n", "0 1\n-1 2\n", "0 1\n1 NaN\n", "0 1\n1 2 3\n", "0 1\n", "")
            write(rotpath,bad)
            @test_throws ArgumentError load_rotation_trajectory(rotpath)
        end
    end
end

function test_system(D)
    zero_point = ntuple(_ -> 0.,D)
    point(x,y=0) = ntuple(d -> d==1 ? x : d==D ? y : 0.,D)
    tr = MotionSequence([KinematicSegment(0.03;velocity=point(1,0.2),acceleration=point(2,-1)),
        KinematicSegment(0.01;velocity=zero_point),
        SinusoidalSegment(0.04;amplitude=point(0.03,0.01),frequency=point(3,4))])
    rot = SampledTrajectory([0,0.02,0.04,0.08],[0,0.04,-0.02,0.1];velocities=[2,0,-1,0])
    axis = D==2 ? (0,0,1) : (1,2,3)
    return MotionSystem((PrescribedMotion(3;parent=2,center=zero_point,rotation=rot),
        PrescribedMotion(1;group_marker=nothing,center=zero_point,rotation=rot,axis),
        PrescribedMotion(2;parent=1,center=zero_point,translation=tr),
        PrescribedMotion(4;center=zero_point,translation=CircularTrajectory(rot;reference=point(0.1),axis))))
end

if CUDA.functional()
    @testset "ownership conflicts and unsupported boundary modes fail explicitly" begin
        for B in (SimpleMDBC,UpdatedMDBC), D in (2,3)
            mktempdir() do dir
                T = Float32
                geometry = [SPHGeometry{D,T}([zero(SVector{D,T})];Density=1000,GroupMarker=1,Type=Moving)]
                meta = SimulationMetaData{D,T,NoShifting,NoKernelOutput,B,NoLog}(
                    SimulationName="InvalidMotion",SaveLocation=dir,VisualizeInParaview=false,OpenLogFile=false)
                particles = AllocateDataStructures(geometry,meta)
                logger = SimulationLogger(dir)
                try
                    @test_throws ArgumentError RunSimulation(SimGeometry=geometry,SimMetaData=meta,SimParticles=particles,
                        SimConstants=SimulationConstants{T}(D;dx=0.01,c₀=10),
                        SimKernel=SPHKernelInstance{D,T}(WendlandC2();dx=T(0.01)),
                        SimViscosity=ZeroViscosity(),SimDensityDiffusion=ZeroDensityDiffusion(),
                        SimTimeStepping=SymplecticTimeStepping(),SimLogger=logger,
                        SimMotion=PrescribedMotion(1;center=ntuple(_->0,D)))
                finally
                    isopen(logger.LoggerIo) && close(logger.LoggerIo)
                end
            end
        end
        mktempdir() do dir
            T,D = Float32,2
            geometry = [SPHGeometry{D,T}([SVector(0f0,0f0)];Density=1000,GroupMarker=1,Type=Moving)]
            meta = SimulationMetaData{D,T}(SimulationName="PeriodicMotion",SaveLocation=dir,
                Periodicity=PeriodicBoundary{D}(1;min=-1,max=1),VisualizeInParaview=false,OpenLogFile=false)
            particles = AllocateDataStructures(geometry,meta)
            logger = SimulationLogger(dir)
            try
                @test_throws ArgumentError RunSimulation(SimGeometry=geometry,SimMetaData=meta,SimParticles=particles,
                    SimConstants=SimulationConstants{T}(D;dx=0.01,c₀=10),
                    SimKernel=SPHKernelInstance{D,T}(WendlandC2();dx=T(0.01)),
                    SimViscosity=ZeroViscosity(),SimDensityDiffusion=ZeroDensityDiffusion(),
                    SimTimeStepping=SymplecticTimeStepping(),SimLogger=logger,
                    SimMotion=PrescribedMotion(1;translation=MotionSequence([KinematicSegment(1;velocity=(1,0))])))
            finally
                isopen(logger.LoggerIo) && close(logger.LoggerIo)
            end
            old = RotationMotion(1;times=[0,1],angles=[0,1])
            @test_throws ArgumentError PM.combine_motions(old,PrescribedMotion(1),geometry,particles)
            moving = (GroupMarker=1,Motion=MotionDetails{D,T}(Velocity=1,StartTime=0,Duration=1,Direction=SVector(1,0)),)
            @test_throws ArgumentError PM.prepare_prescribed_motion(PrescribedMotion(1),(moving,),particles)
        end
    end

    @testset "absolute GPU poses at all stages after repeated sorts" begin
        for D in (2,3), T in (Float32,Float64), TP in (Float32,Float64)
            V = SVector{D,TP}
            reference = [V(ntuple(d -> i/(d+1),D)) for i in 1:5]
            host = (Position=reference,Type=[Moving,Moving,Moving,Fluid,Fixed],
                GroupMarker=[2,3,4,5,6],ID=[1001,8,77,400,53])
            system = test_system(D)
            motion = PM.prepare_prescribed_motion(system,(),host)
            order = collect(1:5)
            owner_pointer = pointer(motion.indices)
            for perm in ([3,2,5,1,4],[5,1,3,4,2],[4,3,2,1,5])
                order = order[perm]
                ROT.update_rotation_indices!(motion,(Perm=CuArray(Int32.(perm)),CellIDScratch=CUDA.zeros(Int32,5)))
                @test pointer(motion.indices) == owner_pointer
                p,v = CuArray(reference[order]),CUDA.zeros(SVector{D,T},5)
                ty,gm,ids = CuArray(host.Type[order]),CuArray(host.GroupMarker[order]),CuArray(host.ID[order])
                for time in (0.0,0.019,0.03,0.04,0.079,0.08), fraction in (0.,0.5,1.)
                    state = StepState{T}(;time,dt=T(0.007))
                    ROT.launch_rotation!(motion,p,v,ty,gm,ids,state;fraction)
                    sampled = Float64(time) + T(fraction)*T(0.007)
                    positions,velocities = Array(p),Array(v)
                    for i in eachindex(order)
                        j = order[i]
                        if j<=3
                            marker = host.GroupMarker[j]
                            @test positions[i] ≈ V(motion_position(system,marker,reference[j],sampled)) atol = 4eps(TP) rtol = 4eps(TP)
                            @test velocities[i] ≈ SVector{D,T}(motion_velocity(system,marker,reference[j],sampled)) atol = 32eps(T) rtol = 32eps(T)
                        else
                            @test positions[i] == reference[j]
                            @test iszero(velocities[i])
                        end
                    end
                    before = (positions,velocities,Array(motion.poses))
                    STEP.set_slot!(state.i,state.ih,STEP.I_STOP,STEP.STOP_REBUILD)
                    ROT.launch_rotation!(motion,p,v,ty,gm,ids,state;fraction=1)
                    @test (Array(p),Array(v),Array(motion.poses)) == before
                end
                ty .= Frozen
                before = (Array(p),Array(v))
                ROT.launch_rotation!(motion,p,v,ty,gm,ids,HostStep(T(0.01),T(0.02)))
                @test (Array(p),Array(v)) == before
            end
            @test_throws ArgumentError PM.prepare_prescribed_motion(PrescribedMotion(99;center=ntuple(_->0,D)),(),host)
            @test_throws ArgumentError PM.prepare_prescribed_motion(PrescribedMotion(5;center=ntuple(_->0,D)),(),host)
        end
    end

    function run_motion(D,T,scheme,graph; legacy=false)
        mktempdir() do dir
            V = SVector{D,T}
            point(x,y=0) = V(ntuple(d -> d==1 ? x : d==D ? y : 0,D))
            geometry = [SPHGeometry{D,T}([point(x),point(x,0.01)];Density=1000,GroupMarker=g,Type=Moving)
                for (g,x) in ((2,0.4),(3,0.2),(4,0.0))]
            metadata = SimulationMetaData{D,T}(SimulationName="PrescribedMotion",SaveLocation=dir,
                SimulationTime=0.055,OutputTimes=0.011,GPUUseGraph=graph,
                GPUMaxStepsPerSync=graph ? 4 : 1,GPUClipTimeStepToOutput=true,
                VisualizeInParaview=false,OpenLogFile=false,ExportGridCells=false,GPUDoublePosition=true)
            particles = AllocateDataStructures(geometry,metadata)
            reference = Dict(id=>x for (id,x) in zip(particles.ID,particles.Position))
            system = test_system(D)
            # Both APIs may coexist when their particle groups are disjoint.
            old = legacy ? RotationMotion(4;times=[0,0.08],angles=[0,0.1]) : nothing
            if legacy
                system = MotionSystem(Tuple(n for n in system.motions if n.group_marker != 4))
            end
            logger = SimulationLogger(dir)
            result = try
                RunSimulation(SimGeometry=geometry,SimMetaData=metadata,SimParticles=particles,
                    SimConstants=SimulationConstants{T}(D;dx=0.01,c₀=10,g=0),
                    SimKernel=SPHKernelInstance{D,T}(WendlandC2();dx=T(0.01)),
                    SimViscosity=ZeroViscosity(),SimDensityDiffusion=ZeroDensityDiffusion(),
                    SimTimeStepping=scheme,SimLogger=logger,SimMotion=system,SimRotation=old)
            finally
                isopen(logger.LoggerIo) && close(logger.LoggerIo)
            end
            @test result.final_state.complete
            @test result.diagnostics.cell_list_rebuilds > 1
            @test result.diagnostics.frozen_particles == 0
            @test result.configuration.execution.graph_capture_failures == 0
            graph && @test result.configuration.execution.graph_batches > 0
            t = result.final_state.time
            for i in eachindex(particles.ID)
                ref,g = reference[particles.ID[i]],Int(particles.GroupMarker[i])
                expected = legacy && g==4 ? rotation_position(old,ref,t) : motion_position(system,g,ref,t)
                vel = if legacy && g==4
                    _,w = rotation_state(old,t)
                    SVector(-w*expected[2],w*expected[1])
                else
                    motion_velocity(system,g,ref,t)
                end
                @test particles.Position[i] ≈ expected atol = 1e-12
                @test particles.Velocity[i] ≈ vel atol = 2e-6
            end
            order = sortperm(particles.ID)
            return (;position=particles.Position[order],velocity=particles.Velocity[order],density=particles.Density[order],time=t)
        end
    end

    @testset "both integrators and CUDA graph replay in 2D/3D" begin
        for D in (2,3), T in (Float32,Float64), scheme in (SymplecticTimeStepping(),SingleNeighborTimeStepping())
            @test run_motion(D,T,scheme,false) == run_motion(D,T,scheme,true)
        end
        @test run_motion(2,Float32,SymplecticTimeStepping(),false;legacy=true) ==
            run_motion(2,Float32,SymplecticTimeStepping(),true;legacy=true)
    end
end

end
