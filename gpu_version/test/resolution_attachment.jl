module ResolutionAttachmentTests
using Test, SPHExampleGPU, CUDA, StaticArrays, StructArrays, LinearAlgebra, HDF5
const VR = SPHExampleGPU.VariableResolution

function attachment_case(::Type{T}=Float32; floating=false, gravity=0, details=nothing, wall_ghosts=false, wet=false) where T
    region = VariableResolutionBox((-1.0,-1.0),(1.0,1.0))
    function level(dx)
        constants = SimulationConstants{T}(2;dx,c₀=10,g=gravity,α=0,δᵩ=0)
        kernel = SPHKernelInstance{2,T}(WendlandC2();h=T(dx))
        positions = wet ? [SVector(x,y) for x in -1.5:dx:1.5 for y in -1.5:dx:1.5 if max(abs(x),abs(y)) > 0.22] :
            [SVector(-1.3,0.0),SVector(0.7,0.7)]
        fluid = SPHGeometry{2,T}(positions;Density=T(1000),Type=Fluid,GroupMarker=2)
        points = [SVector(-0.1,-0.1),SVector(0.1,-0.1),SVector(0.1,0.1),SVector(-0.1,0.1)]
        fields = wall_ghosts ? (;GhostPoints=points .+ Ref(SVector(0.02,0.0)),GhostNormals=fill(SVector(1.0,0.0),4)) : (;)
        data = particle_struct_array(points,T(1000);fields...)
        body = SPHGeometry{2,T}(;Particles=data,Type=floating ? Floating : Moving,GroupMarker=5,
            Motion=floating ? nothing : details,Floating=floating ?
                FloatingDetails{T}(;RelativeWeight=T(0.8),Constraints=details) : nothing)
        return (;geometry=[fluid,body],constants,kernel)
    end
    return (;region,coarse=level(0.1),fine=level(0.05))
end

function run_attachment(case;motion=nothing,rotation=nothing,graphs=true,duration=0.01,
        scheme=SymplecticTimeStepping(),boundary=NoMDBC,single=true,follow=5,grid=false,kwargs...)
    directory = mktempdir()
    T = typeof(case.coarse.constants.dx)
    config = VariableResolutionConfig(case.region;fine_geometry=case.fine.geometry,follow,particle_reserve=4096)
    meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,boundary,NoLog}(;
        SimulationName="Attached",SaveLocation=directory,SimulationTime=duration,OutputTimes=duration/2,
        VariableResolution=config,GPUUseGraph=graphs,GPUMaxStepsPerSync=8,GPUClipTimeStepToOutput=true,
        ExportSingleVTKHDF=single,ExportGridCells=grid,VisualizeInParaview=false,OpenLogFile=false,
        OutputVariables=["Velocity","Density","Type","ID","GroupMarker","Resolution","Owned","IsBuffer"],kwargs...)
    particles = AllocateDataStructures(case.coarse.geometry,meta)
    logger = SimulationLogger(directory)
    result = try
        redirect_stdout(devnull) do
            RunSimulation(;SimGeometry=case.coarse.geometry,SimConstants=case.coarse.constants,SimKernel=case.coarse.kernel,
                SimParticles=particles,SimMetaData=meta,SimLogger=logger,SimMotion=motion,SimRotation=rotation,
                SimDensityDiffusion=ZeroDensityDiffusion(),SimViscosity=ZeroViscosity(),SimTimeStepping=scheme)
        end
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    return (;result,particles,meta,directory)
end

function uniform_floating(case;duration=0.0001)
    level = case.fine
    T = typeof(level.constants.dx)
    directory = mktempdir()
    meta = SimulationMetaData{2,T,NoShifting,NoKernelOutput,NoMDBC,NoLog}(;
        SimulationName="UniformBody",SaveLocation=directory,SimulationTime=duration,OutputTimes=duration/2,
        GPUDoublePosition=true,GPUClipTimeStepToOutput=true,GPULanesPerParticle=1,GPUInteractionFastMath=false,
        GPUUseGraph=false,VisualizeInParaview=false,OpenLogFile=false)
    particles = AllocateDataStructures(level.geometry,meta)
    logger = SimulationLogger(directory)
    try
        return redirect_stdout(devnull) do
            RunSimulation(;SimGeometry=level.geometry,SimConstants=level.constants,SimKernel=level.kernel,
                SimParticles=particles,SimMetaData=meta,SimLogger=logger,
                SimDensityDiffusion=ZeroDensityDiffusion(),SimViscosity=ZeroViscosity(),SimTimeStepping=SymplecticTimeStepping())
        end
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end

function resident_attachment(case,motion;grow=false)
    config = VariableResolutionConfig(case.region;fine_geometry=case.fine.geometry,follow=5)
    settings = VR.resolution_attachment_config(config,case.coarse.geometry,nothing,motion,case.coarse.kernel)
    base = VR.resolution_limits(case.region,Float64(case.coarse.kernel.h),Float64(case.fine.kernel.h))
    coarse = VR.make_resolution_domain([case.coarse.geometry[1]],case.coarse.constants,case.coarse.kernel,base,false,256;
        resident=true,standard=true,boundary_mode=NoMDBC())
    fine = VR.make_resolution_domain(case.fine.geometry,case.fine.constants,case.fine.kernel,base,true,256;
        resident=true,standard=true,boundary_mode=NoMDBC(),attachment_config=settings)
    domains = (coarse,fine)
    limits = VR.AttachedResolutionLimits(base,CuArray(fill(VR.identity_resolution_pose(),2)))
    state = SPHExampleGPU.GPUStepState.StepState{Float64}()
    VR.initialize_resolution_attachment!(fine,limits,state)
    foreach(d -> VR.retire_particles!(d,limits),domains)
    VR.initialize_resident_domains!(domains;state)
    VR.exchange_buffers!(coarse,fine,limits)
    VR.exchange_buffers!(fine,coarse,limits)
    VR.initialize_resolution_numerics!(domains,state,SymplecticTimeStepping();density_diffusion=ZeroDensityDiffusion(),viscosity=ZeroViscosity())
    if grow
        # Both predictor rebuilds must retry without advancing the body twice.
        for d in (coarse.predictor,fine.predictor)
            d.cells.Counts = CUDA.zeros(Int32,8)
            d.cells.CellStart = CUDA.zeros(Int32,9)
        end
    end
    VR.advance_resolution!(domains,limits,state,0.003;batch_steps=8,use_graphs=true,
        scheme=SymplecticTimeStepping(),density_diffusion=ZeroDensityDiffusion(),viscosity=ZeroViscosity())
    snapshots = map(d -> VR.solver_resolution_snapshot(d,case.region,limits;scheme=SymplecticTimeStepping()),domains)
    return (;domains,limits,state,snapshots)
end

@testset "attachment configuration and transformed ownership" begin
    case = attachment_case()
    @test_throws ArgumentError VariableResolutionConfig(case.region;fine_geometry=case.fine.geometry,follow=0)
    @test_throws ArgumentError VariableResolutionConfig(case.region;fine_geometry=case.fine.geometry,follow=true)
    config = VariableResolutionConfig(case.region;fine_geometry=case.fine.geometry,follow=5)
    @test config.follow == 5
    @test occursin("follow=5",sprint(show,config))
    motion = PrescribedMotion(5;translation=SampledTrajectory([0,1],[(0,0),(2,0)]),
        rotation=SampledTrajectory([0,1],[0,pi/2]))
    base = VR.resolution_limits(case.region,0.1,0.05)
    pose = motion_state(motion,0.5)
    limits = VR.AttachedResolutionLimits(base,[pose,VR.identity_resolution_pose()])
    x = motion_position(pose,SVector(0.9,0.9))
    @test VR.resolution_owned(x,limits,true)
    @test !VR.inside_box(x,case.region.lower,case.region.upper)
    @test !VR.outside_domain(x,limits,true)
    @test VR.buffer_particle(motion_position(pose,SVector(1.05,0.0)),limits,true)
    @test VR.outside_domain(motion_position(pose,SVector(1.2,0.0)),limits,true)
    @test VR.resolution_attachment_config(config,case.coarse.geometry,nothing,motion,case.coarse.kernel).marker == 5
    @test_throws ArgumentError VR.resolution_attachment_config(config,case.coarse.geometry,nothing,nothing,case.coarse.kernel)
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "GPU moving attachment: sorting, graphs, rotation and output" begin
        for T in (Float32,Float64), scheme in (SymplecticTimeStepping(),SingleNeighborTimeStepping())
            case = attachment_case(T)
            motion = PrescribedMotion(5;translation=SampledTrajectory([0,1],[(0,0),(3,1)]),
                rotation=SampledTrajectory([0,1],[0,12]))
            direct = run_attachment(case;motion,graphs=false,scheme)
            graph = run_attachment(case;motion,graphs=true,scheme,single=false,grid=true)
            @test graph.result.final_state.complete
            @test graph.result.configuration.effective.variable_resolution.follow == 5
            @test graph.result.diagnostics.captured_graphs > 0
            @test graph.result.diagnostics.step_state_readbacks < graph.meta.Iteration
            for field in (:Position,:Velocity,:Density,:ID,:Type)
                @test getproperty(direct.particles,field) == getproperty(graph.particles,field)
            end
            fine = graph.result.final_state.resolution.fine
            body = findall(==(Moving),fine.Type)
            order = sortperm(fine.ID[body])
            reference = case.fine.geometry[2].Particles.Position
            @test length(body) == length(reference)
            @test count(==(Moving),graph.result.final_state.resolution.coarse.Type) == 0
            @test fine.Position[body[order]] ≈ [motion_position(motion,x,graph.meta.TotalTime) for x in reference] atol=1e-12
            @test fine.Velocity[body[order]] ≈ [T.(motion_velocity(motion,x,graph.meta.TotalTime)) for x in reference]
            @test all(==(1),fine.Owned[body])
            @test graph.result.final_state.resolution_pose.rotation ≈ motion_state(motion,graph.meta.TotalTime).rotation
            @test isfile(joinpath(graph.directory,"Attached.vtkhdf.series"))
        end
    end

    @testset "MotionDetails, legacy rotation and moving mDBC ghosts" begin
        details = MotionDetails{2,Float32}(;Velocity=1,StartTime=-1,Duration=Inf,Direction=SVector(1f0,0f0))
        case = attachment_case(;details,wall_ghosts=true)
        run = run_attachment(case;boundary=SimpleMDBC)
        fine = run.result.final_state.resolution.fine
        body = findall(==(Moving),fine.Type)
        @test all(i -> fine.GhostPoints[i]-fine.Position[i] ≈ SVector(0.02,0.0),body)
        @test run.result.final_state.resolution_pose.offset ≈ SVector(run.meta.TotalTime,0)
        case = attachment_case(;wall_ghosts=true)
        rotation = RotationMotion(5;times=[0,1],angles=[0,2])
        run = run_attachment(case;rotation,boundary=SimpleMDBC)
        fine = run.result.final_state.resolution.fine
        body = findall(==(Moving),fine.Type)
        normal = run.result.final_state.resolution_pose.rotation*SVector(1.0,0.0)
        @test all(i -> fine.GhostNormals[i] ≈ normal,body)
        @test all(i -> fine.GhostPoints[i]-fine.Position[i] ≈ 0.02normal,body)
    end

    @testset "motion hierarchy, pauses, initial time and marker three" begin
        case = attachment_case(Float64)
        for level in (case.coarse,case.fine)
            body = level.geometry[2]
            level.geometry[2] = SPHGeometry{2,Float64}(;Particles=body.Particles,Type=Moving,GroupMarker=3)
        end
        parent = PrescribedMotion(10;group_marker=nothing,
            translation=SampledTrajectory([0,0.006],[(0,0),(0.02,0.01)]),
            rotation=SampledTrajectory([0,0.006],[0.1,0.2]))
        child = PrescribedMotion(11;group_marker=3,parent=10,
            translation=SampledTrajectory([0,0.006],[(0,0),(0.01,0)]))
        motion = MotionSystem((child,parent))
        run = run_attachment(case;motion,follow=3,TotalTime=0.002)
        fine = run.result.final_state.resolution.fine
        body = findall(==(Moving),fine.Type)
        body = body[sortperm(fine.ID[body])]
        reference = case.fine.geometry[2].Particles.Position
        @test fine.Position[body] ≈ [motion_position(motion,11,x,run.meta.TotalTime) for x in reference] atol=1e-12
        @test all(iszero,fine.Velocity[body])
        @test run.result.final_state.resolution_pose.offset ≈ motion_state(motion,11,run.meta.TotalTime).offset
        @test all(==(3),fine.GroupMarker[body])
    end

    @testset "GPU floating attachment: free fall and constrained rotation" begin
        for T in (Float32,Float64)
            case = attachment_case(T;floating=true,gravity=9.81)
            direct = run_attachment(case;graphs=false)
            graph = run_attachment(case)
            f = graph.result.final_state.floating
            t = graph.meta.TotalTime
            @test f.center[1] ≈ SVector(0,-9.81t^2/2) atol=2e-8
            @test f.velocity[1] ≈ SVector(0,-9.81t) atol=2e-7
            @test graph.result.final_state.resolution_pose.offset ≈ f.center[1]
            @test direct.particles.Position == graph.particles.Position
            @test direct.particles.Density == graph.particles.Density
            @test isfile(joinpath(graph.directory,"Attached_Floating.csv"))
            @test length(readlines(joinpath(graph.directory,"Attached_Floating.csv"))) == 4
            h5open(joinpath(graph.directory,"Attached.vtkhdf"),"r") do file
                bodies = file["VTKHDF/FloatingBodies"]
                @test read(bodies["Steps/Values"]) == read(file["VTKHDF/Particles/Steps/Values"])
                @test bodies["PointData/Force"][1:2,end] == f.load_force[1]
                @test bodies["PointData/Torque"][3,end] == f.load_torque[1]
                @test all(==(5),read(bodies["PointData/GroupMarker"]))
            end
            constraints = FloatingConstraints{2,T}(;PrescribedVelocity=(T(1),T(0)),PrescribedOmega=(T(3),))
            case = attachment_case(T;floating=true,details=constraints)
            run = run_attachment(case;scheme=SingleNeighborTimeStepping())
            f = run.result.final_state.floating
            @test f.center[1] ≈ SVector(run.meta.TotalTime,0) atol=2e-8
            @test f.angle[1] ≈ 3run.meta.TotalTime
            fine = run.result.final_state.resolution.fine
            positions = fine.Position[findall(==(Floating),fine.Type)]
            @test all(x -> norm(x-f.center[1]) ≈ sqrt(0.02),positions)
        end
    end

    @testset "attachment validation" begin
        case = attachment_case()
        motion = PrescribedMotion(5)
        @test_throws ArgumentError run_attachment(case;motion,follow=99)
        @test_throws ArgumentError run_attachment(case;motion,scheme=DualSPHysicsSymplectic())
        @test_throws ArgumentError run_attachment(case;motion=PrescribedMotion(6))
        @test_throws ArgumentError run_attachment(case)
        small = merge(case,(;region=VariableResolutionBox((-0.4,-0.4),(0.4,0.4))))
        @test_throws ArgumentError run_attachment(small;motion)
        details = MotionDetails{2,Float32}(;Velocity=1,StartTime=0,Duration=1,Direction=SVector(1f0,0f0),MoveParticles=false)
        @test_throws ArgumentError run_attachment(attachment_case(;details))
    end

    @testset "floating forces agree with the fine uniform solver" begin
        case = attachment_case(Float64;floating=true,wet=true)
        # Asymmetric pressure produces a measurable body acceleration.
        for level in (case.coarse,case.fine)
            fluid = level.geometry[1].Particles
            for i in eachindex(fluid.Density)
                fluid.Density[i] = fluid.Position[i][1] < 0 ? 1010 : 1000
            end
        end
        # Put fluid within support of the body while retaining a dry solid core.
        points = [SVector(-0.15,y) for y in -0.1:0.05:0.1]
        level = case.fine
        fluid = level.geometry[1]
        level.geometry[1] = SPHGeometry{2,Float64}([fluid.Particles.Position;points];
            Density=[fluid.Particles.Density;fill(1010.0,length(points))],Type=Fluid,GroupMarker=2)
        attached = run_attachment(case;duration=0.0001)
        uniform = uniform_floating(case)
        actual,expected = attached.result.final_state.floating,uniform.final_state.floating
        @test norm(expected.velocity[1]) > 1e-5
        @test actual.center ≈ expected.center atol=1e-10 rtol=1e-5
        @test actual.velocity ≈ expected.velocity atol=1e-8 rtol=1e-5
        @test actual.omega ≈ expected.omega atol=1e-8 rtol=1e-5
    end

    @testset "relative flux sweeps stationary fluid without imposing body velocity" begin
        case = attachment_case(Float64;wet=true)
        base = VR.resolution_limits(case.region,0.1,0.05)
        domains = map((case.coarse,case.fine),(false,true)) do level,fine
            d = VR.make_resolution_domain([level.geometry[1]],level.constants,level.kernel,base,fine,4096;boundary_mode=NoMDBC())
            VR.retire_particles!(d,base)
            VR.rebuild_domain!(d)
            d
        end
        frames = CuArray(fill(VR.identity_resolution_pose(),2))
        limits = VR.AttachedResolutionLimits(base,frames)
        motion = PrescribedMotion(5;translation=SampledTrajectory([0,1],[(0,0),(1,0)]))
        for iteration in 1:12
            copyto!(frames,[motion_state(motion,iteration*0.01),motion_state(motion,(iteration-1)*0.01)])
            for d in domains
                VR.retire_particles!(d,limits)
                VR.rebuild_domain!(d)
            end
            VR.transfer_particles!(domains[1],domains[2],limits,0.01)
            VR.transfer_particles!(domains[2],domains[1],limits,0.01)
        end
        for d in domains
            @test Array(d.events)[1] > 0
            @test Array(d.events)[2] > 0
            types,velocity,density = Array(d.particles.Type),Array(d.particles.Velocity),Array(d.particles.Density)
            active = findall(==(Fluid),types)
            @test all(i -> iszero(velocity[i]),active)
            @test all(i -> density[i] ≈ 1000,active)
        end
    end

    @testset "attached body survives a graph rebuild capacity retry" begin
        for floating in (false,true)
            case = attachment_case(Float64;floating,gravity=9.81)
            motion = floating ? nothing : PrescribedMotion(5;translation=SampledTrajectory([0,1],[(0,0),(1,0)]))
            original = resident_attachment(case,motion)
            grown = resident_attachment(case,motion;grow=true)
            @test original.state.ih[3] == grown.state.ih[3]
            @test first(Array(original.limits.frames)) == first(Array(grown.limits.frames))
            for (a,b) in zip(original.snapshots,grown.snapshots)
                @test a.Position[sortperm(a.ID)] == b.Position[sortperm(b.ID)]
                @test a.Velocity[sortperm(a.ID)] == b.Velocity[sortperm(b.ID)]
            end
            @test grown.domains[2].predictor.cells.capacity_growths > original.domains[2].predictor.cells.capacity_growths
        end
    end
end
end
