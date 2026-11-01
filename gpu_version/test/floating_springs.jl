module FloatingSpringTests
using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra
const F = SPHExampleGPU.GPUFloating
const S = SPHExampleGPU.GPUStepState

@testset "linear spring validation and analytic loads" begin
    @test_throws ArgumentError LinearSpring{2,Float32}(;Anchor=(0,0),Attachment=(0,1),Stiffness=-1)
    @test_throws ArgumentError LinearSpring{2,Float32}(;Anchor=(0,0),Attachment=(0,1),Stiffness=1,Damping=Inf)
    @test_throws ArgumentError LinearSpring{2,Float32}(;Anchor=(0,NaN),Attachment=(0,1),Stiffness=1)
    @test_throws ArgumentError LinearSpring{2,Float32}(;Anchor=(0,0),Attachment=(0,1),Stiffness=1,RestLength=-1)
    @test LinearSpring{2,Float64}(;Anchor=(0,0),Attachment=(0,2),Stiffness=1).RestLength == 2
    @test FloatingDetails{Float64}(1.0,0.0,nothing).Springs == ()
    spring = F.DeviceSpring(SVector(0.0,0.0),SVector(1.0,0.0),10.0,2.0,0.5)
    force,torque = F.spring_load(spring,SVector(0.0,0.0),SVector(3.0,0.0),0.0,0.0)
    @test force == SVector(-11.0,0.0)
    @test torque == 0
    offcenter = F.DeviceSpring(SVector(-1.0,1.0),SVector(0.0,1.0),0.0,2.0,1.0)
    force,torque = F.spring_load(offcenter,SVector(0.0,0.0),SVector(0.0,0.0),1.0,0.0)
    @test force == SVector(2.0,0.0)
    @test torque == -2
    collapsed = F.DeviceSpring(SVector(0.0,0.0),SVector(0.0,0.0),10.0,2.0,1.0)
    @test F.spring_load(collapsed,SVector(0.0,0.0),SVector(0.0,0.0),0.0,0.0) == (SVector(0.0,0.0),0.0)
end

function fixture(D,T;k=10,c=2)
    center = SVector{D,Float64}(ntuple(d -> d == D ? 1.0 : 0.0,D))
    points = [center+SVector{D,Float64}(ntuple(d -> ((i >> (d-1)) & 1) == 0 ? -0.1 : 0.1,D)) for i in 0:2^D-1]
    spring = LinearSpring{D,T}(;Anchor=zeros(D),Attachment=center,Stiffness=k,Damping=c,RestLength=0.8)
    constants = SimulationConstants{T}(D;dx=0.1,g=0,c₀=10)
    geometry = [SPHGeometry{D,T}(points;Density=constants.ρ₀,Type=Floating,GroupMarker=5,
        Floating=FloatingDetails{T}(;RelativeWeight=one(T),Springs=(spring,)))]
    particles = AllocateDataStructures(geometry;position_type=Float64)
    f = F.FloatingArrays(geometry,particles,constants;position_type=Float64)
    return (;f,geometry,constants)
end

function run_oscillator(;graphs=true,scheme=SymplecticTimeStepping(),k=20,c=0,duration=0.1)
    case = fixture(2,Float64;k,c)
    directory = mktempdir()
    meta = SimulationMetaData{2,Float64,NoShifting,NoKernelOutput,NoMDBC,NoLog}(;
        SimulationName="SpringOscillator",SaveLocation=directory,SimulationTime=duration,OutputTimes=duration,
        GPUDoublePosition=true,GPUUseGraph=graphs,GPUClipTimeStepToOutput=true,GPUMaxStepsPerSync=16,
        GPUBoundaryForces=false,VisualizeInParaview=false,OpenLogFile=false)
    particles = AllocateDataStructures(case.geometry,meta)
    logger = SimulationLogger(directory)
    result = try
        redirect_stdout(devnull) do
            RunSimulation(;SimGeometry=case.geometry,SimConstants=case.constants,
                SimKernel=SPHKernelInstance{2,Float64}(WendlandC2();h=0.1),SimParticles=particles,
                SimMetaData=meta,SimLogger=logger,SimViscosity=ZeroViscosity(),
                SimDensityDiffusion=ZeroDensityDiffusion(),SimTimeStepping=scheme)
        end
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    return (;result,meta,case)
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "GPU spring force, derivative, pause and freeze D=$D T=$T" for D in (2,3), T in (Float32,Float64)
        case = fixture(D,T)
        f = case.f
        step = S.StepState{T}(;dt=0.001)
        F.launch_floating_derivative!(f,step,zero(T))
        mass = only(Array(f.mass))
        expected = SVector{D,T}(ntuple(d -> d == D ? -2/mass : 0.0,D))
        @test only(Array(f.acceleration)) ≈ expected atol=2e-7
        @test all(iszero,Array(f.force))
        F.launch_floating_update!(f,step,zero(T),false)
        @test only(Array(f.velocity_half)) ≈ expected*T(0.0005) atol=2e-9
        F.launch_floating_update!(f,step,zero(T),true)
        @test all(iszero,Array(f.force))
        records = F.floating_spring_state(f)
        @test length(records) == 1
        @test records[1].attachment ≈ only(Array(f.center))
        @test records[1].force[D] < 0
        fill!(f.frozen,Int32(1))
        F.launch_floating_springs!(f,step,false)
        @test all(iszero,Array(f.force))
        fill!(f.frozen,Int32(0))
        fill!(f.pause,T(1))
        before = F.floating_state(f)
        F.launch_floating_update!(f,step,zero(T),false)
        F.launch_floating_update!(f,step,zero(T),true)
        @test F.floating_state(f).center == before.center
        @test all(iszero,Array(f.force))
        @test isfinite(f.spring_dt) && f.spring_dt > 0
        # A quarter turn places an off-centre attachment above the centre.
        # Its angular velocity creates axial damping and a negative torque.
        center = SVector{D,Float64}(ntuple(d -> d == D ? 1.0 : 0.0,D))
        anchor = center+SVector{D,Float64}(ntuple(d -> d == 1 ? -1.0 : d == 2 ? 1.0 : 0.0,D))
        offset = SVector{D,Float64}(ntuple(d -> d == 1 ? 1.0 : 0.0,D))
        copyto!(f.center,[center])
        fill!(f.velocity,zero(eltype(f.velocity)))
        copyto!(f.springs,[F.DeviceSpring(anchor,offset,zero(T),T(2),one(T))])
        if D == 2
            fill!(f.angle,T(pi/2))
            fill!(f.omega,one(T))
        else
            copyto!(f.orientation,[SVector{4,T}(cos(pi/4),0,0,sin(pi/4))])
            copyto!(f.omega,[SVector{3,T}(0,0,1)])
        end
        F.launch_floating_springs!(f,step,false)
        load = Array(f.force)
        @test load[1] ≈ 2 atol=2e-6
        @test load[end] ≈ -2 atol=2e-6
    end

    @testset "normal solver spring oscillator and CUDA graphs" begin
        for scheme in (SymplecticTimeStepping(),SingleNeighborTimeStepping())
            direct = run_oscillator(;graphs=false,scheme)
            graph = run_oscillator(;scheme)
            mass = only(Array(graph.case.f.mass))
            omega = sqrt(20/mass)
            t = graph.meta.TotalTime
            state = graph.result.final_state.floating
            @test state.center[1][2] ≈ 0.8+0.2cos(omega*t) atol=2e-7
            @test state.velocity[1][2] ≈ -0.2omega*sin(omega*t) atol=2e-7
            @test state == direct.result.final_state.floating
        end
        stiff = run_oscillator(;k=1e9,c=1e7,duration=0.00002)
        @test stiff.meta.CurrentTimeStep <= stiff.case.f.spring_dt
        @test all(isfinite,stiff.result.final_state.floating.velocity[1])
    end
end
end
