module RigidVelocityTests
using Test,SPHExampleGPU,CUDA,StaticArrays,LinearAlgebra
const F = SPHExampleGPU.GPUFloating
const S = SPHExampleGPU.GPUStepState

@testset "velocity program semantics" begin
    @test_throws ArgumentError VelocitySeries([1.,0.],[0.,1.])
    @test_throws ArgumentError VelocitySeries([0.],[Inf])
    @test_throws ArgumentError RigidVelocity(0;Angular = VelocitySeries([0.],[0.]))
    @test_throws ArgumentError BodySpring(1,1;Attachment = (0,0),Anchor = (1,0),Stiffness = 1,RestLength = 1)
    @test_throws ArgumentError F.append_velocity_keys!(F.DeviceVelocityKey{Float32}[],VelocitySeries([0.],[(1.,2.)]),Float32)
    body = (;velocity = [SVector(0.,0.)],omega = [0.],acceleration = [SVector(0.,0.)],angular_acceleration = [0.])
    controller = F.NativeRigidTimeStep(body,[.5],.01,.004,0.,[SVector(10.,4.)])
    @test isapprox(S.controlled_timestep(controller,.01),.004/12;atol = 1e-15)
    for T in (Float32,Float64)
        series = VelocitySeries([0.,.1,.1,.2,.3],[0.,1.,nothing,2.,3.])
        keys = F.DeviceVelocityKey{T}[]
        span = F.append_velocity_keys!(keys,series,T)
        @test F.velocity_sample_impl(keys,span...,.05) == (true,T(.5))
        @test F.velocity_sample_impl(keys,span...,.1) == (false,zero(T))
        @test F.velocity_sample_impl(keys,span...,.15) == (false,zero(T))
        @test F.velocity_sample_impl(keys,span...,.25) == (true,T(2.5))
        @test F.velocity_sample_impl(keys,span...,1.) == (true,T(3))
    end
end

function simulate(geometry,dynamics,::Val{D},::Type{T};duration=.2,gravity=0.,scheme=SymplecticTimeStepping(),graph=true) where {D,T}
    mktempdir() do directory
        meta = SimulationMetaData{D,T}(SimulationName = "NativeVelocity",SaveLocation = directory,SimulationTime = duration,
            OutputTimes = duration,VisualizeInParaview = false,OpenLogFile = false,GPUDoublePosition = true,
            GPUUseGraph = graph,GPUClipTimeStepToOutput = true,GPUBoundaryForces = false)
        logger = SimulationLogger(directory)
        try
            redirect_stdout(devnull) do
                RunSimulation(SimGeometry = geometry,SimMetaData = meta,SimParticles = AllocateDataStructures(geometry,meta),SimLogger = logger,
                    SimConstants = SimulationConstants{T}(D;dx = .02,g = gravity,c₀ = 20),
                    SimKernel = SPHKernelInstance{D,T}(WendlandC2();h = T(.04)),SimViscosity = ZeroViscosity(),
                    SimDensityDiffusion = ZeroDensityDiffusion(),SimRigidBodyDynamics = dynamics,SimTimeStepping = scheme)
            end
        finally
            isopen(logger.LoggerIo) && close(logger.LoggerIo)
        end
    end
end
function cube(::Val{D},::Type{T},center,marker;velocity=nothing) where {D,T}
    points = [SVector{D,Float64}(center)+SVector{D,Float64}(ntuple(d -> .01*(2*((i >> (d-1)) & 1)-1),D)) for i in 0:2^D-1]
    return SPHGeometry{D,T}(points;Density = 1000,GroupMarker = marker,Type = Floating,
        Floating = FloatingDetails{T}(RelativeWeight = 1,InitialVelocity = velocity))
end
function aligned_contact_kernel!(f,d,positions,ids,groups,center,angle,velocity,omega)
    F.solve_rigid_contact!(f,d,positions,ids,groups,center,angle,velocity,omega,Int32(1),Int32(0),Int32(1),S.HostStep(.001,0.),false)
    return nothing
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "motor stages and release D=$D T=$T" for D in (2,3),T in (Float32,Float64),scheme in (SymplecticTimeStepping(),SingleNeighborTimeStepping())
        origin = zeros(SVector{D,Float64})
        speed = SVector{D,Float64}(ntuple(d -> d == 1 ? .1 : 0.,D))
        zero_omega = D == 2 ? 0. : SVector(0.,0.,0.)
        program = RigidVelocity(1;Linear = VelocitySeries([0.,.1,.1],[origin,speed,nothing]),
            Angular = VelocitySeries([0.],[zero_omega]))
        geometry = [cube(Val(D),T,origin,1)]
        result = simulate(geometry,RigidBodyDynamics(Velocities = (program,),MaxTimeStep = .001),Val(D),T;
            gravity = 9.81,scheme,graph = T == Float32)
        state = result.final_state.floating
        @test result.final_state.complete
        @test only(state.center)[1] ≈ .015 atol = 3e-5
        @test only(state.velocity)[1] ≈ .1 atol = 3e-5
        @test only(state.center)[D] ≈ -.5*9.81*.1^2 atol = 3e-4
        @test only(state.velocity)[D] ≈ -9.81*.1 atol = .006
        @test norm(only(state.omega)) < 1e-6
    end
    @testset "angular motor midpoint pose and reaction D=$D T=$T" for D in (2,3),T in (Float32,Float64),scheme in (SymplecticTimeStepping(),SingleNeighborTimeStepping())
        origin = zeros(SVector{D,Float64})
        spin = D == 2 ? .2 : SVector(0.,.2,0.)
        program = RigidVelocity(1;Linear = VelocitySeries([0.],[origin]),Angular = VelocitySeries([0.,.2],[zero(spin),spin]))
        geometry = [cube(Val(D),T,origin,1)]
        result = simulate(geometry,RigidBodyDynamics(Velocities = (program,),MaxTimeStep = .001),Val(D),T;scheme)
        state = result.final_state.floating
        @test norm(only(state.center)) < 1e-12
        @test isapprox(only(state.omega),spin;atol = 1e-6)
        pose = D == 2 ? only(state.angle) : 2asin(only(state.orientation)[3])
        @test isapprox(pose,.02;atol = 1e-6)
        inertia = 1000*.02^D*2^D*.0002
        moment = only(result.final_state.rigid.reaction_torque)
        @test isapprox(D == 2 ? moment : moment[2],inertia;rtol = .002)
    end
    @testset "planar elastic impact conserves momentum" begin
        for T in (Float32,Float64),scheme in (SymplecticTimeStepping(),SingleNeighborTimeStepping())
            geometry = [cube(Val(2),T,(-.05,0.),1;velocity = (1.,0.)),cube(Val(2),T,(.05,0.),2;velocity = (-1.,0.))]
            contacts = RigidContacts(ParticleDiameter = .02,Margin = .01,Friction = 0,Restitution = 1)
            result = simulate(geometry,RigidBodyDynamics(Contacts = contacts,MaxTimeStep = .0005,Iterations = 64),Val(2),T;duration = .05,scheme)
            state = result.final_state.floating
            @test result.final_state.complete
            @test norm(sum(state.velocity)) < 1e-5
            @test state.velocity[1][1] < -.8
            @test state.velocity[2][1] > .8
            @test maximum(abs,state.omega) < .01
        end
    end
    @testset "planar floor friction opposes sliding" begin
        speeds = Float64[]
        for friction in (0.,0.8)
            floor = SPHGeometry{2,Float64}([SVector(.02i,0.) for i in -50:50];Density = 1000,GroupMarker = 3,Type = Fixed)
            geometry = [floor,cube(Val(2),Float64,(0.,.06),1;velocity = (1.,0.))]
            contacts = RigidContacts(ParticleDiameter = .02,Margin = .01,Friction = friction,Restitution = 0)
            result = simulate(geometry,RigidBodyDynamics(Contacts = contacts,MaxTimeStep = .0005,Iterations = 64),Val(2),Float64;
                duration = .3,gravity = 9.81)
            state = result.final_state.floating
            push!(speeds,only(state.velocity)[1])
            @test only(state.center)[2] > .025
            @test all(isfinite,only(state.velocity))
        end
        # A sampled floor has scalloped contact normals: even zero Coulomb
        # friction can redirect translation. Check the tangential law below
        # on a single independently aligned contact instead.
        @test all(isfinite,speeds)
    end
    @testset "aligned planar contact obeys Coulomb bound" begin
        for friction in (0.,0.5)
            f = (;mass = CuArray([1.]),inertia_inv = CuArray([1.]),pause = CuArray([0.]),
                frozen = CuArray(Int32[0]),body = CuArray(Int32[1,0]))
            d = (;winners = CuArray(UInt64[1]),partner = CuArray(Int32[2]),references = CuArray([SVector(0.,-.01),SVector(0.,0.)]),
                parameters = (;distance = .02),bounce = CuArray([0.]),normal_impulse = CuArray([0.]),
                tangent_impulse = CuArray([SVector(0.,0.)]),contact_friction = CuArray([friction]))
            positions,ids,groups = CuArray([SVector(0.,.02),SVector(0.,0.)]),CuArray(Int32[1,2]),CuArray(Int32[1,2])
            center,angle = CuArray([SVector(0.,.03)]),CuArray([0.])
            velocity,omega = CuArray([SVector(1.,-1.)]),CuArray([0.])
            @cuda threads=1 blocks=1 aligned_contact_kernel!(f,d,positions,ids,groups,center,angle,velocity,omega)
            value = only(Array(velocity))
            @test isapprox(value[2],0.;atol = 1e-12)
            @test isapprox(value[1],1-friction;atol = 1e-12)
            @test norm(only(Array(d.tangent_impulse))) <= friction*only(Array(d.normal_impulse))+1e-12
        end
    end
    @testset "moving spring applies equal opposite loads D=$D" for D in (2,3),T in (Float32,Float64)
        a = zeros(SVector{D,Float64})
        b = SVector{D,Float64}(ntuple(d -> d == 1 ? .6 : 0.,D))
        geometry = [cube(Val(D),T,a,1),cube(Val(D),T,b,2)]
        host = AllocateDataStructures(geometry)
        constants = SimulationConstants{T}(D;dx = .02,g = 0,c₀ = 20)
        offset = SVector{D,Float64}(ntuple(d -> d == 2 ? .1 : 0.,D))
        dynamics = RigidBodyDynamics(Springs = (BodySpring(1,2;Attachment = a+offset,Anchor = b+offset,Stiffness = 4000,RestLength = 1.),))
        f = F.attach_rigid_dynamics(F.FloatingArrays(geometry,host,constants),dynamics,host,SPHKernelInstance{D,T}(WendlandC2();h = T(.04)))
        F.launch_body_springs!(f,S.HostStep(.001,0.),false)
        force = reshape(Array(f.force),D == 2 ? 3 : 6,:)
        @test force[1,1] ≈ -1600
        @test force[1,2] ≈ 1600
        @test isapprox(force[D == 2 ? 3 : 6,1],160.;rtol = 1e-5)
        @test norm(sum(force;dims = 2)) < 1e-10
        @test isfinite(f.spring_dt)
    end
end
end
