module PlanarRigidBodyTests
using Test,SPHExampleGPU,CUDA,StaticArrays,LinearAlgebra,HDF5,CSV

square(center;spacing = 0.05) = [SVector{2,Float64}(center)+spacing*SVector(i,j)/2 for i in (-1,1) for j in (-1,1)]

function simulate(geometry,dynamics,path;T = Float64,duration = 0.12,gravity = 9.81,
        scheme = SymplecticTimeStepping(),graph = false,async = false)
    mkpath(path)
    constants = SimulationConstants{T}(2;dx = 0.05,g = gravity,c₀ = 20)
    meta = SimulationMetaData{2,T}(SimulationName = "PlanarRigid",SaveLocation = path,SimulationTime = duration,
        OutputTimes = duration,VisualizeInParaview = false,OpenLogFile = false,GPUDoublePosition = true,
        GPUUseGraph = graph,GPUAsyncOutput = async,GPUClipTimeStepToOutput = true,GPUBoundaryForces = false)
    particles = AllocateDataStructures(geometry,meta)
    logger = SimulationLogger(path)
    try
        return RunSimulation(SimGeometry = geometry,SimMetaData = meta,SimConstants = constants,SimParticles = particles,
            SimLogger = logger,SimKernel = SPHKernelInstance{2,T}(WendlandC2();h = T(0.1)),
            SimViscosity = ZeroViscosity(),SimDensityDiffusion = ZeroDensityDiffusion(),
            SimTimeStepping = scheme,SimRigidBodyDynamics = dynamics)
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end

function pendulum_reference(time)
    theta,omega,t = 0.,0.,0.
    acceleration(theta) = -9.81*0.2*cos(theta)/(0.2^2+0.05^2/2)
    while t < time
        dt = min(0.00001,time-t)
        k1 = SVector(omega,acceleration(theta))
        k2 = SVector(omega+dt*k1[2]/2,acceleration(theta+dt*k1[1]/2))
        k3 = SVector(omega+dt*k2[2]/2,acceleration(theta+dt*k2[1]/2))
        k4 = SVector(omega+dt*k3[2],acceleration(theta+dt*k3[1]))
        theta,omega = SVector(theta,omega)+dt*(k1+2k2+2k3+k4)/6
        t += dt
    end
    return SVector(0.2cos(theta),1+0.2sin(theta)),theta,omega
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "planar hinge follows independent rigid pendulum" begin
        target,theta,omega = pendulum_reference(0.12)
        for T in (Float32,Float64),scheme in (SymplecticTimeStepping(),SingleNeighborTimeStepping()),graph in (false,true)
            geometry = [SPHGeometry{2,T}(square((0.2,1.));Density = 1000,GroupMarker = 1,Type = Floating,
                Floating = FloatingDetails{T}(RelativeWeight = 1))]
            dynamics = RigidBodyDynamics(Joints = (HingeJoint(1;Anchor = (0,0,1),Axis = (0,1,0)),),MaxTimeStep = 0.001)
            mktempdir() do path
                result = redirect_stdout(devnull) do
                    simulate(geometry,dynamics,path;T,scheme,graph)
                end
                state = result.final_state.floating
                @test result.final_state.complete
                @test only(state.center) ≈ target atol = 4e-5
                @test only(state.angle) ≈ theta atol = 3e-4
                @test only(state.omega) ≈ omega atol = 0.003
                @test only(rigid_joint_errors(dynamics,geometry,state)).translation_error < 1e-7
            end
        end
    end

    @testset "planar line slide and reaction output" begin
        for T in (Float32,Float64),scheme in (SymplecticTimeStepping(),SingleNeighborTimeStepping())
            geometry = [SPHGeometry{2,T}(square((0.,1.));Density = 1000,GroupMarker = 1,Type = Floating,
                Floating = FloatingDetails{T}(RelativeWeight = 1))]
            dynamics = RigidBodyDynamics(Joints = (PointLineJoint(1;Point = (0,0,1),Direction = (1,0,-1)),),MaxTimeStep = 0.001)
            mktempdir() do path
                result = redirect_stdout(devnull) do
                    simulate(geometry,dynamics,path;T,scheme,graph = true,async = true)
                end
                state,time = result.final_state.floating,result.final_state.time
                @test only(state.center) ≈ SVector(0.,1.)+SVector(1.,-1.)*(9.81time^2/4) atol = 2e-6
                @test only(state.velocity) ≈ SVector(1.,-1.)*(9.81time/2) atol = 2e-5
                @test only(state.omega) ≈ 0 atol = 1e-6
                @test only(result.final_state.rigid.reaction_force) ≈ 10*SVector(9.81/2,9.81/2) rtol = 0.001
                rows = collect(CSV.File(joinpath(path,"PlanarRigid_Floating.csv")))
                @test hasproperty(last(rows),Symbol("ConstraintForce:1"))
                h5open(joinpath(path,"PlanarRigid.vtkhdf"),"r") do io
                    data = io["VTKHDF/FloatingBodies/PointData"]
                    @test size(read(data["ConstraintForce"]),1) == 3
                    @test all(iszero,read(data["ConstraintForce"])[3,:])
                    @test read(data["TotalForce"]) ≈ read(data["Force"])+read(data["ConstraintForce"])+read(data["GravityForce"])
                end
            end
        end
    end

    @testset "planar native bodies preserve SPH coupling and reject out-of-plane joints" begin
        T = Float64
        geometry = [SPHGeometry{2,T}(square((0.,0.3));Density = 1001,GroupMarker = 1,Type = Fluid),
            SPHGeometry{2,T}(square((0.,0.15));Density = 1000,GroupMarker = 2,Type = Floating,
                Floating = FloatingDetails{T}(RelativeWeight = 1))]
        mktempdir() do path
            a,b = redirect_stdout(devnull) do
                (simulate(geometry,nothing,joinpath(path,"original");duration = 0.004,gravity = 0),
                    simulate(geometry,RigidBodyDynamics(),joinpath(path,"native");duration = 0.004,gravity = 0))
            end
            @test norm(only(b.final_state.floating.velocity)) > 1e-6
            @test only(a.final_state.floating.velocity) ≈ only(b.final_state.floating.velocity) atol = 1e-8
            @test only(a.final_state.floating.center) ≈ only(b.final_state.floating.center) atol = 1e-8
            for dynamics in (RigidBodyDynamics(Contacts = RigidContacts(ParticleDiameter = 0.05)),
                    RigidBodyDynamics(Joints = (BallJoint(2;Anchor = (0,1,0)),)),
                    RigidBodyDynamics(Joints = (HingeJoint(2;Anchor = (0,0,0.15),Axis = (1,0,0)),)))
                @test_throws ArgumentError redirect_stdout(devnull) do
                    simulate(geometry,dynamics,joinpath(path,"invalid");duration = 0.004,gravity = 0)
                end
            end
        end
    end

    @testset "planar linked bodies retain prescribed initial translation" begin
        geometry = [SPHGeometry{2,Float64}(square((x,1.));Density = 1000,GroupMarker = b,Type = Floating,
            Floating = FloatingDetails{Float64}(RelativeWeight = 1,InitialVelocity = (0.2,-0.1))) for (b,x) in enumerate((-0.1,0.1))]
        dynamics = RigidBodyDynamics(Joints = (BallJoint(1,2;Anchor = (0,0,1)),),MaxTimeStep = 0.001)
        mktempdir() do path
            result = redirect_stdout(devnull) do
                simulate(geometry,dynamics,path;duration = 0.04,gravity = 0,graph = true)
            end
            for (b,x) in enumerate((-0.1,0.1))
                @test result.final_state.floating.center[b] ≈ SVector(x,1.)+0.04SVector(0.2,-0.1) atol = 1e-10
                @test result.final_state.floating.velocity[b] ≈ SVector(0.2,-0.1) atol = 1e-10
                @test abs(result.final_state.floating.omega[b]) < 1e-10
            end
        end
    end

    @testset "frozen planar pose ignores a previous turn" begin
        geometry = [SPHGeometry{2,Float64}(square((0.,1.));Density = 1000,GroupMarker = 1,Type = Floating,
            Floating = FloatingDetails{Float64}(RelativeWeight = 1))]
        constants = SimulationConstants{Float64}(2;dx = 0.05)
        meta = SimulationMetaData{2,Float64}(SimulationName = "FrozenPlanar",SaveLocation = tempdir(),
            VisualizeInParaview = false,OpenLogFile = false)
        particles = AllocateDataStructures(geometry,meta)
        F = SPHExampleGPU.GPUFloating
        kernel = SPHKernelInstance{2,Float64}(WendlandC2();h = 0.1)
        f = F.attach_rigid_dynamics(F.FloatingArrays(geometry,particles,constants),RigidBodyDynamics(),particles,kernel)
        f.angle .= 0.3
        f.turn .= 0.2
        f.frozen .= 1
        F.freeze_body_state!(f)
        step = SPHExampleGPU.GPUStepState.HostStep(0.01,0.)
        F.prepare_rigid_stage!(f,step)
        F.launch_floating_update!(f,step,9.81,false)
        F.solve_rigid_stage!(f,nothing,nothing,kernel,step,false)
        @test only(Array(f.mechanics.angle)) == 0.3
        @test only(Array(f.turn)) == 0
    end
end
end # module
