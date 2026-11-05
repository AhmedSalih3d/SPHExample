module RigidTorsionTests
using Test,SPHExampleGPU,CUDA,StaticArrays,LinearAlgebra
const F = SPHExampleGPU.GPUFloating

@testset "torsional hinge specification" begin
    @test HingeJoint(1;Anchor = (0,0,0),Axis = (0,1,0),Stiffness = 750,Damping = 50).stiffness == 750
    @test_throws ArgumentError HingeJoint(1;Anchor = (0,0,0),Axis = (0,1,0),Damping = -1)
    @test_throws ArgumentError HingeJoint(1;Anchor = (0,0,0),Axis = (0,1,0),Stiffness = Inf)
    @test_throws ArgumentError HingeJoint(1;Anchor = (0,0,0),Axis = (0,1,0),RestAngle = NaN)
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "damped torsional oscillator follows analytical solution" begin
        for D in (2,3),scheme in (SymplecticTimeStepping(),SingleNeighborTimeStepping())
            points = D == 2 ? [SVector(0.05i,0.05j) for i in (-1,1) for j in (-1,1)] :
                [SVector(0.05i,0.05j,0.05k) for i in (-1,1) for j in (-1,1) for k in (-1,1)]
            geometry = [SPHGeometry{D,Float64}(points;Density = 1000,GroupMarker = 1,Type = Floating,
                Floating = FloatingDetails{Float64}(RelativeWeight = 1))]
            constants = SimulationConstants{Float64}(D;dx = 0.1,g = 0,c₀ = 20)
            inertia = constants.m₀*sum(p -> p[1]^2+p[2]^2,points)
            stiffness,damping = 2inertia,0.4inertia
            axis = D == 2 ? (0,1,0) : (0,0,1)
            dynamics = RigidBodyDynamics(Joints = (HingeJoint(1;Anchor = (0,0,0),Axis = axis,
                Stiffness = stiffness,Damping = damping,RestAngle = 0.1),),MaxTimeStep = 0.0005)
            mktempdir() do path
                meta = SimulationMetaData{D,Float64}(SimulationName = "Torsion",SaveLocation = path,SimulationTime = 0.2,
                    OutputTimes = 0.2,VisualizeInParaview = false,OpenLogFile = false,GPUUseGraph = true)
                logger = SimulationLogger(path)
                result = try
                    redirect_stdout(devnull) do
                        RunSimulation(SimGeometry = geometry,SimMetaData = meta,SimConstants = constants,
                            SimParticles = AllocateDataStructures(geometry,meta),SimLogger = logger,
                            SimKernel = SPHKernelInstance{D,Float64}(WendlandC2();h = 0.2),
                            SimViscosity = ZeroViscosity(),SimDensityDiffusion = ZeroDensityDiffusion(),
                            SimTimeStepping = scheme,SimRigidBodyDynamics = dynamics)
                    end
                finally
                    isopen(logger.LoggerIo) && close(logger.LoggerIo)
                end
                state,time = result.final_state.floating,result.final_state.time
                frequency = sqrt(2-0.2^2)
                expected_angle = 0.1*(1-exp(-0.2time)*(cos(frequency*time)+0.2/frequency*sin(frequency*time)))
                expected_speed = 0.2/frequency*exp(-0.2time)*sin(frequency*time)
                angle = D == 2 ? -only(state.angle) : 2atan(state.orientation[1][4],state.orientation[1][1])
                speed = D == 2 ? -only(state.omega) : state.omega[1][3]
                @test angle ≈ expected_angle atol = 2e-7
                @test speed ≈ expected_speed atol = 2e-6
                @test norm(only(state.center)) < 1e-10
                @test maximum(e.translation_error for e in rigid_joint_errors(dynamics,geometry,state)) < 1e-10
            end
        end
    end
    @testset "torsional pair applies equal and opposite torque" begin
        for D in (2,3)
            link = F.DeviceRigidTorsion(Int32(1),Int32(2),SVector(0.,1.,0.),750.,50.,0.)
            orientation = D == 2 ? [0.2,-0.1] : [F.rotation_quaternion(SVector(0.,0.2,0.)),F.rotation_quaternion(SVector(0.,-0.1,0.))]
            omega = D == 2 ? [0.4,-0.2] : [SVector(0.,0.4,0.),SVector(0.,-0.2,0.)]
            turn = D == 2 ? [0.,0.] : zeros(SVector{3,Float64},2)
            force = CUDA.zeros(Float64,D == 2 ? 6 : 12)
            CUDA.@cuda threads=128 F.floating_torsion_kernel!(force,CuArray([link]),CuArray(orientation),CuArray(omega),
                CuArray(turn),CUDA.zeros(Int32,2),SPHExampleGPU.GPUStepState.HostStep(0.001,0.),false,Val(D),Int32(2))
            load = Array(force)
            @test load[1:length(load)÷2] ≈ -load[length(load)÷2+1:end]
            @test norm(load) > 1
            moment = F.torsion_moment(link,orientation,omega,turn,false,Val(D))
            @test moment ≈ (D == 2 ? -255. : SVector(0.,-255.,0.))
        end
    end
end
end
