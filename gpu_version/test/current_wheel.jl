module CurrentWheelTests
using Test,SPHExampleGPU,CUDA,StaticArrays,LinearAlgebra,HDF5
include(joinpath(@__DIR__,"..","benchmark","chrono_case.jl"))
const F = SPHExampleGPU.GPUFloating
const S = SPHExampleGPU.GPUStepState

@testset "current-wheel reference and open boundaries" begin
    for variant in (:wheel,:pulley)
        case = prepare_chrono_case("CurrentWheelPulley2d.jl";variant,simulation_time = 0.03)
        @test sum(length(g.Particles) for g in case.geometry if g.Type != Frozen) == (variant == :wheel ? 13397 : 13574)
        @test count(g -> g.Type == Floating,case.geometry) == (variant == :wheel ? 1 : 2)
        @test case.open_boundaries.velocity == 1
        @test case.meta.Periodicity === nothing
        @test case.dynamics.contacts === nothing
        @test length(case.dynamics.joints) == (variant == :wheel ? 1 : 3)
    end
    @test_throws ArgumentError PulleyJoint(1,2;Ratio = 0)
end

if CUDA.functional()
    @testset "pulley projects the analytical angular ratio" begin
        for T in (Float32,Float64)
            geometry = [SPHGeometry{2,T}([SVector(x+0.025i,1+0.025j) for i in (-1,1) for j in (-1,1)];
                Density = 1000,GroupMarker = b,Type = Floating,Floating = FloatingDetails{T}(RelativeWeight = 1)) for (b,x) in enumerate((0.,1.))]
            constants = SimulationConstants{T}(2;dx = 0.05)
            host = AllocateDataStructures(geometry)
            kernel = SPHKernelInstance{2,T}(WendlandC2();h = T(0.1))
            dynamics = RigidBodyDynamics(Joints = (HingeJoint(1;Anchor = (0,0,1),Axis = (0,1,0)),
                HingeJoint(2;Anchor = (1,0,1),Axis = (0,1,0)),PulleyJoint(1,2;Ratio = 6)))
            f = F.attach_rigid_dynamics(F.FloatingArrays(geometry,host,constants),dynamics,host,kernel)
            copyto!(f.omega,T[1,0])
            F.initialize_rigid_velocities!(f,0.)
            speed = Array(f.omega)
            @test speed ≈ T[1/37,6/37] atol = 1e-6
            @test abs(speed[2]-6speed[1]) < 1e-6
            copyto!(f.angle,T[0.1,0])
            step = S.HostStep(T(0.001),0.)
            F.prepare_rigid_stage!(f,step)
            F.solve_rigid_stage!(f,nothing,nothing,kernel,step,true)
            @test maximum(e.rotation_error for e in rigid_joint_errors(dynamics,geometry,F.floating_state(f))) < 1e-6
        end
    end

    @testset "current wheel and pulley emit and retire fluid" begin
        for variant in (:wheel,:pulley)
            mktempdir() do directory
                result,case = redirect_stdout(devnull) do
                    run_chrono_case("CurrentWheelPulley2d.jl";variant,simulation_time = 0.1,save_location = directory,
                        cache_interaction_fields = variant == :pulley)
                end
                state = result.final_state.open_boundary
                @test result.final_state.complete
                @test state.emitted_particles == 507
                @test result.performance.steps < 4000
                @test maximum(result.final_state.particles.Density) < 1100
                @test state.active_fluid == state.initial_active_fluid+state.emitted_particles-state.retired_particles
                @test all(v -> all(isfinite,v),result.final_state.floating.velocity)
                @test maximum(e.translation_error for e in rigid_joint_errors(case.dynamics,case.geometry,result.final_state.floating)) < 1e-5
                if variant == :pulley
                    angles = result.final_state.floating.angle
                    @test abs(angles[2]-6angles[1]) < 1e-5
                end
                file = Base.invokelatest(case.save_surfaces,directory,case.geometry)
                h5open(file,"r") do io
                    @test haskey(io,"VTKHDF/ShaftPointer") == (variant == :pulley)
                end
            end
        end
    end
end
end
