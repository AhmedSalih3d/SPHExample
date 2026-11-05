module FlexibleGateTests
using Test,SPHExampleGPU,CUDA,SHA,LinearAlgebra,HDF5
include(joinpath(@__DIR__,"..","benchmark","chrono_case.jl"))

@testset "FlexibleGate reference geometry and elastic links" begin
    case = prepare_chrono_case("FlexibleGate2d.jl")
    @test length.(getproperty.(case.geometry,:Particles)) == [419,60,80,80,80,84,3686]
    @test all(j -> j.stiffness == 750 && j.damping == 50,case.dynamics.joints)
    @test last(case.dynamics.joints).body_b == 0
    @test case.meta.Periodicity === nothing
    @test case.dynamics.contacts === nothing
    @test_throws ArgumentError prepare_chrono_case("FlexibleGate2d.jl";dx = 0.02)
end
if CUDA.functional()
    @testset "FlexibleGate coupled elastic output" begin
        mktempdir() do directory
            result,case = redirect_stdout(devnull) do
                run_chrono_case("FlexibleGate2d.jl";simulation_time = 0.03,save_location = directory)
            end
            state = result.final_state.floating
            @test result.final_state.complete
            @test result.diagnostics.frozen_particles == 0
            @test state.center[1][1] > 0.515
            @test maximum(e.translation_error for e in rigid_joint_errors(case.dynamics,case.geometry,state)) < 5e-6
            file = Base.invokelatest(case.save_surfaces,directory,case.geometry)
            h5open(file,"r") do io
                @test haskey(io,"VTKHDF/RigidSurfaces")
                @test read(io["VTKHDF/RigidSurfaces/Steps/Values"]) == [0.,0.01,0.02,0.03]
            end
        end
    end
end
end
