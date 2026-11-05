module WaterMillTests
using Test,SPHExampleGPU,CUDA,LinearAlgebra,HDF5
include(joinpath(@__DIR__,"..","benchmark","chrono_case.jl"))

@testset "WaterMill reference geometry and damping" begin
    case = prepare_chrono_case("WaterMill3d.jl")
    @test length.(getproperty.(case.geometry,:Particles)) == [178050,40481,41105,41105,41105,41105,126735,407637]
    @test sum(length(g.Particles) for g in case.geometry) == 917323
    @test getproperty.(case.dynamics.joints,:damping) == (500.,100.,100.,100.,100.)
    @test all(j -> iszero(j.stiffness),case.dynamics.joints)
    @test all(g -> g.Floating.RelativeWeight == Float32(0.04),filter(g -> g.Type == Floating,case.geometry))
    @test case.dynamics.contacts === nothing
    @test case.meta.Periodicity === nothing
end
if CUDA.functional()
    @testset "WaterMill articulated fluid coupling and CAD output" begin
        mktempdir() do directory
            result,case = redirect_stdout(devnull) do
                run_chrono_case("WaterMill3d.jl";simulation_time = 0.02,save_location = directory)
            end
            @test result.final_state.complete
            @test result.diagnostics.frozen_particles == 0
            @test maximum(e.translation_error for e in rigid_joint_errors(case.dynamics,case.geometry,result.final_state.floating)) < 1e-5
            @test all(v -> all(isfinite,v),result.final_state.floating.velocity)
            file = Base.invokelatest(case.save_surfaces,directory,case.geometry)
            h5open(file,"r") do io
                @test haskey(io,"VTKHDF/RigidSurfaces")
            end
        end
    end
end
end
