module DamBreakCubesTests
using Test,SPHExampleGPU,CUDA,LinearAlgebra,HDF5
include(joinpath(@__DIR__,"..","benchmark","chrono_case.jl"))

@testset "DamBreakCubes reference variants" begin
    for (variant,particle_count,bodies,fluid) in ((:one_cube,209630,1,140800),(:three_cubes,214024,3,140800),(:blocks,115436,18,79200))
        case = prepare_chrono_case("DamBreakCubes3d.jl";variant)
        @test sum(length(g.Particles) for g in case.geometry) == particle_count
        @test count(g -> g.Type == Floating,case.geometry) == bodies
        @test length(only(filter(g -> g.Type == Fluid,case.geometry)).Particles) == fluid
        @test case.dynamics.contacts.fixed_groups == (10,)
        @test case.meta.Periodicity === nothing
        @test all(g -> (g.Floating.InteractionMass !== nothing) == (variant == :blocks),filter(g -> g.Type == Floating,case.geometry))
    end
end
if CUDA.functional()
    @testset "DamBreakCubes fluid and native contacts" begin
        for variant in (:one_cube,:three_cubes,:blocks)
            mktempdir() do directory
                result,case = redirect_stdout(devnull) do
                    run_chrono_case("DamBreakCubes3d.jl";variant,simulation_time = 0.02,save_location = directory)
                end
                @test result.final_state.complete
                @test result.diagnostics.frozen_particles == 0
                @test all(v -> all(isfinite,v),result.final_state.floating.center)
                @test all(v -> all(isfinite,v),result.final_state.floating.velocity)
                vtk = Base.invokelatest(case.save_surfaces,directory,case.geometry)
                h5open(vtk,"r") do io
                    @test haskey(io,"VTKHDF/RigidSurfaces")
                end
            end
        end
    end
end
end
