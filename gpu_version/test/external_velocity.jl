module ExternalVelocityTests
using Test,SPHExampleGPU,CSV,CUDA,LinearAlgebra
include(joinpath(@__DIR__,"..","benchmark","chrono_case.jl"))
@testset "external-velocity reference fixtures" begin
    expected = Dict(:boxes => (17004,2),:breaking_wall => (1040455,28),:many_blocks => (1002272,113),:hauling => (4541,2),:ship => (12791,2))
    for (variant,(particles,bodies)) in expected
        case = prepare_chrono_case("ExternalVelocity.jl";variant)
        @test sum(length(g.Particles) for g in case.geometry) == particles
        @test count(g -> g.Type == Floating,case.geometry) == bodies
        @test case.meta.Periodicity === nothing
        @test case.dynamics.max_timestep === (variant == :ship ? nothing : .001)
        @test length(case.dynamics.velocities) == (variant == :boxes ? 2 : 1)
        @test length(case.dynamics.springs) == (variant == :hauling ? 1 : 0)
        @test case.scheme isa SymplecticTimeStepping
        if variant == :boxes
            a,b = case.dynamics.velocities
            @test a.linear.times == b.linear.times
            @test a.linear.values == b.linear.values
            @test a.angular.times == b.angular.times
            @test a.angular.values == b.angular.values
        elseif variant == :ship
            @test case.dynamics.contacts === nothing
            @test only(case.dynamics.joints).body_a == 23
        end
    end
end
end
