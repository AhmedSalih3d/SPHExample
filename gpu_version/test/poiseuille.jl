module PoiseuilleTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra

include(joinpath(@__DIR__, "..", "benchmark", "poiseuille_case.jl"))

unit_velocity(y, t = Inf; kwargs...) = poiseuille_velocity(y, t; height = 1, lower = 0, acceleration = 0.1, viscosity = 0.1, kwargs...)

@testset "Poiseuille analytical solution" begin
    @test unit_velocity(0.5) ≈ 0.125
    @test unit_velocity(0.25) ≈ 0.09375
    for t in (0, 0.01, 0.5, 5, Inf)
        @test iszero(unit_velocity(0, t))
        @test iszero(unit_velocity(1, t))
        @test unit_velocity(0.2, t) ≈ unit_velocity(0.8, t)
    end
    @test iszero(unit_velocity(0.5, 0))
    # Before wall diffusion reaches the centre, acceleration is uniform.
    @test unit_velocity(0.5, 0.001) ≈ 0.0001 rtol = 1e-9
    @test 0 < unit_velocity(0.5, 0.1) < unit_velocity(0.5, 1) < unit_velocity(0.5, 5) < 0.125
    @test unit_velocity(0.5, 50) ≈ 0.125
    @test unit_velocity(0.4, 0.2) ≈ unit_velocity(0.4, 0.2; terms = 256)
    for (y, t) in ((-0.1, 1), (1.1, 1), (NaN, 1), (0.5, -1), (0.5, NaN))
        @test_throws ArgumentError unit_velocity(y, t)
    end
end

@testset "Poiseuille reference geometry and periodic seam" begin
    for T in (Float32, Float64)
        case = poiseuille_2d_geometry(; FloatType = T)
        @test length.(getproperty.(case.geometry, :Particles)) == [500, 12250]
        @test getproperty.(case.geometry, :Type) == [Fixed, Fluid]
        positions = vcat(getproperty.(getproperty.(case.geometry, :Particles), :Position)...)
        @test length(unique(positions)) == 12750
        @test collect(extrema(first, positions)) ≈ [1e-5, 0.00499]
        @test collect(extrema(last, positions)) ≈ [-0.0005, 0.0005]
        @test case.periodicity.axis == 1
        @test case.periodicity.lower == 0
        @test case.periodicity.upper == 0.005
        @test case.periodicity.translation ≈ SVector(-0.005, 0)
        @test SPHExampleGPU.PeriodicBoundaries.wrap_periodic_position(SVector(0.00501, 0), case.periodicity) ≈ SVector(1e-5, 0)
        @test case.constants.g == 0
        @test case.constants.ν₀ == T(1e-6)
        @test case.constants.m₀ ≈ T(4e-7)
        @test case.constants.c₀ == 1
        @test case.constants.CFL == T(0.2)
        @test case.kernel.h ≈ sqrt(2) * 0.00002
        @test case.viscosity isa LaminarSPS
        @test case.meta.Periodicity == case.periodicity
        @test case.meta.SimulationTime == 0.5
        @test case.meta.OutputTimes == 0.002
        water = case.geometry[2].Particles.Position
        @test length(unique(first, water)) == 250
        @test length(unique(last, water)) == 49
        @test all(p -> abs(p[2]) < 0.0005, water)
        for t in (0, 0.25, 0.5)
            @test external_acceleration(case.forces, SVector{2,T}(0.0025, 0), zero(SVector{2,T}), t, zero(T)) == SVector{2,T}(0.0001, 0)
        end
    end
    @test poiseuille_velocity(0) ≈ 1.25e-5
    for dx in (0, -1e-5, Inf, NaN, 0.001, 0.00003)
        @test_throws ArgumentError poiseuille_2d_geometry(; dx)
    end
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "Poiseuille startup and developed GPU profile" begin
        for (graph, duration) in ((false, 0.002), (true, 0.5))
            mktempdir() do save
                result = run_poiseuille_2d(; simulation_time = duration, output_interval = duration,
                    use_graph = graph, visualize = false, open_log_file = false, save_location = save)
                p = result.final_state.particles
                @test result.final_state.complete
                @test result.final_state.time >= duration
                @test result.diagnostics.particle_count == 12750
                @test result.diagnostics.frozen_particles == 0
                @test result.diagnostics.output_frames == 2
                @test result.configuration.execution.graph_capture_failures == 0
                graph && @test result.configuration.execution.graph_batches > 0
                @test all(x -> 0 <= x[1] < 0.005, p.Position)
                fluid = findall(==(Fluid), p.Type)
                @test length(fluid) == 12250
                @test all(iszero, p.Velocity[p.Type .== Fixed])
                @test all(isfinite, p.Density)
                @test maximum(abs, p.Density[fluid] .- 1000) < 5
                expected = [poiseuille_velocity(p.Position[i][2], result.final_state.time) for i in fluid]
                actual = first.(p.Velocity[fluid])
                @test norm(actual - expected) / norm(expected) < 0.05
                @test maximum(abs, last.(p.Velocity[fluid])) < 1e-8
                csv_path = joinpath(save, "profile.csv")
                comparison = poiseuille_profile(result; path = csv_path)
                @test comparison.fluid_particles == 12250
                @test comparison.relative_l2 < 0.05
                @test length(readlines(csv_path)) == 12251
            end
        end
    end

    @testset "including the Poiseuille example runs it" begin
        mktempdir() do save
            scope = Module(gensym(:PoiseuilleInclude))
            Core.eval(scope, :(include(path) = Base.include($scope, path)))
            options = (; save_location = save, simulation_time = 0.002, output_interval = 0.002,
                visualize = false, open_log_file = false)
            result = Base.include(scope, joinpath(@__DIR__, "..", "example", "Poiseuille2d.jl")) do expression
                if expression isa Expr && expression.head === :let
                    for statement in expression.args[end].args
                        if statement isa Expr && statement.head === :(=) && statement.args[1] isa Symbol && haskey(options, statement.args[1])
                            statement.args[2] = QuoteNode(options[statement.args[1]])
                        end
                    end
                end
                expression
            end
            @test result isa SimulationResult
            @test result.final_state.complete
            @test result.diagnostics.particle_count == 12750
            @test isfile(joinpath(save, "Poiseuille2D.vtkhdf"))
            @test length(readlines(joinpath(save, "Poiseuille2D_Profile.csv"))) == 12251
        end
    end
end

end
