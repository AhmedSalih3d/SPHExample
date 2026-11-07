module IISPHDamBreakTests

using Test, SPHExampleGPU, CUDA, Statistics, LinearAlgebra
include(joinpath(@__DIR__,"..","example","IISPH","cases.jl"))
using .IISPHCases

if CUDA.functional()
    @testset "IISPH dam break advances and preserves fluid mass" begin
        mktempdir() do save
            case = tank_case(Float32,save;name="DamBreak",nx=48,ny=24,
                fluid_width=16,fluid_height=20,duration=0.2,output_interval=0.1)
            initial_fluid = case.SimParticles.Type .== Fluid
            initial_ids = sort(case.SimParticles.ID[initial_fluid])
            initial_front = maximum(p -> p[1],case.SimParticles.Position[initial_fluid])
            initial_height = mean(p -> p[2],case.SimParticles.Position[initial_fluid])
            try
                result = RunSimulation(;case...)
                p = result.final_state.particles
                fluid = p.Type .== Fluid
                @test sort(p.ID[fluid]) == initial_ids
                @test maximum(x -> x[1],p.Position[fluid]) > initial_front+0.12
                @test mean(x -> x[2],p.Position[fluid]) < initial_height-0.02
                @test all(x -> all(isfinite,x),p.Position)
                @test all(x -> all(isfinite,x),p.Velocity)
                @test all(>=(0),p.Pressure)
                @test maximum(p.Density[fluid]) < 1.02case.SimConstants.ρ₀
                @test minimum(x -> x[1],p.Position[fluid]) > -case.SimConstants.dx/2
                # The density-based volume wall is soft for under-supported
                # leading-edge particles. Bound the known sub-spacing intrusion;
                # this is not a hard-contact/nonpenetration regression.
                @test minimum(x -> x[2],p.Position[fluid]) > -case.SimConstants.dx
                @test maximum(norm,p.Velocity[fluid]) < 5
                @test result.diagnostics.iisph.max_relative_density_residual <= case.SimSolver.relative_density_tolerance
                @test result.final_state.time == 0.2
            finally
                isopen(case.SimLogger.LoggerIo) && close(case.SimLogger.LoggerIo)
            end
        end
    end
else
    @testset "IISPH dam break requires CUDA" begin
        @test_skip CUDA.functional()
    end
end

end
