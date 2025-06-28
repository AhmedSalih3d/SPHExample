using Test
using SPHExample
using StaticArrays

@testset "time stepping" begin
    pos = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 0.0)]
    vel = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(0.0, 0.0)]
    acc = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(0.0, -9.81)]
    sc = SimulationConstants{Float64}()
    ker = SPHKernelInstance{2, Float64}(WendlandC2(); dx=sc.dx)
    dt  = Δt(pos, vel, acc, sc, ker)
    @test dt > 0
    alloc = @allocated Δt(pos, vel, acc, sc, ker)
    @test alloc == 0
end
