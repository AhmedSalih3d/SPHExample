module Test06ComparisonTests
using Test, SPHExampleGPU, Statistics, LinearAlgebra
include(joinpath(@__DIR__,"dual_compare.jl"))
using .Test06DualComparison
const C = Test06DualComparison

@testset "Motion integration preserves velocity without rounded-position acceleration" begin
    trajectory = C.integrated_trajectory()
    original = C.body_trajectory()
    data = C.read_reference(joinpath(C.REFERENCE_DIR,"Motion_Body.dat"))
    @test first(trajectory_state(trajectory,0.0)) == zero(first(trajectory_state(original,0.0)))
    for time in (2.005,3.17,5.345,7.99)
        offset,velocity,acceleration = trajectory_state(trajectory,time)
        @test velocity[1] ≈ 0.999899 atol = 1e-10
        @test abs(acceleration[1]) < 1e-8
        @test abs(offset[1]-first(trajectory_state(original,time))[1]) < 3e-5
    end
    @test all(abs(trajectory_state(trajectory,data[i,1])[2][1]-data[i,3]) < 1e-10 for i in 1:900)
end

@testset "Noise metric rejects extrapolation and resolves a known oscillation" begin
    times = collect(0.:0.02:8.)
    reference_times = collect(0.:0.003:8.1)
    reference = hcat(reference_times,ones(length(reference_times)),ones(length(reference_times)))
    constant = ones(length(times))
    perfect,_ = C.drag_metrics(times,constant,constant,reference)
    @test perfect["pressure_nrmse"] == 0
    @test perfect["pressure_highpass_rms"] == 0
    @test perfect["pressure_raw_nrmse"] == 0
    oscillatory = 1 .+ 0.1sin.(4π.*times)
    measured,_ = C.drag_metrics(times,oscillatory,constant,reference)
    @test measured["pressure_dominant_hz"] ≈ 2 atol = 0.1
    @test measured["pressure_highpass_rms"] ≈ 0.1/sqrt(2) atol = 0.001
    @test measured["pressure_nrmse"] ≈ 0.066 atol = 0.003
    @test measured["viscous_nrmse"] == 0
    @test measured["pressure_raw_nrmse"] ≈ 0.1/sqrt(2) atol = 0.001
    higher,_ = C.drag_metrics(times,1 .+ 0.1sin.(16π.*times),constant,reference)
    @test higher["pressure_raw_nrmse"] > 0.06
    @test higher["pressure_raw_nrmse"] > 3higher["pressure_nrmse"]
    @test_throws ArgumentError C.drag_metrics([0.,1.],[1.,1.],[1.,1.],reference)
end
end
