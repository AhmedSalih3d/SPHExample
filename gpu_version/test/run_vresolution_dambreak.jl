# Full native dam-break validation, kept separate from the focused suite.
# From the repository root:
# julia --project=gpu_version -t 2,0 gpu_version/test/run_vresolution_dambreak.jl
using Test, SPHExampleGPU, CSV, HDF5, Dates
include(joinpath(@__DIR__,"..","example","VResolutionDamBreak2d","Setup.jl"))

length(ARGS) <= 2 || error("usage: run_vresolution_dambreak.jl [dx] [duration]")
dx = isempty(ARGS) ? 0.01 : parse(Float64,ARGS[1])
duration = length(ARGS) < 2 ? 1.0 : parse(Float64,ARGS[2])
stamp = Dates.format(now(),"yyyymmdd_HHMMSS")
output = joinpath(@__DIR__,"..","output","vresolution_dambreak2d_validation_$stamp")
result = run_vresolution_dambreak(; dx,duration,output_interval = 0.0025,save_location = output)
history = CSV.File(result.history_path)
represented_mass = collect(history.represented_fluid_mass)
mass_error = maximum(abs.(represented_mass ./ first(represented_mass) .- 1))
excluded_mass = sum(Array(d.events)[3] * d.constants.m₀ for d in (result.coarse,result.fine))

@testset "full native variable-resolution dam break" begin
    @test result.time == duration
    @test Array(result.fine.events)[1] > 0
    @test all(Array(d.events)[4] == 0 for d in (result.coarse,result.fine))
    @test excluded_mass / first(represented_mass) < 0.001
    @test mass_error < 0.01
    for domain in (result.coarse,result.fine)
        p = domain.particles
        live = Array(p.Type) .== Fluid
        @test all(all(isfinite,x) for x in Array(p.Position)[live])
        @test all(all(isfinite,v) for v in Array(p.Velocity)[live])
        @test all(isfinite,Array(p.Density)[live])
    end
    h5open(joinpath(result.output,last(result.files)),"r") do file
        data = file["VTKHDF/PointData"]
        @test Set(read(data["Resolution"])) == Set(Int8[0,1])
        ids = read(data["ID"])
        @test length(unique(ids)) == length(ids)
    end
end
println("Maximum represented-mass deviation: ",100mass_error,"%")
println("Mass excluded by the reference density limits: ",excluded_mass)
println("Native variable-resolution output: ",result.output)
