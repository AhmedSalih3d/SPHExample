# Full normal-RunSimulation integration validation (requires CUDA).
# julia --project=gpu_version -t 2,0 gpu_version/test/run_vresolution_solver_dambreak.jl
using Test, SPHExampleGPU, CSV, HDF5, Dates
include(joinpath(@__DIR__,"..","example","VResolutionDamBreak2d","Setup.jl"))
output = joinpath(@__DIR__,"..","output","vresolution_dambreak2d_solver_"*Dates.format(now(),"yyyymmdd_HHMMSS"))
result = run_vresolution_dambreak_solver(;save_location=output)
history = CSV.File(result.diagnostics.resolution.history_path)
mass = collect(history.represented_fluid_mass)
mass_error = maximum(abs.(mass ./ first(mass) .- 1))
events = result.diagnostics.resolution.events
@testset "full metadata-driven variable-resolution solver" begin
    @test result.termination_reason === :completed
    @test result.final_state.time == 1.0
    @test result.diagnostics.output_frames == 401
    @test result.configuration.execution.path === :cuda_graph
    @test events[2][1] > 0
    @test all(e -> e[3] == 0 && e[4] == 0,events)
    @test mass_error < 0.01
    p = result.final_state.particles
    @test all(all(isfinite,x) for x in p.Position)
    @test all(all(isfinite,x) for x in p.Velocity)
    @test all(isfinite,p.Density)
    h5open(joinpath(output,"DamBreak2D_VRes.vtkhdf"),"r") do file
        root = file["VTKHDF"]
        times,counts = read(root["Steps/Values"]),read(root["NumberOfPoints"])
        @test length(times) == 401
        @test times == collect(0:400).*0.0025
        @test length(unique(counts)) > 1
        @test size(root["Points"],2) == sum(counts)
        @test read(root["Steps/PointOffsets"]) == [0;cumsum(counts)[1:end-1]]
        for name in keys(root["PointData"])
            data = read(root["PointData/"*name])
            @test size(data,ndims(data)) == sum(counts)
            @test all(isfinite,data)
        end
        ids,owned = read(root["PointData/ID"]),read(root["PointData/Owned"])
        offset = 0
        for n in counts
            @test length(unique(ids[offset+1:offset+n])) == n
            offset += n
        end
        @test all(==(1),owned)
    end
end
println("Maximum represented-mass deviation: ",100mass_error,"%")
println("Normal solver output: ",output)
