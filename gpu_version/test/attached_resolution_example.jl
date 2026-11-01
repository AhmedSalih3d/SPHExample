module AttachedResolutionExampleTests
using Test, SPHExampleGPU, CUDA, HDF5, CSV, StaticArrays
include(joinpath(@__DIR__, "..", "example", "VResolutionAttachedBody2d.jl"))

if CUDA.functional()
    @testset "reduced attached-body examples save one animated file" begin
        for floating in (false, true)
            directory = mktempdir()
            result = redirect_stdout(devnull) do
                run_attached_resolution_example(; floating, dx = 0.08, duration = 0.002,
                    output_interval = 0.001, save_location = directory)
            end
            @test result.final_state.complete
            name = floating ? "FloatingRefinement" : "MovingRefinement"
            @test filter(f -> endswith(f, ".vtkhdf"), readdir(directory)) == [name * ".vtkhdf"]
            h5open(joinpath(directory, name * ".vtkhdf"), "r") do io
                root = io["VTKHDF"]
                @test read(root["Particles/Steps/Values"]) == [0.0, 0.001, 0.002]
                @test read(root["Geometry/Steps/Values"]) == read(root["Particles/Steps/Values"])
                counts = read(root["Geometry/NumberOfPoints"])
                points = read(root["Geometry/Points"])
                n = first(counts)
                @test all(==(n), counts)
                @test all(isfinite, points)
                # The first eight polygon points form the fixed tank skin.
                @test points[:, 1:8] == points[:, 2n+1:2n+8]
                @test points[:, 9:n] != points[:, 2n+9:3n]
                @test haskey(root, "Assembly/Geometry")
            end
        end
    end
end
end
