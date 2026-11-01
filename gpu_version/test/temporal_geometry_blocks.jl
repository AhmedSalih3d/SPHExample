module TemporalGeometryBlockTests
using Test, SPHExampleGPU, HDF5, StaticArrays

@testset "temporal geometry shares the particle file" begin
    mktempdir() do directory
        filepath = joinpath(directory, "combined.vtkhdf")
        regions = (; body = rectangle((0.0, 0.0), 1.0, 0.5))
        motions = (; body = (point, time) -> point + SVector(time, 0.0, 0.0))
        times = [0.0, 0.5, 1.0]
        # A small existing temporal dataset exercises in-file conversion without CUDA.
        SavePolygonMotionSequence(filepath, regions; motions, times)
        original = h5open(filepath, "r") do io
            read(io["VTKHDF/Points"])
        end
        @test_throws ArgumentError SavePolygonMotionSequence(filepath, regions;
            motions, times = [0.0, 1.0], block_name = "Geometry")
        @test_throws ArgumentError SavePolygonMotionSequence(filepath, regions;
            motions, times, block_name = "Particles")
        h5open(filepath, "r") do io
            @test read(io["VTKHDF/Points"]) == original
        end
        SavePolygonMotionSequence(filepath, regions; motions, times, block_name = "Geometry")
        SavePolygonMotionSequence(filepath, regions; motions, times, block_name = "SecondGeometry")
        @test_throws ArgumentError SavePolygonMotionSequence(filepath, regions;
            motions, times, block_name = "Geometry")
        h5open(filepath, "r") do io
            root = io["VTKHDF"]
            @test read_attribute(root, "Type") == "MultiBlockDataSet"
            @test Set(keys(root["Assembly"])) == Set(["Particles", "Geometry", "SecondGeometry"])
            for name in ("Particles", "Geometry", "SecondGeometry")
                @test read(root["$name/Points"]) == original
                @test read(root["$name/Steps/Values"]) == times
            end
        end
        @test readdir(directory) == ["combined.vtkhdf"]
    end
end
end
