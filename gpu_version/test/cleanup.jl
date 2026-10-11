module CleanupTests

using Test
using SPHExampleGPU

@noinline function orphaned_output(path)
    io = open(path, "w")
    write(io, "unreachable output handle")
    return nothing
end

@testset "simulation output cleanup" begin
    mktempdir() do directory
        @test CleanUpSimulationFolder(directory) === nothing
        output = joinpath(directory, "stale.vtkhdf")
        retained = joinpath(directory, "input.txt")
        write(output, "stale output")
        write(retained, "keep")
        @test CleanUpSimulationFolder(directory) === nothing
        @test !isfile(output)
        @test read(retained, String) == "keep"
        orphaned_output(output)
        @test CleanUpSimulationFolder(directory) === nothing
        @test !isfile(output)
        @test_logs (:warn, r"File could not be deleted") CleanUpSimulationFolder(joinpath(directory, "missing"))
    end
end

end
