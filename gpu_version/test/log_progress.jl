using Test
using SPHExampleGPU

@testset "progress log lines are immediately visible" begin
    mktempdir() do dir
        logger = SimulationLogger(dir)
        try
            for frame in 1:3
                log_line(logger, "Part_$(frame)")
                # A separate reader sees each line before the logger is closed.
                @test read(joinpath(dir, "SimulationOutput.log"), String) ==
                    join(["Part_$(k)\n" for k in 1:frame])
            end
        finally
            close(logger.LoggerIo)
        end
    end
end
