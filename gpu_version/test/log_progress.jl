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

module WallTimeLoggingTests

using Test, SPHExampleGPU, TimerOutputs, Dates

const LOGGER = SPHExampleGPU.SimulationLoggerConfiguration
const DATE_FORMAT = dateformat"dd-mm-yyyy HH:MM:SS"

function check_progress(logger, metadata, timer)
    wall_before = now()
    elapsed_before = LOGGER.elapsed_wall_time(logger)
    fields = split(strip(step_log_line(logger, metadata, timer)); limit = 7)
    elapsed_after = LOGGER.elapsed_wall_time(logger)
    wall_after = now()
    @test length(fields) == 7
    @test elapsed_before - 0.0005 <= parse(Float64, fields[5]) <= elapsed_after + 0.0005
    progress = Float64(metadata.TotalTime) - logger.clock.initial_time
    if progress > 0
        @test elapsed_before / progress - 0.005 <= parse(Float64, fields[6]) <= elapsed_after / progress + 0.005
        remaining = max(0.0, Float64(metadata.SimulationTime) - Float64(metadata.TotalTime))
        earliest = wall_before + Second(ceil(Int, remaining * elapsed_before / progress))
        latest = wall_after + Second(ceil(Int, remaining * elapsed_after / progress))
        @test floor(earliest, Second) <= DateTime(strip(fields[7]), DATE_FORMAT) <= ceil(latest, Second)
    else
        @test fields[6] == "missing"
        @test strip(fields[7]) == "missing"
    end
    return parse(Float64, fields[5])
end

@testset "wall-clock runtime and ETA include untimed work and resumed progress" begin
    mktempdir() do directory
        logger = SimulationLogger(directory)
        metadata = SimulationMetaData{2, Float64}(SimulationName = "WallTime",
            SaveLocation = directory, TotalTime = 4.0, SimulationTime = 10.0)
        timer = metadata.HourGlass
        try
            LOGGER.start_runtime!(logger, metadata.TotalTime)
            # Advance only the elapsed clock; no phase has measured this time.
            logger.clock.start_ns -= UInt64(12_000_000_000)
            @test TimerOutputs.tottime(timer) == 0
            @test check_progress(logger, metadata, timer) >= 12
            metadata.TotalTime = 6.0
            before_reset = check_progress(logger, metadata, timer)
            reset_timer!(timer)
            @test check_progress(logger, metadata, timer) >= before_reset
            disable_timer!(timer)
            @test check_progress(logger, metadata, timer) >= before_reset
            # The finish estimate must not move into the past after overshoot.
            metadata.TotalTime = 10.5
            check_progress(logger, metadata, timer)
            metadata.TotalTime = 3.0
            check_progress(logger, metadata, timer)

            # A new run resets both elapsed time and the physical-time baseline.
            metadata.TotalTime = 7.0
            start_before = time_ns()
            LOGGER.start_runtime!(logger, metadata.TotalTime)
            start_after = time_ns()
            @test start_before <= logger.clock.start_ns <= start_after
            @test logger.clock.initial_time == 7.0
            check_progress(logger, metadata, timer)
        finally
            close(logger.LoggerIo)
        end
    end
end

@testset "final runtime uses the monotonic clock for every exit status" begin
    for status in ("finished", "interrupted", "failed")
        mktempdir() do directory
            logger = SimulationLogger(directory)
            timer = TimerOutput()
            logger.clock.start_ns -= UInt64(8_000_000_000)
            elapsed_before = LOGGER.elapsed_wall_time(logger)
            try
                redirect_stdout(devnull) do
                    LogFinal(logger, timer; status)
                end
            finally
                close(logger.LoggerIo)
            end
            elapsed_after = LOGGER.elapsed_wall_time(logger)
            output = read(joinpath(directory, "SimulationOutput.log"), String)
            runtime = match(r"Simulation took ([0-9.]+)\[s\]", output)
            @test runtime !== nothing
            @test elapsed_before - 0.005 <= parse(Float64, runtime.captures[1]) <= elapsed_after + 0.005
            @test occursin("Simulation $status at:", output)
            @test TimerOutputs.tottime(timer) == 0
            @test occursin(r"─{10,}\s*\z", output)
        end
    end
end

end
