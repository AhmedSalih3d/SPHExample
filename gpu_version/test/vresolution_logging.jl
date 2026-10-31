module VResolutionLoggingTests

using Test, SPHExampleGPU, CUDA, TimerOutputs
const VR = SPHExampleGPU.VariableResolution
include(joinpath(@__DIR__,"..","example","VResolutionDamBreak2d","Setup.jl"))

@testset "variable-resolution logger finalizes failures and interruptions" begin
    for failure in (ErrorException("test log failure"),InterruptException())
        mktempdir() do directory
            captured_logger = Ref{Any}(nothing)
            caught = try
                redirect_stdout(devnull) do
                    VR.with_resolution_logger(directory) do logger,timer
                        captured_logger[] = logger
                        @timeit timer "00 Initialization" begin
                            log_line(logger,"Before failure")
                            @test occursin("Before failure",read(joinpath(directory,"SimulationOutput.log"),String))
                        end
                        throw(failure)
                    end
                end
            catch error
                error
            end
            @test caught === failure
            @test !isopen(captured_logger[].LoggerIo)
            text = read(joinpath(directory,"SimulationOutput.log"),String)
            status = failure isa InterruptException ? "interrupted" : "failed"
            @test occursin("Simulation $status at:",text)
            @test occursin("Simulation error:",text)
            @test occursin("Simulation took ",text)
            @test occursin("00 Initialization",text)
        end
    end
end

@testset "native variable-resolution progress and timing log" begin
    if CUDA.functional()
        case = vresolution_dambreak_case(dx=0.05)
        directory = mktempdir()
        started = time_ns()
        result = redirect_stdout(devnull) do
            run_variable_resolution_simulation(;
                coarse_geometry=case.coarse.geometry,fine_geometry=case.fine.geometry,
                coarse_constants=case.coarse.constants,fine_constants=case.fine.constants,
                coarse_kernel=case.coarse.kernel,fine_kernel=case.fine.kernel,
                region=case.region,simulation_time=0.001,output_interval=0.0005,
                save_location=directory,particle_reserve=32)
        end
        wall = (time_ns()-started)/1e9
        @test result.log_path == joinpath(directory,"SimulationOutput.log")
        text = read(result.log_path,String)
        for label in ("Logger Start Time:","CUDA information:","Coarse spacing:","Fine spacing:",
                "PART [-]","Remaining Time [Date]","Initialization time:","Solver stepping time:",
                "Output time (download + VTKHDF/CSV):","End-to-end throughput:","Stepping throughput",
                "Cell list rebuilds (predictor coarse/fine):","Simulation finished at:","Simulation took ",
                "00 Initialization","01 Solver","02 Output Frame")
            @test occursin(label,text)
        end
        rows = [split(strip(line);limit=7) for line in eachline(IOBuffer(text)) if startswith(strip(line),"Part_")]
        @test length(rows) == 3
        @test all(length(row)==7 for row in rows)
        @test [parse(Float64,row[2]) for row in rows] == result.times
        totals = [parse(Int,row[3]) for row in rows]
        intervals = [parse(Int,row[4]) for row in rows]
        @test cumsum(intervals) == totals
        @test last(totals) == result.iterations
        elapsed = [parse(Float64,row[5]) for row in rows]
        @test issorted(elapsed)
        @test last(elapsed) <= result.timings.total_wall_seconds + 0.001
        @test 0 < result.timings.total_wall_seconds <= wall + 0.01
        @test result.timings.initialization_seconds > 0
        @test result.timings.solver_seconds > 0
        @test result.timings.output_seconds > 0
        phases = result.timings.initialization_seconds + result.timings.solver_seconds + result.timings.output_seconds
        @test phases <= result.timings.total_wall_seconds + 0.01
        @test result.particle_counts == (; coarse=800,fine=0)

        failed_directory = mktempdir()
        @test_throws ArgumentError redirect_stdout(devnull) do
            run_variable_resolution_simulation(;
                coarse_geometry=case.coarse.geometry,fine_geometry=case.fine.geometry,
                coarse_constants=case.coarse.constants,fine_constants=case.fine.constants,
                coarse_kernel=case.coarse.kernel,fine_kernel=case.fine.kernel,
                region=case.region,simulation_time=0.001,save_location=failed_directory,particle_reserve=0)
        end
        failed_log = read(joinpath(failed_directory,"SimulationOutput.log"),String)
        @test occursin("Simulation failed at:",failed_log)
        @test occursin("particle reserve must be a positive integer",failed_log)
        @test !occursin("Simulation finished at:",failed_log)
    else
        @test_skip "CUDA not available"
    end
end

end
