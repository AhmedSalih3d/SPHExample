module SimulationLoggerConfiguration
    using Format
    using TimerOutputs
    using Logging, LoggingExtras
    using Printf

    export SimulationLogger

    struct SimulationLogger
        LoggerIo::IOStream
        Logger::FormatLogger
        FormatStr::String

        function SimulationLogger(SaveLocation::String)
            io_logger = open(SaveLocation * "/" * "SimulationOutput.log", "w")
            logger    = FormatLogger(io_logger::IOStream) do io, args
                # Write the module, level and message only
                # println(io, args._module, " | ", "[", args.level, "] ", args.message)
                println(io, args.message)
            end

            format_string = "%-14s %-17s %-17s %-12s %-14s %-14s"

            new(io_logger, logger, format_string)
        end
        
    end

end