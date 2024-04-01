module SimulationLoggerConfiguration
    using Format
    using TimerOutputs
    using Logging, LoggingExtras
    using Printf

    export SimulationLogger, generate_format_string

    # Function to dynamically generate a format string based on values
    function generate_format_string(values)
        # Calculate the display length for each value
        lengths = [length(string(value)) for value in values]
        
        # Optionally, add extra padding
        padding = 10 #maximum(lengths)  # Adjust padding as needed
        lengths = [len + padding for len in lengths]
        
        # Build format specifiers for each length
        format_specifiers = ["%-$(len)s" for len in lengths]
        
        # Combine into a single format string
        format_str = join(format_specifiers, " ")
        
        return format_str
    end

    struct SimulationLogger
        LoggerIo::IOStream
        Logger::FormatLogger
        FormatStr::String
        ValuesToPrint::String
        ValuesToPrintC::String

        function SimulationLogger(SaveLocation::String)
            io_logger = open(SaveLocation * "/" * "SimulationOutput.log", "w")
            logger    = FormatLogger(io_logger::IOStream) do io, args
                # Write the module, level and message only
                # println(io, args._module, " | ", "[", args.level, "] ", args.message)
                println(io, args.message)
            end

            values        = ("PART [-]", "PartTime [s]", "TotalSteps [-] ", "Steps  [-] ", "Run Time [s]", "Time/Sec [-]", "Remaining Time [Date]")
            values_eq     = map(x -> repeat("=", length(x)), values)
            format_string = generate_format_string(values)

            ValuesToPrint  = @. $join(cfmt(format_string, values))
            ValuesToPrintC = @. $join(cfmt(format_string, values_eq))

            new(io_logger, logger, format_string, ValuesToPrint, ValuesToPrintC)
        end
        
    end

end