module SimulationLoggerConfiguration
    using Format
    using TimerOutputs
    using Logging, LoggingExtras
    using Printf
    using Dates

    export SimulationLogger, generate_format_string, InitializeLogger, LogStep, LogFinal

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
        CurrentDate::DateTime
        CurrentDataStr::String


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

            # This should not be hardcoded here.
            CurrentDate    = now()
            CurrentDataStr = Dates.format(CurrentDate, "dd-mm-yyyy HH:MM:SS")

            new(io_logger, logger, format_string, ValuesToPrint, ValuesToPrintC, CurrentDate, CurrentDataStr)
        end
    end

    function InitializeLogger(SimLogger,SimConstants,SimMetaData)
        with_logger(SimLogger.Logger) do
            @info sprint(versioninfo)
            @info SimConstants
            @info SimMetaData
            
            # Print the formatted date and time
            @info "Logger Start Time: " * SimLogger.CurrentDataStr

            @info @. SimLogger.ValuesToPrint
            @info @. SimLogger.ValuesToPrintC
        end
    end

    function LogStep(SimLogger, SimMetaData, HourGlass)
        with_logger(SimLogger.Logger) do
            PartNumber               = "Part_" * lpad(SimMetaData.OutputIterationCounter,4,"0")
            PartTime                 = string(@sprintf("%-.6f", SimMetaData.TotalTime))
            PartTotalSteps           = string(SimMetaData.Iteration)
            CurrentSteps             = string(SimMetaData.Iteration -SimMetaData.StepsTakenForLastOutput)
            TimeUptillNow            = string(@sprintf("%-.3f",TimerOutputs.tottime(HourGlass)/1e9))
            TimePerPhysicalSecond    = string(@sprintf("%-.2f", TimerOutputs.tottime(HourGlass)/1e9 / SimMetaData.TotalTime))

            SecondsToFinish          = (SimMetaData.SimulationTime - SimMetaData.TotalTime) * (TimerOutputs.tottime(HourGlass)/1e9 / SimMetaData.TotalTime)
            ExpectedFinishTime       = now() + Second(trunc(Int,SecondsToFinish))
            ExpectedFinishTimeString = Dates.format(ExpectedFinishTime, "dd-mm-yyyy HH:MM:SS")

            @info @. $join(cfmt(SimLogger.FormatStr, (PartNumber, PartTime, PartTotalSteps,  CurrentSteps, TimeUptillNow, TimePerPhysicalSecond, ExpectedFinishTimeString)))
        end
    end

    function LogFinal(SimLogger, HourGlass)
        with_logger(SimLogger.Logger) do
            # Get the current date and time
            current_time = now()
            # Format the current date and time
            formatted_time = "Simulation finished at: " * Dates.format(current_time, "dd-mm-yyyy HH:MM:SS")

            @info formatted_time
            show(SimLogger.LoggerIo, HourGlass,sortby=:name)
            @info "\n Sorted by time \n"
            show(SimLogger.LoggerIo, HourGlass)
        end
    end

end