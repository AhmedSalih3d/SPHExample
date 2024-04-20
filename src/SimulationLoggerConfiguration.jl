module SimulationLoggerConfiguration
    using Format
    using TimerOutputs
    using Logging, LoggingExtras
    using Printf
    using Dates
    using InteractiveUtils

    export SimulationLogger, generate_format_string, InitializeLogger, LogSimulationDetails, LogStep, LogFinal

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

    function LogSimulationDetails(SimLogger::SimulationLogger, SimGeometry, SimParticles)
        with_logger(SimLogger.Logger) do
            # Improved logging format for simulation geometry
            @info "Simulation Geometry Details:"
            for (key, value) in pairs(SimGeometry)
                csv_file = value["CSVFile"]
                group_marker = value["GroupMarker"]
                particle_type = value["Type"]
                motion = if value["Motion"] === nothing "None" else string(value["Motion"]) end
    
                @info "$(lpad(string(key), 15)): CSV File -> $(lpad(csv_file, 50)), Group Marker -> $(lpad(string(group_marker), 3)), Type -> $(lpad(string(particle_type), 6)), Motion -> $(lpad(motion, 10))"
            end
    
            # Improved logging format for particle types and counts
            @info "Particle Types and Counts:"
            types = unique(SimParticles.Type)
            for t in types
                count = sum(SimParticles.Type .== t)
                @info "Type $(lpad(string(t), 6)): $(lpad(string(count), 6)) particles"
            end
    
            total_particles = length(SimParticles.Type)
            unique_markers = unique(SimParticles.GroupMarker)
            marker_counts = [(marker, sum(SimParticles.GroupMarker .== marker)) for marker in unique_markers]
            @info "Total number of particles: $total_particles"
            @info "Group Markers and Counts:"
            for (marker, count) in marker_counts
                @info "Marker $(lpad(string(marker), 3)): $(lpad(string(count), 6)) particles"
            end
            @info ""
        end
    end

    function InitializeLogger(SimLogger,SimConstants,SimMetaData, SimGeometry, SimParticles)
        with_logger(SimLogger.Logger) do
            @info sprint(InteractiveUtils.versioninfo)
            @info SimConstants
            @info SimMetaData
            
            # Print the formatted date and time
            @info "Logger Start Time: " * SimLogger.CurrentDataStr

            LogSimulationDetails(SimLogger, SimGeometry, SimParticles)

            @info @. SimLogger.ValuesToPrint
            @info @. SimLogger.ValuesToPrintC
        end
    end


    function LogStep(SimLogger, SimMetaData, HourGlass)
        with_logger(SimLogger.Logger) do
            PartNumber               = "Part_" * lpad(SimMetaData.OutputIterationCounter,4,"0")
            PartTime                 = string(@sprintf("%-.6f", SimMetaData.TotalTime))
            PartTotalSteps           = string(SimMetaData.Iteration)
            CurrentSteps             = string(SimMetaData.Iteration - SimMetaData.StepsTakenForLastOutput)
            TimeUptillNow            = string(@sprintf("%-.3f",TimerOutputs.tottime(HourGlass)/1e9))
            TimePerPhysicalSecond    = string(@sprintf("%-.2f", TimerOutputs.tottime(HourGlass)/1e9 / SimMetaData.TotalTime))

            SecondsToFinish          = (SimMetaData.SimulationTime - SimMetaData.TotalTime) * (TimerOutputs.tottime(HourGlass)/1e9 / SimMetaData.TotalTime)
            ExpectedFinishTime       = now() + Second(ceil(Int,SecondsToFinish))
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
            @info "Simulation took " * @sprintf("%-.2f", TimerOutputs.tottime(HourGlass)/1e9) * "[s]"
            show(SimLogger.LoggerIo, HourGlass,sortby=:name)
            @info "\n Sorted by time \n"
            show(SimLogger.LoggerIo, HourGlass)
        end
    end

end