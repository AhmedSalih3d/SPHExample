module SimulationLoggerConfiguration
    using Format
    using TimerOutputs
    using Logging, LoggingExtras
    using Printf
    using Dates
    using InteractiveUtils

    using ..SimulationGeometry

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

    function LogSimulationDetails(SimLogger::SimulationLogger, SimGeometry::Vector{Geometry{Dimensions, FloatType}}, SimParticles; sort_by=:GroupMarker) where {Dimensions, FloatType}
        with_logger(SimLogger.Logger) do
            # Calculate the maximum lengths for alignment
            max_csv_len = maximum(length(geom.CSVFile) for geom in SimGeometry) + 2
            max_group_marker_len = maximum(length(string(geom.GroupMarker)) for geom in SimGeometry) + 2
            max_type_len = maximum(length(string(geom.Type)) for geom in SimGeometry) + 2
    
            @info "Simulation Geometry Details:"
            for geom in SimGeometry
                csv_file = geom.CSVFile
                group_marker = geom.GroupMarker
                particle_type = geom.Type
                motion = if geom.Motion === nothing "None" else string(geom.Motion) end
    
                formatted_csv_file = rpad(csv_file, max_csv_len)
                formatted_group_marker = rpad(string(group_marker), max_group_marker_len)
                formatted_type = rpad(string(particle_type), max_type_len)
                formatted_motion = motion  # No padding necessary if motion detail is to start immediately after type
    
                @info "CSV File -> $formatted_csv_file, Group Marker -> $formatted_group_marker, Type -> $formatted_type, Motion -> $formatted_motion"
            end
    
            # Handling particle types and counts
            @info "Particle Types and Counts:"
            type_counts = [(t, sum(SimParticles.Type .== t)) for t in unique(SimParticles.Type)]
            sort!(type_counts, by=x -> Int(x[1]))
            for (type, count) in type_counts
                formatted_type = rpad(string(type), max_type_len)
                formatted_count = lpad(string(count), 6)  # Right-aligned count for numerical clarity
                @info "Type $formatted_type: $formatted_count particles"
            end
    
            total_particles = length(SimParticles.Type)
            @info "Total number of particles: $total_particles"
    
            if sort_by == :GroupMarker
                @info "Group Markers and Counts (Sorted):"
                marker_counts = sort([(marker, sum(SimParticles.GroupMarker .== marker)) for marker in unique(SimParticles.GroupMarker)])
                for (marker, count) in marker_counts
                    formatted_marker = rpad(string(marker), max_group_marker_len)
                    formatted_count = lpad(string(count), 6)
                    @info "Marker $formatted_marker: $formatted_count particles"
                end
            end
            @info ""
        end
    end
    
    
    function InitializeLogger(SimLogger,SimConstants,SimMetaData, SimKernel, SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)
        with_logger(SimLogger.Logger) do
            @info sprint(InteractiveUtils.versioninfo)
            @info SimConstants
            @info SimMetaData
            @info SimKernel
            @info SimViscosity
            @info SimDensityDiffusion
            
            # Print the formatted date and time
            @info "Logger Start Time: " * SimLogger.CurrentDataStr

            LogSimulationDetails(SimLogger, SimGeometry, SimParticles)

            @info @. SimLogger.ValuesToPrint
            @info @. SimLogger.ValuesToPrintC
        end
    end


    function LogStep(SimLogger, SimMetaData, HourGlass)
        with_logger(SimLogger.Logger) do
            PartNumber               = "Part_" * lpad(SimMetaData.OutputIterationCounter, 4, "0")
            PartTime                 = string(@sprintf("%-.6f", SimMetaData.TotalTime))
            PartTotalSteps           = string(SimMetaData.Iteration)
            CurrentSteps             = string(SimMetaData.Iteration - SimMetaData.StepsTakenForLastOutput)
    
            elapsed_time_            = TimerOutputs.tottime(HourGlass) / 1e9
            TimeUptillNow            = string(@sprintf("%-.3f", elapsed_time_))
            TimePerPhysicalSecond    = string(@sprintf("%-.2f", elapsed_time_ / SimMetaData.TotalTime))
    
            SecondsToFinish          = (SimMetaData.SimulationTime - SimMetaData.TotalTime) * (elapsed_time_ / SimMetaData.TotalTime)
            if isnan(SecondsToFinish)
                SecondsToFinish = 0.0
                ExpectedFinishTime       = now() + Second(ceil(Int, SecondsToFinish))
                ExpectedFinishTimeString = missing
            else
                ExpectedFinishTime       = now() + Second(ceil(Int, SecondsToFinish))
                ExpectedFinishTimeString = Dates.format(ExpectedFinishTime, "dd-mm-yyyy HH:MM:SS")
            end
            
    
            @info @. $join(cfmt(SimLogger.FormatStr, (PartNumber, PartTime, PartTotalSteps, CurrentSteps, TimeUptillNow, TimePerPhysicalSecond, ExpectedFinishTimeString)))
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