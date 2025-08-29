"""
Utility helpers for logging simulation progress and timings.
"""
module SimulationLoggerConfiguration
    using TimerOutputs
    using Logging, LoggingExtras
    using Printf
    using Dates
    using InteractiveUtils
    using Base.Threads

    using ..SimulationGeometry

    export SimulationLogger, generate_format_string, InitializeLogger, LogSimulationDetails, LogStep, LogFinal

    """
        generate_format_string(values; padding=10)

    Generate a printf compatible format string where each column is padded to
    the length of the corresponding entry in `values` plus `padding` spaces.
    This is used to print nicely aligned progress information.
    """
    function generate_format_string(values; padding=10)
        lengths = [length(string(value)) + padding for value in values]
        format_specifiers = ["%-$(len)s" for len in lengths]
        return join(format_specifiers, " ")
    end

    """
    `SimulationLogger(save_location; filename="SimulationOutput.log", to_console=false)`

    Container holding the loggers and formatting information used during a
    simulation run. By default logging output is written to
    `joinpath(save_location, filename)`. If `to_console` is `true`, log messages
    are also echoed to the Julia REPL via a [`TeeLogger`](https://github.com/JuliaLogging/LoggingExtras.jl).
    When `to_console` is `false`, the code can instead display a terminal
    progress bar.
    """
    struct SimulationLogger
        LoggerIo::IOStream           # handle to the log file
        Logger::AbstractLogger       # may be a TeeLogger or FormatLogger
        FormatStr::String            # format used for progress lines
        ValuesToPrint::String        # header line describing logged values
        ValuesToPrintC::String       # separator line below the header
        CurrentDate::DateTime        # start time of the simulation
        CurrentDataStr::String       # preformatted start time string
        ToConsole::Bool              # whether log output is echoed to REPL


        function SimulationLogger(SaveLocation::String; filename="SimulationOutput.log", to_console::Bool=false)
            io_logger = open(joinpath(SaveLocation, filename), "w")
            file_logger = FormatLogger(io_logger) do io, args
                println(io, args.message)
            end
            logger = if to_console
                console_logger = FormatLogger(stdout) do io, args
                    println(io, args.message)
                end
                TeeLogger(file_logger, console_logger)
            else
                file_logger
            end

            values        = ("PART [-]", "PartTime [s]", "TotalSteps [-] ", "Steps  [-] ", "Run Time [s]", "Time/Sec [-]", "Remaining Time [Date]")
            values_eq     = map(x -> repeat("=", length(x)), values)
            format_string = generate_format_string(values)

            fmt = Printf.Format(format_string)
            ValuesToPrint  = Printf.format(fmt, values...)
            ValuesToPrintC = Printf.format(fmt, values_eq...)

            # This should not be hardcoded here.
            CurrentDate    = now()
            CurrentDataStr = Dates.format(CurrentDate, "dd-mm-yyyy HH:MM:SS")

            new(io_logger, logger, format_string, ValuesToPrint, ValuesToPrintC, CurrentDate, CurrentDataStr, to_console)
        end
    end

    """
        LogSimulationDetails(logger, geometries, particles; sort_by=:GroupMarker)

    Print information about all geometries involved in the simulation and a
    summary of particle counts per type. The optional `sort_by` argument controls
    how the group marker statistics are ordered.
    """
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
    
    
    """
        InitializeLogger(logger, constants, metadata, kernel, viscosity,
                         densitydiffusion, geometry, particles)

    Write a short summary of the simulation configuration to the log file and
    store the start time. This is typically called once before the time stepping
    loop begins.
    """
    function InitializeLogger(SimLogger,SimConstants,SimMetaData, SimKernel, SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles)
        with_logger(SimLogger.Logger) do
            @info sprint(InteractiveUtils.versioninfo)
            @info "Julia threads: $(Threads.nthreads())"
            @info SimConstants
            @info SimMetaData
            @info SimKernel
            @info SimViscosity
            @info SimDensityDiffusion
            
            # Print the formatted date and time
            @info "Logger Start Time: " * SimLogger.CurrentDataStr

            LogSimulationDetails(SimLogger, SimGeometry, SimParticles)

            @info SimLogger.ValuesToPrint
            @info SimLogger.ValuesToPrintC
        end
    end


    """
        LogStep(logger, metadata, timer)

    Record information about the current iteration such as physical time,
    wall-clock time and an estimate of the remaining run time.
    """
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
            
            formatted_line = Printf.format(
                Printf.Format(SimLogger.FormatStr),
                PartNumber,
                PartTime,
                PartTotalSteps,
                CurrentSteps,
                TimeUptillNow,
                TimePerPhysicalSecond,
                ExpectedFinishTimeString,
            )
            @info formatted_line
        end
    end
    

    """
        LogFinal(logger, timer)

    Called once the simulation loop ends. Prints total run time and a summary of
    the collected [`TimerOutput`] information.
    """
    function LogFinal(SimLogger, HourGlass)
        with_logger(SimLogger.Logger) do
            # Get the current date and time
            current_time = now()
            # Format the current date and time
            formatted_time = "\n Simulation finished at: " * Dates.format(current_time, "dd-mm-yyyy HH:MM:SS")

            @info formatted_time
            @info "\n Simulation took " * @sprintf("%-.2f", TimerOutputs.tottime(HourGlass)/1e9) * "[s]"
            show(SimLogger.LoggerIo, HourGlass,sortby=:name)
            @info "\n Sorted by time \n"
            show(SimLogger.LoggerIo, HourGlass)
        end
    end

end
