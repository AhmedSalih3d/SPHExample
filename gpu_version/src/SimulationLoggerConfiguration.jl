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
    using CUDA
    using StructArrays: components

    using ..SimulationGeometry
    using ..SimulationMetaDataConfiguration: position_float_type

    export SimulationLogger, generate_format_string, InitializeLogger, LogSimulationDetails, LogStep, step_log_line, log_line, LogFinal

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

        function SimulationLogger(SaveLocation::String; filename="SimulationOutput.log")
            io_logger = open(joinpath(SaveLocation, filename), "w")
            file_logger = FormatLogger(io_logger) do io, args
                println(stdout, args.message)
                println(io, args.message)
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

            new(io_logger, file_logger, format_string, ValuesToPrint, ValuesToPrintC, CurrentDate, CurrentDataStr)
        end
    end

    function host_particle_bytes(particles)
        seen = Base.IdSet()
        return sum(components(particles); init = 0) do buffer
            buffer in seen && return 0
            push!(seen, buffer)
            return sizeof(buffer)
        end
    end

    function memory_usage_line(label, bytes, particles)
        per_particle = particles == 0 ? "n/a" : @sprintf("%.2f", bytes / particles)
        return @sprintf("%s: %d bytes (%.4f MiB; %s bytes/particle)",
            label, bytes, bytes / 2.0^20, per_particle)
    end

    """
        device_buffer_bytes(x, seen = Base.IdSet())

    Count live `CuArray` payload bytes in data containers without downloading
    their contents. Shared array objects are counted once; host arrays with
    inline elements, allocator reservations and allocation overhead are excluded.
    """
    function device_buffer_bytes(x, seen = Base.IdSet())
        (x === nothing || isbitstype(typeof(x))) && return 0
        x in seen && return 0
        push!(seen, x)
        if x isa CuArray
            return length(x) * sizeof(eltype(x))
        elseif x isa AbstractArray
            isbitstype(eltype(x)) && return 0
            return sum(v -> device_buffer_bytes(v, seen), x; init = 0)
        end
        return sum(name -> device_buffer_bytes(getfield(x, name), seen),
            fieldnames(typeof(x)); init = 0)
    end

    function gpu_memory_bytes(buffers)
        seen = Base.IdSet()
        return (; particles = device_buffer_bytes(buffers.particles, seen),
            support = device_buffer_bytes(buffers.support, seen),
            cells = device_buffer_bytes(buffers.cells, seen),
            auxiliary = device_buffer_bytes(buffers.auxiliary, seen))
    end

    function log_gpu_memory(logger, buffers, particles; phase)
        bytes = gpu_memory_bytes(buffers)
        with_logger(logger.Logger) do
            @info "GPU live-buffer memory ($phase):"
            @info memory_usage_line("GPU particle arrays (state + reorder)", bytes.particles, particles)
            @info memory_usage_line("GPU support arrays", bytes.support, particles)
            @info memory_usage_line("GPU cell-list workspace", bytes.cells, particles)
            @info memory_usage_line("GPU auxiliary arrays (motion, reductions, step state, floating)", bytes.auxiliary, particles)
            @info memory_usage_line("GPU accounted buffers total", sum(bytes), particles)
            @info "GPU memory scope: live array payload; shared arrays counted once. Excludes output/staging buffers, CUDA context/JIT/graphs, allocator reservations and allocation overhead; not process or peak VRAM."
        end
        return nothing
    end

    """
        LogSimulationDetails(logger, geometries, particles; sort_by=:GroupMarker)

    Print information about all geometries involved in the simulation and a
    summary of particle counts, precision and host particle-array payload.
    Shared array objects are counted once. The optional `sort_by` argument
    controls how the group marker statistics are ordered.
    """
    function LogSimulationDetails(SimLogger::SimulationLogger, SimGeometry::Vector{SPHGeometry{Dimensions, FloatType}}, SimParticles; sort_by=:GroupMarker) where {Dimensions, FloatType}
        with_logger(SimLogger.Logger) do
            # Calculate the maximum lengths for alignment
            max_csv_len = maximum(length(isempty(geom.CSVFile) ? "generated particles" : geom.CSVFile)
                                  for geom in SimGeometry) + 2
            max_group_marker_len = maximum(length(string(geom.GroupMarker)) for geom in SimGeometry) + 2
            max_type_len = maximum(length(string(geom.Type)) for geom in SimGeometry) + 2
    
            @info "Simulation SPHGeometry Details:"
            for geom in SimGeometry
                csv_file = isempty(geom.CSVFile) ? "generated particles" : geom.CSVFile
                group_marker = geom.GroupMarker
                particle_type = geom.Type
                motion = if geom.Motion === nothing "None" else string(geom.Motion) end
    
                formatted_csv_file = rpad(csv_file, max_csv_len)
                formatted_group_marker = rpad(string(group_marker), max_group_marker_len)
                formatted_type = rpad(string(particle_type), max_type_len)
                formatted_motion = motion  # No padding necessary if motion detail is to start immediately after type
    
                @info "Particle Source -> $formatted_csv_file, Group Marker -> $formatted_group_marker, Type -> $formatted_type, Motion -> $formatted_motion"
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
            @info "Simulation dimensions: $Dimensions"
            @info "Particle precision: physics $(eltype(SimParticles.Density)); CPU positions $(eltype(eltype(SimParticles.Position)))"
            @info memory_usage_line("CPU particle arrays", host_particle_bytes(SimParticles), total_particles)
            @info "CPU particle memory scope: array payload only; excludes geometry inputs, output/staging buffers, array headers and unused capacity."
    
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
                         densitydiffusion, SPHGeometry, particles; gpu_buffers = nothing)

    Log the configuration, CUDA toolchain and active device, particle counts,
    precision and CPU particle-array payload. `RunSimulation` supplies
    `gpu_buffers = (; particles, support, cells, auxiliary)` for an alias-aware
    device-buffer breakdown after allocation, before the first cell-list rebuild.
    Host-only callers may omit these buffers; GPU memory is then explicitly
    marked as unreported.
    """
    function InitializeLogger(SimLogger,SimConstants,SimMetaData, SimKernel, SimViscosity, SimDensityDiffusion, SimGeometry, SimParticles; gpu_buffers = nothing)
        with_logger(SimLogger.Logger) do
            @info sprint(InteractiveUtils.versioninfo)
            @info "Julia threads: $(Threads.nthreads())"
            if CUDA.functional()
                @info "CUDA information:\n$(sprint(CUDA.versioninfo))"
                dev = CUDA.device()
                @info "GPU: $(CUDA.name(dev)), compute capability $(CUDA.capability(dev)), " *
                      "$(CUDA.totalmem(dev)) bytes total VRAM, CUDA.jl $(pkgversion(CUDA))"
                @info "GPU float type: $(eltype(SimParticles.Density))"
                @info "GPU position type: $(position_float_type(SimMetaData))"
            else
                @info "CUDA information: unavailable (CUDA is not functional)"
            end
            @info SimConstants
            @info SimMetaData
            @info SimKernel
            @info SimViscosity
            @info SimDensityDiffusion
            
            # Print the formatted date and time
            @info "Logger Start Time: " * SimLogger.CurrentDataStr

            LogSimulationDetails(SimLogger, SimGeometry, SimParticles)

            if gpu_buffers === nothing
                @info "GPU buffer memory: not reported (device buffers not supplied)"
            else
                log_gpu_memory(SimLogger, gpu_buffers, length(SimParticles);
                    phase = "startup, before first cell-list rebuild")
            end

            @info SimLogger.ValuesToPrint
            @info SimLogger.ValuesToPrintC
        end
    end


    """
        step_log_line(SimLogger, SimMetaData, HourGlass) -> String

    The log line of one output ("Part_0001 ..."), formatted from the current
    meta data and the time measured so far. Formatting is separate from the
    emission (`log_line`). Progress is emitted by the simulation thread while
    the values belong to the current frame, independently of disk output.
    """
    function step_log_line(SimLogger, SimMetaData, HourGlass)
        PartNumber               = "Part_" * lpad(SimMetaData.OutputIterationCounter, 4, "0")
        PartTime                 = string(@sprintf("%-.6f", SimMetaData.TotalTime))
        PartTotalSteps           = string(SimMetaData.Iteration)
        CurrentSteps             = string(SimMetaData.Iteration - SimMetaData.StepsTakenForLastOutput)

        elapsed_time_            = TimerOutputs.tottime(HourGlass) / 1e9
        TimeUptillNow            = string(@sprintf("%-.3f", elapsed_time_))
        TimePerPhysicalSecond    = string(@sprintf("%-.2f", elapsed_time_ / SimMetaData.TotalTime))

        SecondsToFinish          = (SimMetaData.SimulationTime - SimMetaData.TotalTime) * (elapsed_time_ / SimMetaData.TotalTime)
        if !isfinite(SecondsToFinish)
            ExpectedFinishTimeString = missing
        else
            ExpectedFinishTime       = now() + Second(ceil(Int, SecondsToFinish))
            ExpectedFinishTimeString = Dates.format(ExpectedFinishTime, "dd-mm-yyyy HH:MM:SS")
        end

        return Printf.format(
            Printf.Format(SimLogger.FormatStr),
            PartNumber,
            PartTime,
            PartTotalSteps,
            CurrentSteps,
            TimeUptillNow,
            TimePerPhysicalSecond,
            ExpectedFinishTimeString,
        )
    end

    """
        log_line(SimLogger, line)

    Emit a preformatted line and flush the file and optional console output.
    """
    function log_line(SimLogger, line::AbstractString)
        with_logger(SimLogger.Logger) do
            @info line
        end
        flush(SimLogger.LoggerIo)
        return nothing
    end

    """
        LogStep(SimLogger, SimMetaData, HourGlass)

    Format and emit the log line of the current output (see `step_log_line`).
    """
    LogStep(SimLogger, SimMetaData, HourGlass) = log_line(SimLogger, step_log_line(SimLogger, SimMetaData, HourGlass))
    

    """
        LogFinal(logger, timer; status="finished")

    Print the simulation status, total run time and collected [`TimerOutput`]
    information when output is finalized.
    """
    function LogFinal(SimLogger, HourGlass; status::String = "finished")
        with_logger(SimLogger.Logger) do
            # Get the current date and time
            current_time = now()
            # Format the current date and time
            formatted_time = "\n Simulation $(status) at: " *
                             Dates.format(current_time, "dd-mm-yyyy HH:MM:SS")

            @info formatted_time
            @info "\n Simulation took " * @sprintf("%-.2f", TimerOutputs.tottime(HourGlass)/1e9) * "[s]"
            show(SimLogger.LoggerIo, HourGlass)
        end
    end

end
