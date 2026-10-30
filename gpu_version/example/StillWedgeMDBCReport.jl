module StillWedgeMDBCReport

using Printf: @sprintf
import SPHExampleGPU
using SPHExampleGPU: EquationOfStateGamma7, Fluid, Fixed, Moving, Floating, Frozen
using TimerOutputs
using Typst_jll

export write_still_wedge_report

const REPORT_TEMPLATE = joinpath(@__DIR__, "StillWedgeMDBCReport.typ")
const RUNTIME_METADATA_FIELDS = (
    :Iteration,
    :OutputIterationCounter,
    :StepsTakenForLastOutput,
    :CurrentTimeStep,
    :TotalTime,
    :IndexCounter,
)

_format_number(value) = @sprintf("%.5g", Float64(value))

function _format_bytes(value::Integer)
    value < 1024 && return "$value B"
    value < 1024^2 && return @sprintf("%.2f KiB", value / 1024)
    value < 1024^3 && return @sprintf("%.2f MiB", value / 1024^2)
    return @sprintf("%.2f GiB", value / 1024^3)
end

function _format_value(value)
    value === nothing && return "None"
    value isa Bool && return value ? "true" : "false"
    value isa AbstractFloat && return _format_number(value)
    value isa Integer && return string(value)
    value isa Enum && return string(value)
    value isa Type && return string(value)
    value isa AbstractString && return String(value)
    if value isa AbstractArray
        isempty(value) && return "Empty array of $(eltype(value))"
        if length(value) <= 10
            return "[" * join((_format_value(item) for item in value), ", ") * "]"
        end
        first_value = _format_value(first(value))
        last_value = _format_value(last(value))
        return "Array{$(eltype(value))}, size $(size(value)); from $first_value to $last_value"
    end
    if value isa NamedTuple
        return join(("$key: $(_format_value(item))" for (key, item) in pairs(value)), "; ")
    elseif value isa Tuple
        return "(" * join((_format_value(item) for item in value), ", ") * ")"
    end
    return string(value)
end

function _field_rows(component; excluded_fields = ())
    rows = Tuple{String, String}[("Component type", string(typeof(component)))]
    for field in fieldnames(typeof(component))
        if field === :HourGlass
            push!(rows, ("HourGlass", "Detailed phase timings are shown below"))
        elseif field in excluded_fields
            continue
        else
            push!(rows, (string(field), _format_value(getfield(component, field))))
        end
    end
    return rows
end

function _flatten_rows!(rows, prefix::String, value)
    if value isa NamedTuple
        if isempty(value)
            push!(rows, (prefix, "None"))
        else
            for (name, nested) in pairs(value)
                _flatten_rows!(rows, "$prefix / $name", nested)
            end
        end
    elseif value isa Tuple && all(item -> item isa NamedTuple, value)
        if isempty(value)
            push!(rows, (prefix, "None"))
        else
            for (index, nested) in enumerate(value)
                _flatten_rows!(rows, "$prefix $(index)", nested)
            end
        end
    elseif value isa AbstractDict
        if isempty(value)
            push!(rows, (prefix, "None"))
        else
            for key in sort!(collect(keys(value)); by = string)
                _flatten_rows!(rows, "$prefix / $key", value[key])
            end
        end
    else
        push!(rows, (prefix, _format_value(value)))
    end
    return rows
end

function _configuration_rows(configuration, section::Symbol)
    if configuration === nothing || !hasproperty(configuration, section)
        return [("Configuration", "Not recorded")]
    end
    rows = Tuple{String, String}[]
    _flatten_rows!(rows, string(section), getproperty(configuration, section))
    return rows
end

function _component_description(type)
    type == Fluid && return "Fluid particles advanced by the solver."
    type == Fixed && return "Stationary boundary particles defining the domain."
    type == Moving && return "Prescribed moving boundary particles."
    type == Floating && return "Particles belonging to a force-responsive rigid body."
    type == Frozen && return "Particles frozen after crossing a locked grid limit."
    return "Particle group represented in the simulation geometry."
end

function _particle_type_rows(geometry)
    counts = Dict{String, Int}()
    type_names = Dict{String, Any}()
    for group in geometry
        type_name = string(group.Type)
        counts[type_name] = get(counts, type_name, 0) + length(group.Particles)
        type_names[type_name] = group.Type
    end
    order = (Fluid, Fixed, Moving, Floating, Frozen)
    rows = Tuple{String, String, String}[]
    for type in order
        type_name = string(type)
        haskey(counts, type_name) || continue
        push!(rows, (type_name, string(counts[type_name]), _component_description(type_names[type_name])))
    end
    return rows
end

function _geometry_rows(geometry, particles)
    rows = Tuple{String, String, String, String, String}[]
    for group in geometry
        count = sum(particles.GroupMarker .== group.GroupMarker)
        source = isempty(group.CSVFile) ? "Generated in memory" : group.CSVFile
        motion = group.Motion === nothing ? "None" : _format_value(group.Motion)
        floating = hasproperty(group, :Floating) ? group.Floating : nothing
        if floating !== nothing
            motion = motion == "None" ? _format_value(floating) : "$motion; $(_format_value(floating))"
        end
        push!(rows, (
            string(group.GroupMarker),
            string(group.Type),
            string(count),
            source,
            motion,
        ))
    end
    return rows
end

function _split_log_line(line; fallback = "Details")
    fields = split(line, ": "; limit = 2)
    return length(fields) == 2 ? (strip(fields[1]), strip(fields[2])) : (fallback, line)
end

function _environment_rows(log_text)
    rows = Tuple{String, String}[]
    reading_julia = true
    reading_cuda = false

    for raw_line in eachsplit(log_text, '\n')
        line = strip(raw_line)
        isempty(line) && continue

        if reading_julia
            if startswith(line, "Julia threads:")
                label, value = _split_log_line(line; fallback = "Runtime")
                push!(rows, (label, value))
                reading_julia = false
            else
                label, value = _split_log_line(line; fallback = "Runtime detail")
                label == "Runtime detail" && startswith(line, "Julia Version") && (label = "Julia version")
                push!(rows, ("Julia / $label", value))
            end
        elseif startswith(line, "CUDA information:")
            details = strip(replace(line, "CUDA information:" => ""))
            if isempty(details)
                reading_cuda = true
            else
                push!(rows, ("CUDA information", details))
            end
        elseif reading_cuda && startswith(line, "GPU:")
            label, value = _split_log_line(line; fallback = "Device")
            push!(rows, ("GPU device", value))
            reading_cuda = false
        elseif reading_cuda && startswith(line, "SimulationConstants")
            reading_cuda = false
        elseif reading_cuda
            label, value = _split_log_line(line; fallback = "CUDA detail")
            push!(rows, ("CUDA / $label", value))
        elseif startswith(line, "GPU:")
            label, value = _split_log_line(line; fallback = "Device")
            push!(rows, ("GPU device", value))
        elseif startswith(line, "GPU float type:") ||
               startswith(line, "GPU position type:") ||
               startswith(line, "CUDA information:")
            label, value = _split_log_line(line)
            push!(rows, (label, value))
        elseif startswith(line, "Logger Start Time:")
            label, value = _split_log_line(line)
            push!(rows, (label, value))
        end
    end
    isempty(rows) && push!(rows, ("Runtime", "Environment details were not recorded"))
    return rows
end

function _progress_rows(log_text)
    rows = NTuple{4, String}[]
    for line in eachsplit(log_text, '\n')
        fields = split(strip(line); limit = 7)
        if length(fields) == 7 && startswith(fields[1], "Part_")
            push!(rows, (
                "$(fields[1]) / $(fields[2]) s",
                "$(fields[3]) total / $(fields[4]) interval",
                "$(fields[5]) s / $(fields[6]) s per simulated second",
                String(fields[7]),
            ))
        end
    end
    isempty(rows) && push!(rows, ("No output rows", "Not recorded", "Not recorded", "Not recorded"))
    return rows
end

function _memory_rows(log_text)
    rows = NTuple{3, String}[]
    phase = "GPU"
    prefixes = (
        "GPU particle arrays",
        "GPU support arrays",
        "GPU cell-list workspace",
        "GPU auxiliary arrays",
        "GPU accounted buffers total",
        "CUDA pool:",
        "Output staging & buffer capacity:",
        "GPU memory scope:",
    )

    for raw_line in eachsplit(log_text, '\n')
        line = strip(raw_line)
        match_phase = match(r"^GPU live-buffer memory \((.*)\):$", line)
        if match_phase !== nothing
            phase = match_phase.captures[1]
        elseif startswith(line, "CPU particle arrays:") ||
               startswith(line, "CPU particle memory scope:")
            label, value = _split_log_line(line)
            push!(rows, ("Host", label, value))
        elseif any(prefix -> startswith(line, prefix), prefixes)
            label, value = _split_log_line(line)
            push!(rows, (phase, label, value))
        elseif startswith(line, "GPU buffer memory:")
            label, value = _split_log_line(line)
            push!(rows, ("GPU", label, value))
        end
    end

    isempty(rows) && push!(rows, ("Not reported", "GPU memory", "No memory details were logged"))
    return rows
end

function _notice_rows(log_text)
    rows = Tuple{String, String}[]
    keywords = ("warning", "unsupported", "fallback", "froze", "frozen", "interrupted", "failed", "skipping")
    for raw_line in eachsplit(log_text, '\n')
        line = strip(raw_line)
        isempty(line) && continue
        lower_line = lowercase(line)
        any(keyword -> occursin(keyword, lower_line), keywords) &&
            push!(rows, ("Notice", line))
    end
    isempty(rows) && push!(rows, ("Notices", "No warnings or fallback notices were logged"))
    return rows
end

function _finish_time(log_text)
    for raw_line in eachsplit(log_text, '\n')
        line = strip(raw_line)
        match_time = match(r"^Simulation (?:finished|interrupted|failed) at:\s*(.*)$", line)
        match_time === nothing || return match_time.captures[1]
    end
    return "Not recorded"
end

function _logged_line(log_text, prefix; fallback = "Not recorded")
    for raw_line in eachsplit(log_text, '\n')
        line = strip(raw_line)
        startswith(line, prefix) && return line
    end
    return fallback
end

function _timing_rows(timer)
    rows = NTuple{4, String}[
        (
            "All measured sections",
            "—",
            _format_number(TimerOutputs.tottime(timer) / 1e9),
            _format_bytes(TimerOutputs.totallocated(timer)),
        ),
    ]

    function append_children!(parent, path)
        for name in keys(parent)
            section = parent[name]
            section_path = isempty(path) ? name : "$path / $name"
            push!(rows, (
                section_path,
                string(TimerOutputs.ncalls(section)),
                _format_number(TimerOutputs.time(section) / 1e9),
                _format_bytes(TimerOutputs.allocated(section)),
            ))
            append_children!(section, section_path)
        end
        return nothing
    end

    append_children!(timer, "")
    return rows
end

function _typst_string(value)
    text = string(value)
    return replace(
        text,
        "\\" => "\\\\",
        "\"" => "\\\"",
        "\r" => " ",
        "\n" => " ",
        "\t" => " ",
    )
end

_typst_cell(value) = "[#text(\"$(_typst_string(value))\")]"

function _typst_table_rows(rows; columns::Int = 2)
    isempty(rows) && return _typst_table_rows([ntuple(_ -> "Not recorded", columns)]; columns)
    rendered = String[]
    for row in rows
        length(row) == columns ||
            throw(DimensionMismatch("Typst table expected $columns cells, got $(length(row))"))
        push!(rendered, "  " * join((_typst_cell(value) for value in row), ", ") * ",")
    end
    return join(rendered, "\n")
end

function _render_template(report)
    source = read(REPORT_TEMPLATE, String)
    for (key, value) in pairs(report)
        placeholder = "{{$(uppercase(string(key)))}}"
        occursin(placeholder, source) ||
            throw(ArgumentError("StillWedge Typst report template is missing $placeholder"))
        rendered = endswith(string(key), "_rows") ? string(value) : _typst_string(value)
        source = replace(source, placeholder => rendered)
    end
    occursin("{{", source) &&
        throw(ArgumentError("StillWedge Typst report template has an unresolved placeholder"))
    return source
end

"""
    write_still_wedge_report(result, constants; output_dir, simulation_time,
        metadata, geometry, kernel, viscosity, density_diffusion,
        simulation_name = "StillWedge", compile_pdf = true)

Create a structured Typst report with colored summary cards, parameter tables,
particle and geometry counts, effective GPU settings, memory accounting,
progress history and phase timings. The logger output is parsed into these
sections; it is not copied into the report as a raw appendix.
"""
function write_still_wedge_report(
    result,
    constants;
    output_dir::AbstractString,
    simulation_time::Real,
    metadata,
    geometry,
    kernel,
    viscosity,
    density_diffusion,
    simulation_name::AbstractString = "StillWedge",
    compile_pdf::Bool = true,
)
    log_path = joinpath(output_dir, "SimulationOutput.log")
    isfile(log_path) ||
        throw(ArgumentError("SimulationOutput.log was not found at '$log_path'"))
    log_text = read(log_path, String)
    isempty(strip(log_text)) &&
        throw(ArgumentError("SimulationOutput.log is empty at '$log_path'"))

    particles = result.final_state.particles
    fluid_indices = findall(==(Fluid), particles.Type)
    isempty(fluid_indices) &&
        throw(ArgumentError("StillWedge report requires at least one fluid particle"))

    density = particles.Density[fluid_indices]
    pressure = EquationOfStateGamma7.(density, constants.c₀, constants.ρ₀)
    speed = [
        sqrt(sum(abs2, velocity)) for
        velocity in particles.Velocity[fluid_indices]
    ]
    variable_rows = [
        ("Density (kg/m^3)", _format_number(minimum(density)), _format_number(sum(density) / length(density)), _format_number(maximum(density))),
        ("Pressure (Pa)", _format_number(minimum(pressure)), _format_number(sum(pressure) / length(pressure)), _format_number(maximum(pressure))),
        ("Speed (m/s)", _format_number(minimum(speed)), _format_number(sum(speed) / length(speed)), _format_number(maximum(speed))),
    ]

    completion = result.final_state.complete ? "Completed" : "Partial"
    execution_path = result.configuration === nothing ||
                     !hasproperty(result.configuration, :execution) ?
                     "Not recorded" :
                     _format_value(result.configuration.execution.path)
    performance_rows = [
        ("Time advanced in this run", "$(_format_number(result.performance.simulated_time)) s"),
        ("Mean timestep", "$(_format_number(result.performance.mean_timestep)) s"),
        ("Wall time", "$(_format_number(result.performance.wall_time_seconds)) s"),
        ("Milliseconds per step", "$(_format_number(result.performance.milliseconds_per_step)) ms"),
        ("Throughput", "$(_format_number(result.performance.mparticle_steps_per_second)) million particle-steps/s"),
        ("Output writer time", "$(_format_number(result.performance.output_write_seconds)) s"),
        ("Output writer mode and flush size", _logged_line(log_text, "Output frame write time")),
        ("Phase timers synchronized", _format_value(result.performance.phase_timings_synchronized)),
        ("Observed execution path", execution_path),
        ("Completed at", _finish_time(log_text)),
    ]
    diagnostic_rows = [
        ("Particle count", _format_value(result.diagnostics.particle_count)),
        ("Output frames", _format_value(result.diagnostics.output_frames)),
        ("Cell-list rebuilds", _format_value(result.diagnostics.cell_list_rebuilds)),
        ("Grid dimensions", join(result.diagnostics.grid_dims, " x ")),
        ("Frozen particles", _format_value(result.diagnostics.frozen_particles)),
        ("Step-state readbacks", _format_value(result.diagnostics.step_state_readbacks)),
        ("Grid-status readbacks", _format_value(result.diagnostics.grid_status_readbacks)),
        ("Capacity growths", _format_value(result.diagnostics.capacity_growths)),
        ("Bounding-box host partial reads", _format_value(result.diagnostics.bbox_host_readbacks)),
        ("Captured step graphs", _format_value(result.diagnostics.captured_graphs)),
        ("Periodic source rebuilds", _format_value(result.diagnostics.periodic_source_rebuilds)),
        ("Periodic source count", _format_value(result.diagnostics.periodic_source_count)),
        ("Simulation dimensions", string(length(first(particles.Position)))),
        (
            "Particle precision",
            "Physics $(eltype(particles.Density)); CPU positions $(eltype(eltype(particles.Position)))",
        ),
        ("Output iteration counter", _format_value(metadata.OutputIterationCounter)),
        ("Steps since last output", _format_value(metadata.StepsTakenForLastOutput)),
        ("Index counter", _format_value(metadata.IndexCounter)),
    ]
    summary_rows = [
        ("Requested simulation duration", "$(_format_number(simulation_time)) s"),
        ("Final solver time", "$(_format_number(result.final_state.time)) s"),
        ("Termination reason", string(result.termination_reason)),
        ("Steps in this run", _format_value(result.performance.steps)),
        ("Final solver iteration", _format_value(result.final_state.iteration)),
        ("Final timestep", "$(_format_number(result.final_state.timestep)) s"),
        ("Total particles", _format_value(result.diagnostics.particle_count)),
        ("Fluid particles", string(length(fluid_indices))),
        ("Output frames", _format_value(result.diagnostics.output_frames)),
    ]

    metadata_rows = _field_rows(
        metadata;
        excluded_fields = RUNTIME_METADATA_FIELDS,
    )
    report = (
        simulation_name = simulation_name,
        completion,
        final_time = _format_number(result.final_state.time),
        steps = _format_value(result.performance.steps),
        total_particles = _format_value(result.diagnostics.particle_count),
        fluid_particles = string(length(fluid_indices)),
        wall_time = _format_number(result.performance.wall_time_seconds),
        summary_rows = _typst_table_rows(summary_rows),
        variable_rows = _typst_table_rows(variable_rows; columns = 4),
        environment_rows = _typst_table_rows(_environment_rows(log_text)),
        constants_rows = _typst_table_rows(_field_rows(constants)),
        metadata_rows = _typst_table_rows(metadata_rows),
        geometry_rows = _typst_table_rows(_geometry_rows(geometry, particles); columns = 5),
        particle_type_rows = _typst_table_rows(_particle_type_rows(geometry); columns = 3),
        kernel_rows = _typst_table_rows(_field_rows(kernel)),
        viscosity_rows = _typst_table_rows(_field_rows(viscosity)),
        diffusion_rows = _typst_table_rows(_field_rows(density_diffusion)),
        requested_rows = _typst_table_rows(_configuration_rows(result.configuration, :requested)),
        capability_rows = _typst_table_rows(_configuration_rows(result.configuration, :capabilities)),
        effective_rows = _typst_table_rows(_configuration_rows(result.configuration, :effective)),
        fallback_rows = _typst_table_rows(_configuration_rows(result.configuration, :fallbacks)),
        execution_rows = _typst_table_rows(_configuration_rows(result.configuration, :execution)),
        performance_rows = _typst_table_rows(performance_rows),
        diagnostic_rows = _typst_table_rows(diagnostic_rows),
        memory_rows = _typst_table_rows(_memory_rows(log_text); columns = 3),
        notice_rows = _typst_table_rows(_notice_rows(log_text)),
        timing_rows = _typst_table_rows(_timing_rows(result.performance.phase_timings); columns = 4),
        progress_rows = _typst_table_rows(_progress_rows(log_text); columns = 4),
        timing_note = result.performance.phase_timings_synchronized ?
                      "Phase timers were synchronized with GPU work." :
                      "GPUSyncTimers is false: phase timings show host enqueue and readback boundaries, not per-kernel GPU execution time.",
    )

    mkpath(output_dir)
    typ_path = joinpath(output_dir, "StillWedge_report.typ")
    pdf_path = joinpath(output_dir, "StillWedge_report.pdf")
    write(typ_path, _render_template(report))

    if compile_pdf
        run(`$(Typst_jll.typst()) compile $typ_path $pdf_path`)
    end

    return (; typ_path, pdf_path = compile_pdf ? pdf_path : nothing)
end

end # module StillWedgeMDBCReport
