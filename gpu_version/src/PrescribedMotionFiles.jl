# Deliberately small, strict numeric file reader: no dependency changes and no
# heuristic header/column detection that could silently consume a wrong file.
function load_motion_table(path; time_column, value_columns, skip_rows)
    cols = (time_column, value_columns...)
    all(c -> c isa Integer && c > 0, cols) && length(unique(cols)) == length(cols) ||
        throw(ArgumentError("file columns must be distinct positive one-based indices"))
    skip_rows isa Integer && skip_rows >= 0 || throw(ArgumentError("skip_rows must be nonnegative"))
    times, rows = Float64[], Vector{Float64}[]
    width = nothing
    open(path) do io
        for (line_number, raw) in enumerate(eachline(io))
            line_number <= skip_rows && continue
            line = strip(first(split(raw, '#'; limit = 2)))
            isempty(line) && continue
            fields = split(line, r"[\s,;]+")
            width === nothing && (width = length(fields))
            length(fields) == width && maximum(cols) <= width ||
                throw(ArgumentError("motion file $path line $line_number has an invalid column count"))
            values = tryparse.(Float64, fields)
            all(v -> v !== nothing && isfinite(v), values) ||
                throw(ArgumentError("motion file $path line $line_number contains nonnumeric or nonfinite data"))
            push!(times, values[time_column])
            push!(rows, Float64[values[c] for c in value_columns])
        end
    end
    MotionTrajectories.checked_times(times)
    return times, rows
end

"""
    load_translation_trajectory(path; value_columns = (2,3), time_column = 1,
                                start_time = 0, relative = true, skip_rows = 0)
    load_rotation_trajectory(path; angle_column = 2, time_column = 1,
                             angle_units = :radians, start_time = 0,
                             relative = true, skip_rows = 0)

Read numeric whitespace/comma/semicolon-separated data, with optional `#`
comments. Column indices are one-based; select two/three position columns in
solver coordinate order (e.g. x,z for 2D). Headers require explicit `skip_rows`.
All rows must have the same width and finite numeric fields. Times must be
nonnegative and strictly increasing, with at least two rows.

File time is rebased to `start_time`. By default subtract the first position
or angle, matching DualSPHysics file-motion displacement semantics. Set
`relative = false` to retain absolute displacement/angle values. Angles accept
`:degrees` or `:radians`, are converted to radians and never wrapped. Both
loaders use linear interpolation and hold endpoints with zero rates outside
the active interval. Files are read once, before CUDA graph capture.
"""
function load_translation_trajectory(path; value_columns = (2,3), time_column = 1,
        start_time = 0, relative::Bool = true, skip_rows = 0)
    length(value_columns) in (2,3) || throw(DimensionMismatch("translation files need two or three selected value columns"))
    ts, rows = load_motion_table(path; time_column, value_columns, skip_rows)
    values = motion_value.(rows)
    relative && (values = values .- Ref(first(values)))
    return SampledTrajectory(ts .- first(ts) .+ start_time, values)
end

function load_rotation_trajectory(path; angle_column = 2, time_column = 1,
        angle_units::Symbol = :radians, start_time = 0, relative::Bool = true, skip_rows = 0)
    angle_units in (:degrees, :radians) || throw(ArgumentError("angle_units must be :degrees or :radians"))
    ts, rows = load_motion_table(path; time_column, value_columns = (angle_column,), skip_rows)
    values = first.(rows)
    relative && (values = values .- first(values))
    angle_units == :degrees && (values = deg2rad.(values))
    return SampledTrajectory(ts .- first(ts) .+ start_time, values)
end

@doc (@doc load_translation_trajectory) load_rotation_trajectory
