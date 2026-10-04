module SPHMeasurements

using HDF5

export MeasurementProbe, WaterColumnProbe, FreeSurfaceDomain, MeasurementConfig

"""
    MeasurementProbe(name, location)

Describe a pressure or velocity probe. `location` has one coordinate per
simulation dimension. Values are sampled from the nearest fluid particle.
"""
struct MeasurementProbe
    name::String
    location::Tuple{Vararg{Float64}}
    function MeasurementProbe(name::AbstractString, location)
        return new(_measurement_name(name), _coordinates(location))
    end
end

"""
    WaterColumnProbe(name, location; radius = nothing)

Measure the fluid height above `location`'s vertical coordinate. The optional
`radius` selects particles within this horizontal distance; by default the
kernel support radius is used. The result is zero when the surface lies below
the probe and `NaN` when no fluid particle is found within the radius.
"""
struct WaterColumnProbe
    name::String
    location::Tuple{Vararg{Float64}}
    radius::Union{Nothing, Float64}
    function WaterColumnProbe(name::AbstractString, location; radius = nothing)
        column_radius = radius === nothing ? nothing : Float64(radius)
        if column_radius !== nothing &&
           (!isfinite(column_radius) || column_radius < 0)
            throw(ArgumentError("water-column radius must be finite and nonnegative"))
        end
        return new(
            _measurement_name(name),
            _coordinates(location),
            column_radius,
        )
    end
end

"""
    FreeSurfaceDomain(lower, upper, spacing)

Track the highest fluid particle in each horizontal bin of the domain.
`lower` and `upper` have one coordinate per simulation dimension; the
vertical-axis bounds are ignored.
"""
struct FreeSurfaceDomain
    lower::Tuple{Vararg{Float64}}
    upper::Tuple{Vararg{Float64}}
    spacing::Float64
    function FreeSurfaceDomain(lower, upper, spacing::Real)
        lo = _coordinates(lower)
        hi = _coordinates(upper)
        length(lo) == length(hi) ||
            throw(ArgumentError("free-surface bounds must have matching dimensions"))
        step = Float64(spacing)
        isfinite(step) && step > 0 ||
            throw(ArgumentError("free-surface spacing must be finite and positive"))
        return new(lo, hi, step)
    end
end

"""
    MeasurementConfig(; pressure_probes = [], velocity_probes = [],
                        water_column_probes = [], free_surface = nothing,
                        vertical_axis = nothing, sample_every = 1)

Select the measurements to record. Pressure and velocity use
`MeasurementProbe`s, water heights use `WaterColumnProbe`s, and free-surface
tracking is enabled by passing a `FreeSurfaceDomain`. Measurements are sampled
at the initial frame and then every `sample_every` output frames.

The default vertical axis is the last simulation axis (axis 2 in 2D and axis 3
in 3D). Only requested measurement groups are written.
"""
struct MeasurementConfig
    pressure_probes::Vector{MeasurementProbe}
    velocity_probes::Vector{MeasurementProbe}
    water_column_probes::Vector{WaterColumnProbe}
    free_surface::Union{Nothing, FreeSurfaceDomain}
    vertical_axis::Union{Nothing, Int}
    sample_every::Int
    function MeasurementConfig(
            pressure_probes::Vector{MeasurementProbe},
            velocity_probes::Vector{MeasurementProbe},
            water_column_probes::Vector{WaterColumnProbe},
            free_surface::Union{Nothing, FreeSurfaceDomain},
            vertical_axis::Union{Nothing, Int},
            sample_every::Int,
        )
        sample_every >= 1 ||
            throw(ArgumentError("sample_every must be at least 1"))
        vertical_axis === nothing || vertical_axis >= 1 ||
            throw(ArgumentError("vertical_axis must be positive"))
        return new(
            pressure_probes, velocity_probes, water_column_probes,
            free_surface, vertical_axis, sample_every,
        )
    end
end

function MeasurementConfig(;
        pressure_probes = MeasurementProbe[],
        velocity_probes = MeasurementProbe[],
        water_column_probes = WaterColumnProbe[],
        free_surface = nothing,
        vertical_axis = nothing,
        sample_every::Integer = 1,
    )
    sample_every >= 1 ||
        throw(ArgumentError("sample_every must be at least 1"))
    axis = if vertical_axis === nothing
        nothing
    elseif vertical_axis isa Integer && !(vertical_axis isa Bool) && vertical_axis >= 1
        Int(vertical_axis)
    else
        throw(ArgumentError("vertical_axis must be a positive integer or nothing"))
    end
    domain = if free_surface === nothing || free_surface isa FreeSurfaceDomain
        free_surface
    else
        throw(ArgumentError("free_surface must be a FreeSurfaceDomain or nothing"))
    end

    pressure = _measurement_list(
        MeasurementProbe, pressure_probes, "pressure_probes",
    )
    velocity = _measurement_list(
        MeasurementProbe, velocity_probes, "velocity_probes",
    )
    columns = _measurement_list(
        WaterColumnProbe, water_column_probes, "water_column_probes",
    )
    return MeasurementConfig(
        pressure, velocity, columns, domain, axis, Int(sample_every),
    )
end

struct ResolvedMeasurements
    dimension::Int
    vertical_axis::Int
    sample_every::Int
    pressure_names::Vector{String}
    pressure_locations::Matrix{Float64}
    velocity_names::Vector{String}
    velocity_locations::Matrix{Float64}
    column_names::Vector{String}
    column_locations::Matrix{Float64}
    column_radii::Vector{Float64}
    surface_shape::Vector{Int}
    surface_axes::Vector{Int}
    surface_strides::Vector{Int}
    surface_lower::Vector{Float64}
    surface_upper::Vector{Float64}
    surface_spacing::Float64
    surface_locations::Matrix{Float64}
end

struct ScalarSeries
    dataset::HDF5.Dataset
    buffer::Matrix{Float64}
end

struct VectorSeries
    dataset::HDF5.Dataset
    buffer::Array{Float64, 3}
end

mutable struct MeasurementWriter{F}
    plan::ResolvedMeasurements
    fluid_type::F
    time_dataset::HDF5.Dataset
    time_buffer::Vector{Float64}
    pressure::Union{Nothing, ScalarSeries}
    velocity::Union{Nothing, VectorSeries}
    column_height::Union{Nothing, ScalarSeries}
    surface_height::Union{Nothing, ScalarSeries}
    pressure_nearest::Vector{Int}
    pressure_distance²::Vector{Float64}
    velocity_nearest::Vector{Int}
    velocity_distance²::Vector{Float64}
    column_maximum::Vector{Float64}
    surface_maximum::Vector{Float64}
    capacity::Int
    pending::Int
    frames_written::Int
    output_frames_seen::Int
end

function _measurement_name(name::AbstractString)
    value = String(name)
    isempty(strip(value)) && throw(ArgumentError("measurement names cannot be empty"))
    occursin('\n', value) &&
        throw(ArgumentError("measurement names cannot contain newlines"))
    occursin('\r', value) &&
        throw(ArgumentError("measurement names cannot contain newlines"))
    return value
end

function _coordinates(values)
    (values isa Tuple || values isa AbstractVector) ||
        throw(ArgumentError("measurement locations must be tuples or vectors"))
    coordinates = Tuple(Float64(value) for value in values)
    length(coordinates) in (2, 3) ||
        throw(ArgumentError("measurement locations must have 2 or 3 coordinates"))
    all(isfinite, coordinates) ||
        throw(ArgumentError("measurement locations must be finite"))
    return coordinates
end

function _measurement_list(::Type{T}, values, label) where {T}
    values === nothing && return T[]
    result = T[]
    for value in values
        value isa T ||
            throw(ArgumentError("$label must contain only $(T) values"))
        push!(result, value)
    end
    names = Set{String}()
    for value in result
        value.name in names &&
            throw(ArgumentError("$label contains duplicate name `$(value.name)`"))
        push!(names, value.name)
    end
    return result
end

has_measurements(config::MeasurementConfig) =
    !isempty(config.pressure_probes) ||
    !isempty(config.velocity_probes) ||
    !isempty(config.water_column_probes) ||
    config.free_surface !== nothing

function required_particle_fields(config::Union{Nothing, MeasurementConfig})
    config === nothing && return Symbol[]
    has_measurements(config) || return Symbol[]

    fields = Symbol[:Type]
    isempty(config.pressure_probes) || push!(fields, :Pressure)
    isempty(config.velocity_probes) || push!(fields, :Velocity)
    return fields
end

function _locations(probes, dimension)
    locations = Matrix{Float64}(undef, dimension, length(probes))
    for (index, probe) in enumerate(probes)
        length(probe.location) == dimension ||
            throw(ArgumentError(
                "measurement `$(probe.name)` has $(length(probe.location)) " *
                "coordinates; expected $dimension",
            ))
        for axis in 1:dimension
            locations[axis, index] = probe.location[axis]
        end
    end
    return locations
end

function resolve_measurements(
        config::MeasurementConfig, dimension::Integer, default_column_radius::Real,
    )
    dimension in (2, 3) ||
        throw(ArgumentError("measurements support 2D and 3D simulations"))
    vertical_axis = something(config.vertical_axis, Int(dimension))
    1 <= vertical_axis <= dimension ||
        throw(ArgumentError(
            "vertical_axis must be between 1 and $dimension",
        ))
    default_radius = Float64(default_column_radius)
    isfinite(default_radius) && default_radius > 0 ||
        throw(ArgumentError("kernel support radius must be finite and positive"))

    pressure_locations = _locations(config.pressure_probes, dimension)
    velocity_locations = _locations(config.velocity_probes, dimension)
    column_locations = _locations(config.water_column_probes, dimension)
    column_radii = Float64[
        something(probe.radius, default_radius)
        for probe in config.water_column_probes
    ]

    surface_shape = Int[]
    surface_axes = Int[]
    surface_lower = zeros(Float64, dimension)
    surface_upper = zeros(Float64, dimension)
    surface_spacing = 0.0
    surface_locations = Matrix{Float64}(undef, max(dimension - 1, 0), 0)
    if config.free_surface !== nothing
        domain = config.free_surface
        length(domain.lower) == dimension ||
            throw(ArgumentError(
                "free-surface bounds have $(length(domain.lower)) coordinates; " *
                "expected $dimension",
            ))
        surface_lower .= domain.lower
        surface_upper .= domain.upper
        surface_spacing = domain.spacing
        for axis in 1:dimension
            axis == vertical_axis && continue
            upper = surface_upper[axis]
            lower = surface_lower[axis]
            upper > lower ||
                throw(ArgumentError(
                    "free-surface upper bound must exceed lower bound on axis $axis",
                ))
            push!(surface_axes, axis)
            push!(surface_shape, ceil(Int, (upper - lower) / surface_spacing))
        end

        number_of_bins = foldl(Base.checked_mul, surface_shape; init = 1)
        surface_locations = Matrix{Float64}(
            undef, length(surface_axes), number_of_bins,
        )
        for linear_index in 1:number_of_bins
            remaining = linear_index - 1
            for (horizontal_axis, axis) in enumerate(surface_axes)
                bin_index = rem(remaining, surface_shape[horizontal_axis]) + 1
                remaining = div(remaining, surface_shape[horizontal_axis])
                bin_lower = surface_lower[axis] +
                            (bin_index - 1) * surface_spacing
                bin_upper = min(
                    bin_lower + surface_spacing, surface_upper[axis],
                )
                surface_locations[horizontal_axis, linear_index] =
                    (bin_lower + bin_upper) / 2
            end
        end
    end

    return ResolvedMeasurements(
        Int(dimension),
        vertical_axis,
        config.sample_every,
        [probe.name for probe in config.pressure_probes],
        pressure_locations,
        [probe.name for probe in config.velocity_probes],
        velocity_locations,
        [probe.name for probe in config.water_column_probes],
        column_locations,
        column_radii,
        surface_shape,
        surface_axes,
        _strides(surface_shape),
        surface_lower,
        surface_upper,
        surface_spacing,
        surface_locations,
    )
end

function _strides(shape)
    strides = Int[]
    stride = 1
    for bins in shape
        push!(strides, stride)
        stride *= bins
    end
    return strides
end

function _scalar_series(group, count, capacity)
    dataset = HDF5.create_dataset(
        group,
        "Values",
        Float64,
        ((count, 0), (count, -1)),
        chunk = (count, capacity),
    )
    return ScalarSeries(dataset, Matrix{Float64}(undef, count, capacity))
end

function _scalar_group(root, name, locations, names, capacity)
    group = HDF5.create_group(root, name)
    try
        group["Locations"] = locations
        HDF5.attrs(group)["Names"] = join(names, "\n")
        return _scalar_series(group, length(names), capacity)
    finally
        close(group)
    end
end

function _vector_group(root, name, locations, names, dimension, capacity)
    group = HDF5.create_group(root, name)
    try
        group["Locations"] = locations
        HDF5.attrs(group)["Names"] = join(names, "\n")
        HDF5.attrs(group)["Components"] = Int32(dimension)
        dataset = HDF5.create_dataset(
            group,
            "Values",
            Float64,
            ((dimension, length(names), 0), (dimension, length(names), -1)),
            chunk = (dimension, length(names), capacity),
        )
        buffer = Array{Float64, 3}(
            undef, dimension, length(names), capacity,
        )
        return VectorSeries(dataset, buffer)
    finally
        close(group)
    end
end

function MeasurementWriter(file::HDF5.File, plan::ResolvedMeasurements, fluid_type;
                           capacity::Integer = 4)
    capacity >= 1 ||
        throw(ArgumentError("measurement buffer capacity must be at least 1"))
    group = HDF5.create_group(file, "Measurements")
    time_dataset = nothing
    pressure = nothing
    velocity = nothing
    column_height = nothing
    surface_height = nothing
    try
        HDF5.attrs(group)["Version"] = Int32(1)
        HDF5.attrs(group)["CoordinateDimension"] = Int32(plan.dimension)
        HDF5.attrs(group)["VerticalAxis"] = Int32(plan.vertical_axis)
        HDF5.attrs(group)["SampleEveryOutputFrames"] = Int32(plan.sample_every)
        time_dataset = HDF5.create_dataset(
            group, "Time", Float64, ((0,), (-1,)), chunk = (capacity,),
        )

        if !isempty(plan.pressure_names)
            pressure = _scalar_group(
                group, "Pressure", plan.pressure_locations,
                plan.pressure_names, capacity,
            )
            pressure_group = group["Pressure"]
            try
                HDF5.attrs(pressure_group)["Sampling"] =
                    "nearest fluid particle"
            finally
                close(pressure_group)
            end
        end
        if !isempty(plan.velocity_names)
            velocity = _vector_group(
                group, "Velocity", plan.velocity_locations,
                plan.velocity_names, plan.dimension, capacity,
            )
            velocity_group = group["Velocity"]
            try
                HDF5.attrs(velocity_group)["Sampling"] =
                    "nearest fluid particle"
            finally
                close(velocity_group)
            end
        end
        if !isempty(plan.column_names)
            column_height = _scalar_group(
                group, "WaterColumnHeight", plan.column_locations,
                plan.column_names, capacity,
            )
            column_group = group["WaterColumnHeight"]
            try
                column_group["Radii"] = plan.column_radii
                HDF5.attrs(column_group)["Definition"] =
                    "maximum fluid-particle height minus probe height"
            finally
                close(column_group)
            end
        end
        if !isempty(plan.surface_axes)
            surface_group = HDF5.create_group(group, "FreeSurface")
            try
                surface_group["Locations"] = plan.surface_locations
                surface_group["GridShape"] = Int64.(plan.surface_shape)
                surface_group["HorizontalAxes"] = Int32.(plan.surface_axes)
                HDF5.attrs(surface_group)["Spacing"] = plan.surface_spacing
                HDF5.attrs(surface_group)["Ordering"] =
                    "first horizontal axis varies fastest"
                HDF5.attrs(surface_group)["Definition"] =
                    "maximum fluid-particle height in each horizontal bin"
                surface_height = _scalar_series(
                    surface_group, size(plan.surface_locations, 2), capacity,
                )
            finally
                close(surface_group)
            end
        end
    finally
        close(group)
    end

    return MeasurementWriter(
        plan,
        fluid_type,
        time_dataset,
        Vector{Float64}(undef, capacity),
        pressure,
        velocity,
        column_height,
        surface_height,
        zeros(Int, length(plan.pressure_names)),
        Vector{Float64}(undef, length(plan.pressure_names)),
        zeros(Int, length(plan.velocity_names)),
        Vector{Float64}(undef, length(plan.velocity_names)),
        Vector{Float64}(undef, length(plan.column_names)),
        Vector{Float64}(undef, size(plan.surface_locations, 2)),
        Int(capacity),
        0,
        0,
        0,
    )
end

function _nearest_indices!(writer, positions, particle_types)
    fill!(writer.pressure_nearest, 0)
    fill!(writer.pressure_distance², Inf)
    fill!(writer.velocity_nearest, 0)
    fill!(writer.velocity_distance², Inf)
    fill!(writer.column_maximum, -Inf)
    fill!(writer.surface_maximum, -Inf)

    plan = writer.plan
    vertical_axis = plan.vertical_axis
    surface_lower = plan.surface_lower
    surface_upper = plan.surface_upper
    surface_spacing = plan.surface_spacing

    @inbounds for particle in eachindex(positions)
        particle_types[particle] == writer.fluid_type || continue
        position = positions[particle]

        for probe in eachindex(writer.pressure_nearest)
            distance² = 0.0
            for axis in 1:plan.dimension
                difference =
                    Float64(position[axis]) - plan.pressure_locations[axis, probe]
                distance² += difference * difference
            end
            if distance² < writer.pressure_distance²[probe]
                writer.pressure_distance²[probe] = distance²
                writer.pressure_nearest[probe] = particle
            end
        end

        for probe in eachindex(writer.velocity_nearest)
            distance² = 0.0
            for axis in 1:plan.dimension
                difference =
                    Float64(position[axis]) - plan.velocity_locations[axis, probe]
                distance² += difference * difference
            end
            if distance² < writer.velocity_distance²[probe]
                writer.velocity_distance²[probe] = distance²
                writer.velocity_nearest[probe] = particle
            end
        end

        for probe in eachindex(writer.column_maximum)
            distance² = 0.0
            for axis in 1:plan.dimension
                axis == vertical_axis && continue
                difference =
                    Float64(position[axis]) - plan.column_locations[axis, probe]
                distance² += difference * difference
            end
            if distance² <= plan.column_radii[probe]^2
                writer.column_maximum[probe] =
                    max(writer.column_maximum[probe], Float64(position[vertical_axis]))
            end
        end

        if !isempty(plan.surface_shape)
            bin = 1
            inside = true
            for (horizontal_axis, axis) in enumerate(plan.surface_axes)
                coordinate = Float64(position[axis])
                if coordinate < surface_lower[axis] ||
                   coordinate > surface_upper[axis]
                    inside = false
                    break
                end
                bin_index = min(
                    floor(Int, (coordinate - surface_lower[axis]) / surface_spacing) + 1,
                    plan.surface_shape[horizontal_axis],
                )
                bin += (bin_index - 1) * plan.surface_strides[horizontal_axis]
            end
            if inside
                writer.surface_maximum[bin] =
                    max(writer.surface_maximum[bin], Float64(position[vertical_axis]))
            end
        end
    end

    return nothing
end

function append_measurements!(writer::MeasurementWriter, time, particles)
    writer.output_frames_seen += 1
    mod(writer.output_frames_seen - 1, writer.plan.sample_every) == 0 ||
        return writer

    hasproperty(particles, :Position) ||
        throw(ArgumentError("measurements require particle positions"))
    hasproperty(particles, :Type) ||
        throw(ArgumentError("measurements require particle types"))
    positions = particles.Position
    particle_types = particles.Type
    length(positions) == length(particle_types) ||
        throw(DimensionMismatch("particle positions and types must have matching lengths"))
    if !isempty(positions) &&
       length(positions[firstindex(positions)]) != writer.plan.dimension
        throw(DimensionMismatch(
            "particle positions do not match the measurement dimension",
        ))
    end
    if !isempty(writer.pressure_nearest)
        hasproperty(particles, :Pressure) ||
            throw(ArgumentError("pressure probes require the Pressure particle field"))
        length(particles.Pressure) == length(positions) ||
            throw(DimensionMismatch(
                "particle pressure and positions must have matching lengths",
            ))
    end
    if !isempty(writer.velocity_nearest)
        hasproperty(particles, :Velocity) ||
            throw(ArgumentError("velocity probes require the Velocity particle field"))
        length(particles.Velocity) == length(positions) ||
            throw(DimensionMismatch(
                "particle velocity and positions must have matching lengths",
            ))
    end

    _nearest_indices!(writer, positions, particle_types)
    slot = writer.pending + 1
    writer.time_buffer[slot] = Float64(time)

    if writer.pressure !== nothing
        values = view(writer.pressure.buffer, :, slot)
        fill!(values, NaN)
        for probe in eachindex(writer.pressure_nearest)
            particle = writer.pressure_nearest[probe]
            particle == 0 ||
                (values[probe] = Float64(particles.Pressure[particle]))
        end
    end
    if writer.velocity !== nothing
        values = view(writer.velocity.buffer, :, :, slot)
        fill!(values, NaN)
        for probe in eachindex(writer.velocity_nearest)
            particle = writer.velocity_nearest[probe]
            particle == 0 && continue
            for axis in 1:writer.plan.dimension
                values[axis, probe] = Float64(particles.Velocity[particle][axis])
            end
        end
    end
    if writer.column_height !== nothing
        values = view(writer.column_height.buffer, :, slot)
        fill!(values, NaN)
        for probe in eachindex(writer.column_maximum)
            height = writer.column_maximum[probe]
            isfinite(height) &&
                (values[probe] =
                    max(
                        0.0,
                        height - writer.plan.column_locations[
                            writer.plan.vertical_axis, probe
                        ],
                    ))
        end
    end
    if writer.surface_height !== nothing
        values = view(writer.surface_height.buffer, :, slot)
        for bin in eachindex(writer.surface_maximum)
            height = writer.surface_maximum[bin]
            values[bin] = isfinite(height) ? height : NaN
        end
    end

    writer.pending += 1
    writer.pending == writer.capacity && flush_measurements!(writer)
    return writer
end

function _flush_scalar!(series::ScalarSeries, frames_written, count)
    old_frames = frames_written
    total_frames = old_frames + count
    HDF5.set_extent_dims(
        series.dataset, (size(series.dataset, 1), total_frames),
    )
    series.dataset[:, (old_frames + 1):total_frames] =
        view(series.buffer, :, 1:count)
    return nothing
end

function _flush_vector!(series::VectorSeries, frames_written, count)
    old_frames = frames_written
    total_frames = old_frames + count
    HDF5.set_extent_dims(
        series.dataset,
        (size(series.dataset, 1), size(series.dataset, 2), total_frames),
    )
    series.dataset[:, :, (old_frames + 1):total_frames] =
        view(series.buffer, :, :, 1:count)
    return nothing
end

function flush_measurements!(writer::MeasurementWriter)
    count = writer.pending
    count == 0 && return writer
    first_frame = writer.frames_written
    total_frames = first_frame + count
    HDF5.set_extent_dims(writer.time_dataset, (total_frames,))
    writer.time_dataset[(first_frame + 1):total_frames] =
        view(writer.time_buffer, 1:count)
    writer.pressure === nothing ||
        _flush_scalar!(writer.pressure, first_frame, count)
    writer.velocity === nothing ||
        _flush_vector!(writer.velocity, first_frame, count)
    writer.column_height === nothing ||
        _flush_scalar!(writer.column_height, first_frame, count)
    writer.surface_height === nothing ||
        _flush_scalar!(writer.surface_height, first_frame, count)
    writer.frames_written = total_frames
    writer.pending = 0
    return writer
end

end
