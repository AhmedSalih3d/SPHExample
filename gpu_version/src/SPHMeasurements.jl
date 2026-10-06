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


# ---------------------------------------------------------------------------
# VTKHDF output
# ---------------------------------------------------------------------------
#
# Measurements are stored in the particle `.vtkhdf` file as blocks of a
# `MultiBlockDataSet` (VTKHDF specification, section
# "PartitionedDataSetCollection and MultiBlockDataSet"). Every measurement
# category is one temporal `PolyData` block of vertices with one point per
# probe (or free-surface bin), so the standard VTKHDF reader loads it next to
# the particle block (`ProduceHDFVTK` links the blocks into the `Assembly`):
#
#   /VTKHDF/PressureProbes     PointData: Pressure, Sampled, SampleTime
#   /VTKHDF/VelocityProbes     PointData: Velocity, Sampled, SampleTime
#   /VTKHDF/WaterColumnProbes  PointData: WaterColumnHeight, Radius, Sampled, SampleTime
#   /VTKHDF/FreeSurface        PointData: FreeSurfaceHeight, Sampled, SampleTime
#
# The reader requires every block of a temporal composite to have the same
# number of steps and the same time values, so a block receives one frame per
# output frame of the particles even when `sample_every > 1`: frames between
# samples repeat the previous sample with `Sampled == 0` and `SampleTime`
# equal to the time of that sample; sampled frames carry `Sampled == 1`.
#
# Pressure and velocity probe locations and static metadata (`Radius`, probe
# names, and free-surface grid shape/spacing) are written once. Water-column
# and free-surface points move vertically to their latest measured height;
# unsampled frames repeat the last sampled point locations.
# Frame data and moving points are buffered in host memory and written
# `capacity` frames at a time by `flush_measurements!`.

const BLOCK_VERSION = Int32[2, 3]
const IdType = Int64

const PRESSURE_BLOCK = "PressureProbes"
const VELOCITY_BLOCK = "VelocityProbes"
const WATER_COLUMN_BLOCK = "WaterColumnProbes"
const FREE_SURFACE_BLOCK = "FreeSurface"

"""
    measurement_block_names(plan) -> Vector{String}

Names of the VTKHDF blocks written for `plan`, in writing order.
"""
function measurement_block_names(plan::ResolvedMeasurements)
    names = String[]
    isempty(plan.pressure_names) || push!(names, PRESSURE_BLOCK)
    isempty(plan.velocity_names) || push!(names, VELOCITY_BLOCK)
    isempty(plan.column_names) || push!(names, WATER_COLUMN_BLOCK)
    isempty(plan.surface_axes) || push!(names, FREE_SURFACE_BLOCK)
    return names
end

function _write_ascii_attribute(group, name, value::AbstractString)
    dtype = HDF5.datatype(value)
    HDF5.API.h5t_set_cset(dtype.id, HDF5.API.H5T_CSET_ASCII)
    attribute = HDF5.create_attribute(group, name, dtype, HDF5.dataspace(value))
    HDF5.write_attribute(attribute, dtype, value)
    return group
end

function _write_attribute(group, name, value::AbstractString)
    return _write_ascii_attribute(group, name, value)
end

function _write_attribute(group, name, value)
    HDF5.attrs(group)[name] = value
    return group
end

# Append `count` entries to the one dimensional dataset `ds` and return the
# (one based) index of the first new entry. New entries read as zero.
function _extend!(ds::HDF5.Dataset, count::Integer)
    len = length(ds)
    HDF5.set_extent_dims(ds, (len + count,))
    return len + 1
end

function _append!(ds::HDF5.Dataset, data::AbstractVector)
    isempty(data) && return ds
    first = _extend!(ds, length(data))
    ds[first:(first + length(data) - 1)] = data
    return ds
end

# Append `count` zero columns to the `rows × M` dataset `ds`.
function _extend_columns!(ds::HDF5.Dataset, count::Integer)
    rows, cols = size(ds)
    HDF5.set_extent_dims(ds, (rows, cols + count))
    return cols + 1
end

function _append_columns!(ds::HDF5.Dataset, data::AbstractMatrix)
    size(data, 2) == 0 && return ds
    first = _extend_columns!(ds, size(data, 2))
    ds[:, first:(first + size(data, 2) - 1)] = data
    return ds
end

_step_dataset(group, name, ::Type{T}, capacity) where {T} =
    HDF5.create_dataset(group, name, T, ((0,), (-1,)), chunk = (capacity,))

_step_columns(group, name, rows, capacity) =
    HDF5.create_dataset(
        group, name, IdType, ((rows, 0), (rows, -1)), chunk = (rows, capacity),
    )

"""
A temporal point data array of a block: the dataset, its step offsets and the
frame buffer (`n × capacity` for scalars, `3 × n × capacity` for vectors).
"""
struct FrameArray{A <: AbstractArray}
    dataset::HDF5.Dataset
    offsets::HDF5.Dataset
    buffer::A
end

frame_slot(array::FrameArray{<:AbstractMatrix}, slot) = view(array.buffer, :, slot)
frame_slot(array::FrameArray{<:AbstractArray{<:Any, 3}}, slot) =
    view(array.buffer, :, :, slot)

_flush!(array::FrameArray{<:AbstractMatrix}, count) =
    _append!(array.dataset, vec(array.buffer[:, 1:count]))

function _flush!(array::FrameArray{<:AbstractArray{<:Any, 3}}, count)
    npoints = size(array.buffer, 2)
    return _append_columns!(
        array.dataset, reshape(array.buffer[:, :, 1:count], 3, npoints * count),
    )
end

"""
One measurement block: a temporal vertex `PolyData` under `/VTKHDF/<name>`
with one point per probe. Holds the handles of the datasets that grow by one
entry per frame, grouped by the value they receive.
"""
struct MeasurementBlock
    name::String
    npoints::Int
    points::Matrix{Float64}
    points_dataset::Union{Nothing, HDF5.Dataset}
    points_buffer::Union{Nothing, Matrix{Float64}}
    dynamic_points::Bool
    nsteps::HDF5.Attribute
    step_values::HDF5.Dataset           # Steps/Values: the frame times
    part_offsets::HDF5.Dataset          # Steps/PartOffsets: the frame index
    point_offsets::HDF5.Dataset         # Steps/PointOffsets
    point_counts::Vector{HDF5.Dataset}  # `npoints` points and vertices per frame
    part_counts::Vector{HDF5.Dataset}   # Steps/NumberOfParts: one part per frame
    zero_entries::Vector{HDF5.Dataset}  # offsets into static data, empty topologies
    zero_columns::Vector{HDF5.Dataset}  # Steps/CellOffsets, ConnectivityIdOffsets
    field_sizes::Vector{Pair{HDF5.Dataset, Vector{IdType}}} # Steps/FieldDataSizes
    frame_arrays::Vector{FrameArray}    # values, Sampled, SampleTime
end

"""
    MeasurementBlock(root, name, points, arrays, static_arrays, field_arrays,
                     attributes, capacity)

Create the block `name` in the composite group `root`. `points` (`3 x n`) are
the probe locations. `arrays` lists the temporal point arrays as
`name => (eltype, components)`, `static_arrays` the point arrays and
`field_arrays` the field arrays written once as `name => values`.
`attributes` are descriptive HDF5 attributes of the block group.
Set `dynamic_points = true` to store a point geometry frame and offset for
every output frame.
"""
function MeasurementBlock(
        root::HDF5.Group, name::AbstractString, points::AbstractMatrix{Float64},
        arrays, static_arrays, field_arrays, attributes, capacity::Integer;
        dynamic_points::Bool = false,
    )
    size(points, 1) == 3 || throw(ArgumentError("block points must be 3 x n"))
    npoints = size(points, 2)
    npoints >= 1 || throw(ArgumentError("block `$name` needs at least one point"))

    block = HDF5.create_group(root, name)
    HDF5.attrs(block)["Version"] = BLOCK_VERSION
    _write_ascii_attribute(block, "Type", "PolyData")
    for (key, value) in attributes
        _write_attribute(block, key, value)
    end

    points_data = if dynamic_points
        HDF5.create_dataset(
            block,
            "Points",
            Float64,
            ((3, 0), (3, -1)),
            chunk = (3, npoints * capacity),
        )
    else
        block["Points"] = Matrix{Float64}(points)
        nothing
    end
    points_buffer = dynamic_points ?
        Matrix{Float64}(undef, 3, npoints * capacity) : nothing
    point_counts = HDF5.Dataset[_step_dataset(block, "NumberOfPoints", IdType, capacity)]
    zero_entries = HDF5.Dataset[]
    for topology in ("Vertices", "Lines", "Polygons", "Strips")
        group = HDF5.create_group(block, topology)
        cells = _step_dataset(group, "NumberOfCells", IdType, capacity)
        ids = _step_dataset(group, "NumberOfConnectivityIds", IdType, capacity)
        if topology == "Vertices"
            group["Connectivity"] = collect(IdType, 0:(npoints - 1))
            group["Offsets"] = collect(IdType, 0:npoints)
            push!(point_counts, cells, ids)
        else
            group["Connectivity"] = IdType[]
            group["Offsets"] = IdType[0]
            push!(zero_entries, cells, ids)
        end
    end

    steps = HDF5.create_group(block, "Steps")
    nsteps, _ = HDF5.create_attribute(steps, "NSteps", Int32)
    HDF5.write_attribute(nsteps, HDF5.datatype(Int32), Int32(0))
    step_values = _step_dataset(steps, "Values", Float64, capacity)
    part_offsets = _step_dataset(steps, "PartOffsets", IdType, capacity)
    part_counts = HDF5.Dataset[_step_dataset(steps, "NumberOfParts", IdType, capacity)]
    point_offsets = _step_dataset(steps, "PointOffsets", IdType, capacity)
    zero_columns = HDF5.Dataset[
        _step_columns(steps, "CellOffsets", 4, capacity),
        _step_columns(steps, "ConnectivityIdOffsets", 4, capacity),
    ]

    point_data = HDF5.create_group(block, "PointData")
    point_data_offsets = HDF5.create_group(steps, "PointDataOffsets")
    frame_arrays = FrameArray[]
    for (array_name, (T, components)) in arrays
        components in (1, 3) ||
            throw(ArgumentError("point arrays must have 1 or 3 components"))
        dataset, buffer = if components == 1
            HDF5.create_dataset(
                point_data, array_name, T, ((0,), (-1,)),
                chunk = (npoints * capacity,),
            ), Matrix{T}(undef, npoints, capacity)
        else
            HDF5.create_dataset(
                point_data, array_name, T, ((3, 0), (3, -1)),
                chunk = (3, npoints * capacity),
            ), Array{T, 3}(undef, 3, npoints, capacity)
        end
        offsets = _step_dataset(point_data_offsets, array_name, IdType, capacity)
        push!(frame_arrays, FrameArray(dataset, offsets, buffer))
    end
    for (array_name, values) in static_arrays
        length(values) == npoints ||
            throw(DimensionMismatch("static array `$array_name` needs one value per point"))
        point_data[array_name] = values
        push!(
            zero_entries,
            _step_dataset(point_data_offsets, array_name, IdType, capacity),
        )
    end

    field_sizes = Pair{HDF5.Dataset, Vector{IdType}}[]
    if !isempty(field_arrays)
        field_data = HDF5.create_group(block, "FieldData")
        field_offsets = HDF5.create_group(steps, "FieldDataOffsets")
        sizes = HDF5.create_group(steps, "FieldDataSizes")
        for (array_name, values) in field_arrays
            field_data[array_name] = values
            push!(zero_entries, _step_dataset(field_offsets, array_name, IdType, capacity))
            push!(
                field_sizes,
                _step_columns(sizes, array_name, 2, capacity) =>
                    IdType[1, length(values)],
            )
        end
    end

    return MeasurementBlock(
        String(name), npoints, Matrix{Float64}(points), points_data,
        points_buffer, dynamic_points, nsteps, step_values, part_offsets,
        point_offsets, point_counts, part_counts, zero_entries, zero_columns,
        field_sizes, frame_arrays,
    )
end

# Write `count` buffered frames with the times `times` to the block, the
# first being frame `first_frame` (zero based) of the file.
function _flush_block!(block::MeasurementBlock, times::AbstractVector{Float64},
                       first_frame::Integer)
    count = length(times)
    frames = collect(IdType, first_frame:(first_frame + count - 1))
    _append!(block.step_values, times)
    _append!(block.part_offsets, frames)
    if block.dynamic_points
        ncolumns = block.npoints * count
        _append_columns!(
            block.points_dataset::HDF5.Dataset,
            view(block.points_buffer::Matrix{Float64}, :, 1:ncolumns),
        )
        _append!(block.point_offsets, frames .* block.npoints)
    else
        _append!(block.point_offsets, zeros(IdType, count))
    end
    foreach(ds -> _append!(ds, fill(IdType(block.npoints), count)), block.point_counts)
    foreach(ds -> _append!(ds, ones(IdType, count)), block.part_counts)
    foreach(ds -> _extend!(ds, count), block.zero_entries)
    foreach(ds -> _extend_columns!(ds, count), block.zero_columns)
    for (ds, column) in block.field_sizes
        _append_columns!(ds, repeat(column, 1, count))
    end
    for array in block.frame_arrays
        _append!(array.offsets, frames .* block.npoints)
        _flush!(array, count)
    end
    HDF5.write_attribute(block.nsteps, HDF5.datatype(Int32), Int32(first_frame + count))
    return block
end

# The `3 × n` point coordinates of probe `locations` (`dimension × n`).
function _points3d(locations::AbstractMatrix)
    points = zeros(Float64, 3, size(locations, 2))
    points[1:size(locations, 1), :] = locations
    return points
end

# The `3 × n` coordinates of the free-surface bins: bin centres on the
# horizontal axes and the lower domain bound on the vertical axis.
function _surface_points3d(plan::ResolvedMeasurements)
    points = zeros(Float64, 3, size(plan.surface_locations, 2))
    for axis in 1:plan.dimension
        if axis == plan.vertical_axis
            points[axis, :] .= plan.surface_lower[axis]
        else
            horizontal = findfirst(==(axis), plan.surface_axes)
            points[axis, :] = plan.surface_locations[horizontal, :]
        end
    end
    return points
end

"""
Writer of the measurement blocks (see the section comment above). Keeps the
latest sample of every series so that output frames between samples repeat
it, and buffers `capacity` frames before writing.
"""
mutable struct MeasurementWriter{F}
    plan::ResolvedMeasurements
    fluid_type::F
    blocks::Vector{MeasurementBlock}
    pressure::Union{Nothing, MeasurementBlock}
    velocity::Union{Nothing, MeasurementBlock}
    column_height::Union{Nothing, MeasurementBlock}
    surface_height::Union{Nothing, MeasurementBlock}
    pressure_values::Vector{Float64}   # latest sample per probe
    velocity_values::Matrix{Float64}   # `3 × probes`, latest sample
    column_values::Vector{Float64}
    surface_values::Vector{Float64}
    pressure_nearest::Vector{Int}
    pressure_distance²::Vector{Float64}
    velocity_nearest::Vector{Int}
    velocity_distance²::Vector{Float64}
    column_maximum::Vector{Float64}
    surface_maximum::Vector{Float64}
    time_buffer::Vector{Float64}
    capacity::Int
    last_sample_time::Float64
    pending::Int
    frames_written::Int
    output_frames_seen::Int
end

"""
    MeasurementWriter(root, plan, fluid_type; capacity = 4)

Create the measurement blocks of `plan` in the `MultiBlockDataSet` group
`root` (`/VTKHDF`) and buffer `capacity` frames before a write. The caller
links the blocks named by `measurement_block_names(plan)` into the `Assembly`
group.
"""
function MeasurementWriter(root::HDF5.Group, plan::ResolvedMeasurements, fluid_type;
                           capacity::Integer = 4)
    capacity >= 1 ||
        throw(ArgumentError("measurement buffer capacity must be at least 1"))

    common = Pair{String, Any}[
        "SampleEveryOutputFrames" => Int32(plan.sample_every),
        "UnsampledFrames" => "repeat previous sample (Sampled = 0)",
        "CoordinateDimension" => Int32(plan.dimension),
        "VerticalAxis" => Int32(plan.vertical_axis),
    ]
    probe_attributes(names, extra...) =
        [common; "Names" => join(names, "\n"); extra...]
    sampled_arrays = ["Sampled" => (UInt8, 1), "SampleTime" => (Float64, 1)]
    none = Pair{String, Vector{Float64}}[]

    blocks = MeasurementBlock[]
    pressure = velocity = column_height = surface_height = nothing
    if !isempty(plan.pressure_names)
        pressure = MeasurementBlock(
            root, PRESSURE_BLOCK, _points3d(plan.pressure_locations),
            ["Pressure" => (Float64, 1); sampled_arrays], none,
            ["Names" => plan.pressure_names],
            probe_attributes(
                plan.pressure_names, "Sampling" => "nearest fluid particle",
            ),
            capacity,
        )
        push!(blocks, pressure)
    end
    if !isempty(plan.velocity_names)
        velocity = MeasurementBlock(
            root, VELOCITY_BLOCK, _points3d(plan.velocity_locations),
            ["Velocity" => (Float64, 3); sampled_arrays], none,
            ["Names" => plan.velocity_names],
            probe_attributes(
                plan.velocity_names, "Sampling" => "nearest fluid particle",
            ),
            capacity,
        )
        push!(blocks, velocity)
    end
    if !isempty(plan.column_names)
        column_height = MeasurementBlock(
            root, WATER_COLUMN_BLOCK, _points3d(plan.column_locations),
            ["WaterColumnHeight" => (Float64, 1); sampled_arrays],
            ["Radius" => plan.column_radii],
            ["Names" => plan.column_names],
            probe_attributes(
                plan.column_names,
                "Definition" => "maximum fluid-particle height minus probe height",
            ),
            capacity;
            dynamic_points = true,
        )
        push!(blocks, column_height)
    end
    if !isempty(plan.surface_axes)
        surface_height = MeasurementBlock(
            root, FREE_SURFACE_BLOCK, _surface_points3d(plan),
            ["FreeSurfaceHeight" => (Float64, 1); sampled_arrays], none,
            [
                "GridShape" => Int64.(plan.surface_shape),
                "HorizontalAxes" => Int32.(plan.surface_axes),
                "Spacing" => [plan.surface_spacing],
                "DomainLower" => plan.surface_lower,
                "DomainUpper" => plan.surface_upper,
            ],
            [
                common
                "Ordering" => "first horizontal axis varies fastest"
                "Definition" => "maximum fluid-particle height in each horizontal bin"
            ],
            capacity;
            dynamic_points = true,
        )
        push!(blocks, surface_height)
    end

    return MeasurementWriter(
        plan,
        fluid_type,
        blocks,
        pressure,
        velocity,
        column_height,
        surface_height,
        fill(NaN, length(plan.pressure_names)),
        fill(NaN, 3, length(plan.velocity_names)),
        fill(NaN, length(plan.column_names)),
        fill(NaN, size(plan.surface_locations, 2)),
        zeros(Int, length(plan.pressure_names)),
        Vector{Float64}(undef, length(plan.pressure_names)),
        zeros(Int, length(plan.velocity_names)),
        Vector{Float64}(undef, length(plan.velocity_names)),
        Vector{Float64}(undef, length(plan.column_names)),
        Vector{Float64}(undef, size(plan.surface_locations, 2)),
        Vector{Float64}(undef, capacity),
        Int(capacity),
        NaN,
        0,
        0,
        0,
    )
end

"""Number of frames written to the file so far."""
frames_written(writer::MeasurementWriter) = writer.frames_written

"""Number of frames buffered and not yet written."""
frames_pending(writer::MeasurementWriter) = writer.pending

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

function _update_height_points!(writer::MeasurementWriter)
    vertical_axis = writer.plan.vertical_axis
    if writer.column_height !== nothing
        for probe in eachindex(writer.column_values)
            height = writer.column_values[probe]
            location = writer.plan.column_locations[vertical_axis, probe]
            writer.column_height.points[vertical_axis, probe] =
                isfinite(height) ? location + height : location
        end
    end
    if writer.surface_height !== nothing
        lower = writer.plan.surface_lower[vertical_axis]
        for bin in eachindex(writer.surface_values)
            height = writer.surface_values[bin]
            writer.surface_height.points[vertical_axis, bin] =
                isfinite(height) ? height : lower
        end
    end
    return nothing
end

function _buffer_points!(block::MeasurementBlock, slot::Integer)
    block.dynamic_points || return nothing
    first_column = (slot - 1) * block.npoints + 1
    columns = first_column:(first_column + block.npoints - 1)
    (block.points_buffer::Matrix{Float64})[:, columns] = block.points
    return nothing
end


function _validate_particles(writer::MeasurementWriter, particles)
    hasproperty(particles, :Position) ||
        throw(ArgumentError("measurements require particle positions"))
    hasproperty(particles, :Type) ||
        throw(ArgumentError("measurements require particle types"))
    positions = particles.Position
    length(positions) == length(particles.Type) ||
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
    return nothing
end

# Sample every series from `particles` into the writer's latest-value arrays.
function _sample!(writer::MeasurementWriter, particles)
    _nearest_indices!(writer, particles.Position, particles.Type)
    plan = writer.plan

    values = writer.pressure_values
    fill!(values, NaN)
    for probe in eachindex(writer.pressure_nearest)
        particle = writer.pressure_nearest[probe]
        particle == 0 || (values[probe] = Float64(particles.Pressure[particle]))
    end

    velocity = writer.velocity_values
    fill!(velocity, NaN)
    for probe in eachindex(writer.velocity_nearest)
        particle = writer.velocity_nearest[probe]
        particle == 0 && continue
        for axis in 1:plan.dimension
            velocity[axis, probe] = Float64(particles.Velocity[particle][axis])
        end
        for axis in (plan.dimension + 1):3
            velocity[axis, probe] = 0.0
        end
    end

    heights = writer.column_values
    fill!(heights, NaN)
    for probe in eachindex(writer.column_maximum)
        height = writer.column_maximum[probe]
        isfinite(height) &&
            (heights[probe] = max(
                0.0, height - plan.column_locations[plan.vertical_axis, probe],
            ))
    end

    surface = writer.surface_values
    for bin in eachindex(writer.surface_maximum)
        height = writer.surface_maximum[bin]
        surface[bin] = isfinite(height) ? height : NaN
    end
    return nothing
end

"""
    append_measurements!(writer, time, particles)

Record the output frame at `time`. The frame is sampled from `particles` when
it is the first frame or `sample_every` frames after the previous sample;
otherwise the previous sample is repeated with `Sampled = 0`. Every frame is
written to all blocks so that they share the time steps of the particles.
"""
function append_measurements!(writer::MeasurementWriter, time, particles)
    writer.output_frames_seen += 1
    sampled = mod(writer.output_frames_seen - 1, writer.plan.sample_every) == 0
    if sampled
        _validate_particles(writer, particles)
        _sample!(writer, particles)
        writer.last_sample_time = Float64(time)
    end
    _update_height_points!(writer)

    slot = writer.pending + 1
    writer.time_buffer[slot] = Float64(time)
    flag = UInt8(sampled)
    for block in writer.blocks
        _buffer_points!(block, slot)
        fill!(frame_slot(block.frame_arrays[2], slot), flag)
        fill!(frame_slot(block.frame_arrays[3], slot), writer.last_sample_time)
    end
    # The first frame array of a block holds the measured values.
    store!(block, values) = copyto!(frame_slot(block.frame_arrays[1], slot), values)
    writer.pressure === nothing || store!(writer.pressure, writer.pressure_values)
    writer.velocity === nothing || store!(writer.velocity, writer.velocity_values)
    writer.column_height === nothing ||
        store!(writer.column_height, writer.column_values)
    writer.surface_height === nothing ||
        store!(writer.surface_height, writer.surface_values)

    writer.pending += 1
    writer.pending == writer.capacity && flush_measurements!(writer)
    return writer
end

"""
    flush_measurements!(writer)

Write the buffered frames to every measurement block.
"""
function flush_measurements!(writer::MeasurementWriter)
    count = writer.pending
    count == 0 && return writer
    times = writer.time_buffer[1:count]
    for block in writer.blocks
        _flush_block!(block, times, writer.frames_written)
    end
    writer.frames_written += count
    writer.pending = 0
    return writer
end

end
