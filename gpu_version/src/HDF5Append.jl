module HDF5Append

using HDF5

# Grow a dataset and return the one-based start of the appended region.
function extend!(ds::HDF5.Dataset, n::Integer)
    len = length(ds)
    HDF5.set_extent_dims(ds, (len + n,))
    return len + 1
end

function extend_columns!(ds::HDF5.Dataset, n::Integer)
    rows, cols = size(ds)
    HDF5.set_extent_dims(ds, (rows, cols + n))
    return cols + 1
end

function append_slice!(ds::HDF5.Dataset, data::AbstractVector)
    n = length(data)
    n == 0 && return ds
    first = extend!(ds, n)
    ds[first:(first + n - 1)] = data
    return ds
end

function append_columns!(ds::HDF5.Dataset, data::AbstractMatrix)
    n = size(data, 2)
    n == 0 && return ds
    first = extend_columns!(ds, n)
    ds[:, first:(first + n - 1)] = data
    return ds
end

end
