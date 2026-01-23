module AuxiliaryFunctions
using StaticArrays
using Base.Threads
using HDF5
using SIMD: Vec
import LinearAlgebra: dot

export ResetArrays!, to_3d, CloseHDFVTKManually, CleanUpSimulationFolder, DotSimd

"""
    ResetArrays!(arrays...)

Fill each array in `arrays` with zeros in place.
"""
@inline ResetArrays!(arrays...) = foreach(a -> fill!(a, zero(eltype(a))), arrays)

"""
    to_3d(vec_2d)

Convert a vector of 2D `SVector`s to 3D by appending a zero z-component.
"""
@inline to_3d(vec_2d) = [SVector(v..., 0.0) for v in vec_2d]

"""
    DotSimd(a, b)

SIMD-accelerated dot product for `SVector` inputs where it is practical,
falling back to `dot` for other sizes and types.
"""
@inline function DotSimd(a::SVector{2,T}, b::SVector{2,T}) where {T<:AbstractFloat}
    veca = Vec{2,T}(Tuple(a))
    vecb = Vec{2,T}(Tuple(b))
    return sum(veca * vecb)
end

@inline function DotSimd(a::SVector{3,T}, b::SVector{3,T}) where {T<:AbstractFloat}
    veca = Vec{4,T}(a[1], a[2], a[3], zero(T))
    vecb = Vec{4,T}(b[1], b[2], b[3], zero(T))
    return sum(veca * vecb)
end

@inline function DotSimd(a::SVector{D,T}, b::SVector{D,T}) where {D,T}
    return dot(a, b)
end

"""
    to_3d!(dest, src)

Fill the preallocated vector `dest` with 3D versions of the 2D vectors in
`src`. The resulting `SVector`s share the element type with `src`.
"""
function to_3d!(dest::AbstractVector{SVector{3,T}}, src::AbstractVector{SVector{2,T}}) where T
    @inbounds @simd for i in eachindex(src)
        v = src[i]
        dest[i] = SVector{3,T}(v[1], v[2], zero(T))
    end
    return dest
end

"""
    CloseHDFVTKManually(directory_path)

Iterate over all `.vtkhdf` files in `directory_path` and close them.
Useful when a simulation terminated before closing its output files.
"""
function CloseHDFVTKManually(directory_path::String)
    all_files = readdir(directory_path, join=true)
    vtkhdf_files = filter(file -> endswith(file, ".vtkhdf"), all_files)

    @threads for file_path in vtkhdf_files
        file = h5open(file_path, "r")
        try
            close(file)
        catch e
            @warn(e)
        end
    end
end

"""
    CleanUpSimulationFolder(FilePath)

Remove stale `.vtkhdf` files from `FilePath`.
"""
function CleanUpSimulationFolder(FilePath)
    GC.gc()
    try
        foreach(rm, filter(endswith(".vtkhdf"), readdir(FilePath, join=true)))
    catch err
        @warn("File could not be deleted, manually delete else program cannot conclude.")
        display(err)
    end

    return nothing
end

end
