module AuxiliaryFunctions
using StaticArrays
using Base.Threads
using Polyester
using HDF5

export ResetArrays!, to_3d, CloseHDFVTKManually, CleanUpSimulationFolder

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

    Polyester.@batch per=thread for file_path in vtkhdf_files
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
