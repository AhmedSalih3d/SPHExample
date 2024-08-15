module AuxillaryFunctions
using StaticArrays
using Base.Threads
using HDF5

export ResetArrays!, to_3d, CloseHDFVTKManually, CleanUpSimulationFolder

ResetArrays!(arrays...) = foreach(a -> fill!(a, zero(eltype(a))), arrays)

to_3d(vec_2d) = [SVector(v..., 0.0) for v in vec_2d]

function CloseHDFVTKManually(directory_path::String)
    # Get all files in the directory
    all_files = readdir(directory_path, join=true)

    # Filter out the files that end with .vtkhdf
    vtkhdf_files = filter(file -> endswith(file, ".vtkhdf"), all_files)
    
    # Process each file in a threaded manner
    @threads for file_path in vtkhdf_files
        file = h5open(file_path, "r")
        try
            close(file)
        catch e
            @warn(e)
        end
    end
end

function CleanUpSimulationFolder(FilePath)
    # Delete previous result files
    # https://discourse.julialang.org/t/find-what-has-locked-held-a-file/23278
    GC.gc()
    try
        foreach(rm, filter(endswith(".vtkhdf"), readdir(FilePath,join=true)))
    catch err
        @warn("File could not be deleted, manually delete else program cannot conclude.")
        display(err)
    end

    return nothing
end

end