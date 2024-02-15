module SimulationDataArrays

export ResetArrays!, ResizeBuffers!, DimensionalData

using StaticArrays
using StructArrays

struct DimensionalData{D, T}
    vectors::Tuple{Vararg{Vector{T}, D}}
    V::StructArray{SVector{D, T}, 1, Tuple{Vararg{Vector{T}, D}}}

    # General constructor for vectors
    function DimensionalData(vectors::Vector{T}...) where {T}
        D = length(vectors)
        V = StructArray{SVector{D, T}}(vectors)
        new{D, T}(Tuple(vectors), V)
    end

    # Constructor for initializing with all zeros, adapting to dimension D
    function DimensionalData{D, T}(len::Int) where {D, T}
        vectors = ntuple(d -> zeros(T, len), D) # Create D vectors of zeros
        V = StructArray{SVector{D, T}}(vectors)
        new{D, T}(vectors, V)
    end
end

# Overwrite resizing and fill functions for DimensionalData
Base.resize!(data::DimensionalData,n::Int) = resize!(data.V,n) 
reset!(data::DimensionalData)              = fill!(data.V,zero(eltype(data.V)))
length(::DimensionalData)                  = length(data.V)

ResetArrays!(arrays...) = foreach(a -> fill!(a, zero(eltype(a))), arrays)

function ResizeBuffers!(args...; N::Int = 0)
    for a in args
        if length(a) != N resize!(a, N) end
    end
    args
end

end