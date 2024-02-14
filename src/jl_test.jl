using StaticArrays
using LoopVectorization
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

function updateV!(data::DimensionalData)
    # Ensure I and J have the same length
    @assert length(I) == length(J) "Index vectors I and J must have the same length"

    # Iterate through each dimension of data.vectors
    for d in 1:length(data.vectors)
        p_d = data.vectors[d]  # Access the d-th dimension vector
        
        # Preallocate or ensure x_ij_d is correctly sized for the output
        # Assuming x_ij_d should be stored in data.V or a similar structure
        # This step depends on the specific structure of data.V and how you want to store the results
        # For demonstration, assume we're directly updating p_d in place for simplicity
        
        @tturbo for k in 1:length(I)
            i, j = I[k], J[k]
            p_d[k] = p_d[i] - p_d[j]  # Compute the difference for the d-th dimension
        end
    end
    
    # Depending on the structure of data.V and your specific needs, additional steps
    # to update or reflect changes in data.V might be necessary here.
end

# Create a 2D DimensionalData with vectors of length 5
data = DimensionalData([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0])
updateV!(data)