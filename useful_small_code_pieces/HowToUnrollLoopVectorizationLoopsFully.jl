using StaticArrays
using LoopVectorization
using StructArrays
using BenchmarkTools

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

function flatten(data::DimensionalData)
    flatVector = Vector{eltype(data.V)}()  # Initialize an empty vector of the appropriate type
    for vec in data.vectors
        append!(flatVector, vec)  # Append each vector's contents to the flatVector
    end
    return flatVector
end


function updateV!(result::DimensionalData, data::DimensionalData, I, J)
    for d ∈ 1:length(result.vectors)
        for iter ∈ eachindex(I,J)
            i, j = I[iter], J[iter]
            result.vectors[d][iter] = data.vectors[d][i] - data.vectors[d][j]  # Compute the difference for the d-th dimension
        end
    end
end

function updateVT!(result::DimensionalData, data::DimensionalData, I, J)
    for d ∈ 1:length(result.vectors)
        @tturbo for iter ∈ eachindex(I,J)
            i, j = I[iter], J[iter]
            result.vectors[d][iter] = data.vectors[d][i] - data.vectors[d][j]  # Compute the difference for the d-th dimension
        end
    end
end

function updateVTManual!(result::DimensionalData, data::DimensionalData, I, J)
        @tturbo for iter ∈ eachindex(I,J)
            i, j = I[iter], J[iter]
            result.vectors[1][iter] = data.vectors[1][i] - data.vectors[1][j]  # Compute the difference for the d-th dimension
            result.vectors[2][iter] = data.vectors[2][i] - data.vectors[2][j]  # Compute the difference for the d-th dimension
        end
end

using LoopVectorization

# Define a generated function to dynamically create expressions based on D
@generated function updateVTAuto!(result::DimensionalData{D}, data::DimensionalData, I, J) where {D}
    quote
        @tturbo for iter ∈ eachindex(I,J)
            i, j = I[iter], J[iter]
            Base.Cartesian.@nexprs $D d -> begin
              result.vectors[d][iter] = data.vectors[d][i] - data.vectors[d][j]  # Compute the difference for the d-th dimension
            end
        end
    end
end


# Test performance
let
    D    = 3
    T    = Float32
    N    = 10000
    NL   = 500000
    data = DimensionalData(rand(N),rand(N))
    I    = rand(1:N, NL)
    J    = rand(1:N, NL)
    P    = DimensionalData{2,Float64}(NL)

    println("Naive:"); display(@benchmark updateV!($P,$data,$I,$J))
    println("Turbo:"); display(@benchmark updateVT!($P,$data,$I,$J))
    println("Turbo Manual Unroll:"); display(@benchmark updateVTManual!($P,$data,$I,$J))
    println("Turbo Auto Unroll:"); display(@benchmark updateVTAuto!($P,$data,$I,$J))
end

# Test correctness
let
    D     = 1
    T     = Float64
    N     = 10000
    NL    = 500000
    data  = DimensionalData(rand(N),rand(N))
    I     = rand(1:N, NL)
    J     = rand(1:N, NL)
    P1    = DimensionalData{2,Float64}(NL)
    P2    = DimensionalData{2,Float64}(NL)

    updateV!(P1,data,I,J)
    updateVTAuto!(P2,data,I,J)

    println("Dimension 1 is: ", P1.vectors[1] ≈ P2.vectors[1])
    println("Dimension 2 is: ", P1.vectors[2] ≈ P2.vectors[2])

    D     = 2
    T     = Float64
    N     = 10000
    NL    = 500000
    data  = DimensionalData(rand(N),rand(N))
    I     = rand(1:N, NL)
    J     = rand(1:N, NL)
    P1    = DimensionalData{2,Float64}(NL)
    P2    = DimensionalData{2,Float64}(NL)

    updateV!(P1,data,I,J)
    updateVTAuto!(P2,data,I,J)

    println("Dimension 1 is: ", P1.vectors[1] ≈ P2.vectors[1])
    println("Dimension 2 is: ", P1.vectors[2] ≈ P2.vectors[2])

    D     = 3
    T     = Float64
    N     = 10000
    NL    = 500000
    data  = DimensionalData(rand(N),rand(N))
    I     = rand(1:N, NL)
    J     = rand(1:N, NL)
    P1    = DimensionalData{2,Float64}(NL)
    P2    = DimensionalData{2,Float64}(NL)

    updateV!(P1,data,I,J)
    updateVTAuto!(P2,data,I,J)

    println("Dimension 1 is: ", P1.vectors[1] ≈ P2.vectors[1])
    println("Dimension 2 is: ", P1.vectors[2] ≈ P2.vectors[2])
end