# using Base.Threads
# using Bumper
# using StrideArrays # Not necessary, but can make operations like broadcasting with Bumper.jl faster.
# using Polyester
# using BenchmarkTools
# using LoopVectorization
# using ChunkSplitters
# using NNlib

# function NaiveReductionFunction!(dρdtI, I,J,drhopLp,drhopLn)
#     # Reduction
#     @inbounds for iter in eachindex(I,J)
#         i = I[iter]
#         j = J[iter]

#         dρdtI[i] +=  drhopLp[iter]
#         dρdtI[j] +=  drhopLn[iter]
#     end
# end


# function ReductionFunction!(dρdtI,I,J,drhopLp,drhopLn)
#     XT = eltype(dρdtI); XL = length(dρdtI); X0 = zero(XT)

#     @inbounds @no_escape begin
#         local_X = @alloc(XT,XL,nthreads())

#         fill!(local_X,X0)

#         @batch for iter in eachindex(I,J)
#             i = I[iter]
#             j = J[iter]

#             # Accumulate in thread-local storage
#             @turbo local_X[:, threadid()][i] +=  drhopLp[iter]
#             @turbo local_X[:, threadid()][j] +=  drhopLn[iter]
#         end

#         # Reduce the thread-local storage into the shared arrays
#         @inbounds for tid in 1:nthreads()
#             @turbo dρdtI .+= local_X[:, tid]
#         end
#     end

#     return nothing
# end

# function ReductionFunctionChunk!(dρdtI, I, J, drhopLp, drhopLn)
#     XT = eltype(dρdtI); XL = length(dρdtI); X0 = zero(XT)
#     nchunks = nthreads()  # Assuming nchunks is defined somewhere as nthreads()

#     @inbounds @no_escape begin
#         local_X = @alloc(XT, XL, nchunks)

#         fill!(local_X,X0)

#         # Directly iterate over the chunks
#         @batch for ichunk in 1:nchunks
#             chunk_inds = getchunk(I, ichunk; n=nchunks)
#             for idx in chunk_inds
#                 i = I[idx]
#                 j = J[idx]

#                 # Accumulate the contributions into the correct place
#                 local_X[i, ichunk] += drhopLp[idx]
#                 local_X[j, ichunk] += drhopLn[idx]
#             end
#         end

#         # Reduction step
#         @tturbo for ix in 1:XL
#             for chunk in 1:nchunks
#                 dρdtI[ix] += local_X[ix, chunk]
#             end
#         end
#     end



#     return nothing
# end


# begin
#     ProblemScaleFactor  = 5
#     NumberOfPoints      = 6195*ProblemScaleFactor
#     NumberOfInterations = 50000*ProblemScaleFactor
#     I           = rand(1:NumberOfPoints, NumberOfInterations)
#     J           = I; #rand(1:NumberOfPoints, NumberOfInterations)
#     dρdtI       = zeros(NumberOfPoints) 
#     drhopLp     = rand(NumberOfInterations)    
#     drhopLn     = rand(NumberOfInterations)
#     nchunks = nthreads()
#     local_X = [zeros(length(dρdtI)) for _ in 1:nchunks]


#     dρdtI .= zero(eltype(dρdtI))
#     NaiveReductionFunction!(dρdtI,I,J,drhopLp,drhopLn)
#     println("Value when doing naive reduction: ", sum(dρdtI))

#     dρdtI .= zero(eltype(dρdtI))
#     ReductionFunction!(dρdtI, I,J,drhopLp,drhopLn)
#     println("Value when doing advanced reduction: ", sum(dρdtI))

#     dρdtI .= zero(eltype(dρdtI))
#     ReductionFunctionChunk!(dρdtI, I,J,drhopLp,drhopLn)
#     println("Value when doing chunk reduction: ", sum(dρdtI))


#     # Benchmark
#     println("Naive function:")
#     display(@benchmark NaiveReductionFunction!($dρdtI , $I, $J, $drhopLp, $drhopLn))

#     println("Reduction function:")
#     display(@benchmark ReductionFunction!($dρdtI , $I, $J, $drhopLp, $drhopLn))

#     println("Chunk function:")
#     display(@benchmark ReductionFunctionChunk!($dρdtI , $I, $J, $drhopLp, $drhopLn))

# end



using Base.Threads
using Bumper
using StrideArrays # Not necessary, but can make operations like broadcasting with Bumper.jl faster.
using Polyester
using BenchmarkTools
using LoopVectorization
using ChunkSplitters
using StaticArrays
using StructArrays

struct DimensionalData{D, T <: AbstractFloat}
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


function NaiveReductionFunction!(dρdtI, I,J,drhopLp,drhopLn)
    # Reduction
    @inbounds for iter in eachindex(I,J)
        i = I[iter]
        j = J[iter]

        dρdtI[i] +=  drhopLp[iter]
        dρdtI[j] +=  drhopLn[iter]
    end
end

function ReductionFunctionChunk!(dρdtI, I, J, drhopLp, drhopLn)
    XT = eltype(dρdtI); XL = length(dρdtI); X0 = zero(XT)
    nchunks = nthreads()  # Assuming nchunks is defined somewhere as nthreads()

    @inbounds @no_escape begin
        local_X = @alloc(XT, XL, nchunks)

        fill!(local_X,X0)

        # Directly iterate over the chunks
        @batch for ichunk in 1:nchunks
            chunk_inds = getchunk(I, ichunk; n=nchunks)
            for idx in chunk_inds
                i = I[idx]
                j = J[idx]

                # Accumulate the contributions into the correct place
                local_X[i, ichunk] += drhopLp[idx]
                local_X[j, ichunk] += drhopLn[idx]
            end
        end

        # Reduction step
        # using @tturbo is slightly faster than batch (28 mus), but @batch (30 mus) works for svector, so we prefer this.
        @batch for ix in 1:XL
            for chunk in 1:nchunks
                dρdtI[ix] += local_X[ix, chunk]
            end
        end
    end
    
    return nothing
end


begin
    ProblemScaleFactor  = 1
    NumberOfPoints      = 6195*ProblemScaleFactor
    NumberOfInterations = 50000*ProblemScaleFactor
    I           = rand(1:NumberOfPoints, NumberOfInterations)
    J           = I; #rand(1:NumberOfPoints, NumberOfInterations)
    dρdtI       = zeros(NumberOfPoints) 
    drhopLp     = rand(NumberOfInterations)    
    drhopLn     = rand(NumberOfInterations)
    V           = zeros(SVector{2,Float64},NumberOfPoints)
    VL          = rand(eltype(V),NumberOfInterations)

    VD          = DimensionalData{2, Float64}(NumberOfPoints)
    VDL         = DimensionalData{2, Float64}(NumberOfInterations)
    VDL.V      .= VL

    dρdtI .= zero(eltype(dρdtI))
    NaiveReductionFunction!(dρdtI,I,J,drhopLp,drhopLn)
    println("Value when doing naive reduction: ", sum(dρdtI))

    dρdtI .= zero(eltype(dρdtI))
    ReductionFunctionChunk!(dρdtI, I,J,drhopLp,drhopLn)
    println("Value when doing chunk reduction: ", sum(dρdtI))

    V .*= 0
    NaiveReductionFunction!(V,I,J,VL,VL)
    println("Value when doing naive svector reduction: ", sum(V))

    V .*= 0
    ReductionFunctionChunk!(V,I,J,VL,VL)
    println("Value when doing chunk svector reduction: ", sum(V))

    VD.V .*= 0
    ReductionFunctionChunk!(VD.V,I,J,VDL.V,VDL.V)
    println("Value when doing chunk svector reduction with dimensional data: ", sum(VD.V))


    # # Benchmark
    # println("Naive function:")
    # display(@benchmark NaiveReductionFunction!($dρdtI , $I, $J, $drhopLp, $drhopLn))

    # println("Chunk function:")
    # display(@benchmark ReductionFunctionChunk!($dρdtI , $I, $J, $drhopLp, $drhopLn))

    # # Benchmark
    # println("Naive svector function:")
    # display(@benchmark NaiveReductionFunction!($V,$I,$J,$VL,$VL))

    # println("Chunk function:")
    # display(@benchmark ReductionFunctionChunk!($V,$I,$J,$VL,$VL))

    println("Chunk function:")
    display(@benchmark ReductionFunctionChunk!($VD.V,$I,$J,$VDL.V,$VDL.V))



    

end
