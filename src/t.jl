
using Base.Threads
using JET
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


function ReductionFunctionChunk!(dρdtI, I, J, drhopLp, drhopLn)
    XT = eltype(dρdtI); XL = length(dρdtI); X0 = zero(XT)
    nchunks = nthreads()  # Assuming nchunks is defined somewhere as nthreads()

    @inbounds @no_escape begin
        local_X = @alloc(XT, XL, nchunks)

        fill!(local_X,X0)

        # Directly iterate over the chunks
        b = @macroexpand @batch for ichunk in 1:nchunks
            chunk_inds = getchunk(I, ichunk; n=nchunks)
            for idx in chunk_inds
                i = I[idx]
                j = J[idx]

                # Accumulate the contributions into the correct place
                local_X[i, ichunk] += drhopLp[idx]
                local_X[j, ichunk] += drhopLn[idx]
            end
        end
        println(b)


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

function ReductionFunctionChunkNoBatch!(dρdtI, I, J, drhopLp, drhopLn)
    XT = eltype(dρdtI); XL = length(dρdtI); X0 = zero(XT)
    nchunks = nthreads()  # Assuming nchunks is defined somewhere as nthreads()

    @inbounds @no_escape begin
        local_X = @alloc(XT, XL, nchunks)

        fill!(local_X,X0)

        # Directly iterate over the chunks
        for ichunk in 1:nchunks
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
        for ix in 1:XL
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

    V           = zeros(SVector{2,Float64},NumberOfPoints)
    VL          = rand(eltype(V),NumberOfInterations)

    VD          = DimensionalData{2, Float64}(NumberOfPoints)
    VDL         = DimensionalData{2, Float64}(NumberOfInterations)
    VDL.V      .= VL


    V .*= 0
    ReductionFunctionChunk!(V,I,J,VL,VL)
    println("Value when doing chunk svector reduction: ", sum(V))

    println(@report_opt    ReductionFunctionChunk!(V,I,J,VL,VL))
    println(@report_call   ReductionFunctionChunk!(V,I,J,VL,VL))
    #println(@code_warntype ReductionFunctionChunk!(V,I,J,VL,VL))

    VD.V .*= 0
    ReductionFunctionChunk!(VD.V,I,J,VDL.V,VDL.V)
    println("Value when doing chunk svector reduction with dimensional data: ", sum(VD.V))


    println(@report_opt    ReductionFunctionChunk!(VD.V,I,J,VDL.V,VDL.V))
    println(@report_call   ReductionFunctionChunk!(VD.V,I,J,VDL.V,VDL.V))
    #println(@code_warntype ReductionFunctionChunk!(VD.V,I,J,VDL.V,VDL.V))

    VD.V .*= 0
    ReductionFunctionChunkNoBatch!(VD.V,I,J,VDL.V,VDL.V)
    println("Value when doing chunk svector reduction with dimensional data and no @batch: ", sum(VD.V))


    println(@report_opt    ReductionFunctionChunkNoBatch!(VD.V,I,J,VDL.V,VDL.V))
    println(@report_call   ReductionFunctionChunkNoBatch!(VD.V,I,J,VDL.V,VDL.V))
    # println(@code_warntype ReductionFunctionChunkNoBatch!(VD.V,I,J,VDL.V,VDL.V))

    # println("Chunk function:")
    # display(@benchmark ReductionFunctionChunk!($V,$I,$J,$VL,$VL))

    # println("Chunk function with dimensional data:")
    # display(@benchmark ReductionFunctionChunk!($(VD.V),$I,$J,$(VDL.V),$(VDL.V)))

    # println("Chunk function with dimensional data and no @batch:")
    # display(@benchmark ReductionFunctionChunkNoBatch!($(VD.V),$I,$J,$(VDL.V),$(VDL.V)))

end