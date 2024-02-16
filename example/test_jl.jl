using Base.Threads
using Bumper
using StrideArrays # Not necessary, but can make operations like broadcasting with Bumper.jl faster.
using Polyester
using BenchmarkTools
using LoopVectorization
using ChunkSplitters

function NaiveReductionFunction!(dρdtI, I,J,drhopLp,drhopLn)
    # Reduction
    for iter in eachindex(I,J)
        i = I[iter]
        j = J[iter]

        dρdtI[i] +=  drhopLp[iter]
        dρdtI[j] +=  drhopLn[iter]
    end
end

function ReductionFunction!(dρdtI,I,J,drhopLp,drhopLn, buf)
    # Set up a scope where memory may be allocated, and does not escape:
    @no_escape buf begin
        # Allocate a `PtrArray` (see StrideArraysCore.jl) using memory from the default buffer.
        XT = eltype(dρdtI); XL = length(dρdtI)

        # How can I do this so that I don't have to manually hard code to 4 threads?
        # If I do, local_X = [@alloc(XT,XL) for _ in 1:nthreads()] then it will allocate!
        local_X1 = @alloc(XT, XL)
        local_X2 = @alloc(XT, XL)
        local_X3 = @alloc(XT, XL)
        local_X4 = @alloc(XT, XL)
        local_X  = (local_X1, local_X2, local_X3, local_X4)

        # I thought Bumper.jl would automatically reset the buffer?
        for x in local_X
            x .*= zero(XT)
        end

        # Remove @batch here to verify the two functions give the same answer..
        # @batch seems to only use 1 thread, @tturbo uses all four?
        @batch for iter in eachindex(I,J)
            i = I[iter]
            j = J[iter]

            # Accumulate in thread-local storage
            local_X[threadid()][i] +=  drhopLp[iter]
            local_X[threadid()][j] +=  drhopLn[iter]
        end

        # Reduce the thread-local storage into the shared arrays
        for tid in 1:nthreads()
            dρdtI .+= local_X[tid]
        end

    end

    return nothing
end

function ChunkFunction!(dρdtI, I,J,drhopLp,drhopLn)
    nchunks = nthreads()
    local_X = [zeros(length(drhopLp)) for _ in 1:nchunks]
    @threads for (ichunk, inds) in enumerate(chunks(dρdtI; n=nchunks))
        i = I[inds]
        j = J[inds]

        @. local_X[ichunk][i] += drhopLp[i]
        @. local_X[ichunk][j] += drhopLn[j]
    end
end


begin
    buf                 = default_buffer()
    NumberOfPoints      = 6195
    NumberOfInterations = 100000
    I           = rand(1:NumberOfPoints, NumberOfInterations)
    J           = rand(1:NumberOfPoints, NumberOfInterations)
    dρdtI       = zeros(NumberOfPoints) 
    drhopLp     = rand(1.0:2.0, NumberOfInterations)    
    drhopLn     = rand(1.0:2.0, NumberOfInterations)    

    NaiveReductionFunction!(dρdtI,I,J,drhopLp,drhopLn)
    println("Value when doing naive reduction: ", sum(dρdtI))
    dρdtI .= zero(eltype(dρdtI)); #Just resetting array.
    ReductionFunction!(dρdtI, I, J, drhopLp,drhopLn, buf)
    println("Value doing optimized reduction: ",  sum(dρdtI))

    # dρdtI .= zero(eltype(dρdtI)); #Just resetting array.
    # dρdtI[I] .+= drhopLp[I]
    # dρdtI[J] .+= drhopLp[J]
    # println("Value doing fused: ",  sum(dρdtI))

    ChunkFunction!(dρdtI, I,J,drhopLp,drhopLn)
    println("Value doing chunks: ",  sum(dρdtI))


    println("Naive: ")
    display(@benchmark NaiveReductionFunction!($dρdtI , $I, $J, $drhopLp,drhopLn))
    println("Optimized: ")
    display(@benchmark ReductionFunction!($dρdtI , $I, $J, $drhopLp,drhopLn, $buf))
    println("Using ChunkSplitters: ")
    display(@benchmark ChunkFunction!($dρdtI , $I, $J , $drhopLp,drhopLn))
    println("I have threads enabled: ", nthreads())
end