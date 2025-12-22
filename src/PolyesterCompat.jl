module PolyesterCompat

using Base.Threads

"""
    Polyester.batch(iterable; threads=Threads.nthreads(), batch_size=nothing)

Lightweight, allocation-free batching helper that mirrors the key interface of
`Polyester.batch`. It splits `iterable` into contiguous ranges sized for the
available threads.
"""
module Polyester
    using Base.Threads

    struct StaticBatch
        len::Int
        chunk::Int
    end

    Base.eltype(::Type{StaticBatch}) = UnitRange{Int}

    @inline function Base.length(b::StaticBatch)
        return b.len == 0 ? 0 : cld(b.len, b.chunk)
    end

    @inline Base.iterate(b::StaticBatch) = iterate(b, 1)
    @inline function Base.iterate(b::StaticBatch, state::Int)
        state > b.len && return nothing

        stop = min(state + b.chunk - 1, b.len)
        return (state:stop, stop + 1)
    end

    @inline function chunk_range(b::StaticBatch, chunk_id::Int)
        start = (chunk_id - 1) * b.chunk + 1
        stop = min(chunk_id * b.chunk, b.len)
        return start:stop
    end

    @inline function batch(len::Int; threads::Int = Threads.nthreads(),
                           batch_size::Union{Nothing, Int} = nothing)
        chunk = something(batch_size, threads == 0 ? len : cld(len, threads))
        return StaticBatch(len, max(chunk, 1))
    end

    @inline function batch(iterable; threads::Int = Threads.nthreads(),
                           batch_size::Union{Nothing, Int} = nothing)
        return batch(length(iterable); threads, batch_size)
    end
end

export Polyester, foreach_batch

@inline function foreach_batch(batches::Polyester.StaticBatch, f)
    nbatches = length(batches)

    @inbounds Threads.@threads for chunk_id in 1:nbatches
        range = Polyester.chunk_range(batches, chunk_id)
        f(chunk_id, range)
    end

    return nothing
end

end
