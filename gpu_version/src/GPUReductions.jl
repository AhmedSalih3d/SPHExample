"""
Small custom GPU reductions.

`mapreduce` from CUDA.jl does not compile for tuple or `SVector` valued
reductions, and a scalar `mapreduce` launches several kernels. The time step
loop needs, once per step, three quantities that all come from the same pass
over the particles (viscous time step limit, force time step limit and the
maximum displacement since the last cell list update). This module provides a
single-pass block reduction with an `SVector` accumulator and a tiny host side
finish, so that one kernel launch and one device to host copy suffice. The
block level part (`block_reduce_store!`) can also be embedded in other
kernels so that the reduction rides along with an element-wise pass.
"""
module GPUReductions

using CUDA
using StaticArrays

export ReductionWorkspace, reduce_svector, launch_reduce!, finish_reduction, block_reduce,
       block_reduce_store!, REDUCE_THREADS

const REDUCE_THREADS = 256
const REDUCE_MAX_BLOCKS = 512

"""
    ReductionWorkspace{V}(n)

Preallocated device and host buffers for `reduce_svector`. `V` is the
`SVector` type of the accumulator and `n` the number of elements that will be
reduced (used to size the number of blocks). The host buffer is pinned so that
the device to host copy of the partial results is a plain DMA transfer.
"""
struct ReductionWorkspace{V}
    partial::CuVector{V}
    host::Vector{V}
    nblocks::Int
end

function ReductionWorkspace{V}(n::Integer) where {V}
    nblocks = max(1, min(REDUCE_MAX_BLOCKS, cld(n, REDUCE_THREADS)))
    host = Vector{V}(undef, nblocks)
    try
        CUDA.pin(host)
    catch
        # pinning is an optimisation only
    end
    return ReductionWorkspace{V}(CuVector{V}(undef, nblocks), host, nblocks)
end

"""
    block_reduce(acc, op) -> V

Reduce the per thread accumulators `acc` of the current block with an
associative, commutative `op`. Must be called by all threads of a one-dimensional
block of at most `REDUCE_THREADS` threads; every thread receives the block
result. Float32/Float64 `SVector` accumulators use warp shuffles for whole-warp
blocks; other types and partial-warp blocks use a shared-memory tree.
Shared storage is safe to reuse immediately after the call.
"""
@inline block_reduce(acc, op) = block_reduce_shared(acc, op)

@inline function block_reduce_shared(acc::V, op::O) where {V, O}
    tid = threadIdx().x
    shmem = CuStaticSharedArray(V, REDUCE_THREADS)
    @inbounds shmem[tid] = acc
    sync_threads()

    s = Int32(REDUCE_THREADS ÷ 2)
    while s >= Int32(1)
        if tid <= s && tid + s <= blockDim().x
            @inbounds shmem[tid] = op(shmem[tid], shmem[tid + s])
        end
        sync_threads()
        s ÷= Int32(2)
    end

    @inbounds r = shmem[1]
    sync_threads()
    return r
end

@inline warp_shuffle_down(acc::SVector, offset) =
    map(value -> CUDA.shfl_down_sync(0xffffffff, value, offset), acc)

@inline function warp_reduce(acc::SVector, op::O, lane, count) where {O}
    offset = Int32(16)
    while offset >= Int32(1)
        if offset < count
            other = warp_shuffle_down(acc, offset)
            if count == Int32(32) || lane + offset <= count
                acc = op(acc, other)
            end
        end
        offset ÷= Int32(2)
    end
    return acc
end

@inline function block_reduce(acc::SVector{N,T}, op::O) where {N,T<:Union{Float32,Float64},O}
    blockDim().x % Int32(32) == Int32(0) || return block_reduce_shared(acc, op)
    tid = threadIdx().x
    lane = (tid - Int32(1)) % Int32(32) + Int32(1)
    warp = (tid - Int32(1)) ÷ Int32(32) + Int32(1)
    nwarps = blockDim().x ÷ Int32(32)
    shmem = CuStaticSharedArray(SVector{N,T}, REDUCE_THREADS ÷ 32)

    acc = warp_reduce(acc, op, lane, Int32(32))
    if lane == Int32(1)
        @inbounds shmem[warp] = acc
    end
    sync_threads()

    if warp == Int32(1)
        if lane <= nwarps
            @inbounds acc = shmem[lane]
        end
        acc = warp_reduce(acc, op, lane, nwarps)
        if lane == Int32(1)
            @inbounds shmem[1] = acc
        end
    end
    sync_threads()
    @inbounds r = shmem[1]
    # Every reader must finish before a subsequent reduction reuses this storage.
    sync_threads()
    return r
end

"""
    block_reduce_store!(partial, acc, op)

`block_reduce` followed by a store of the block result in
`partial[blockIdx().x]`.
"""
@inline function block_reduce_store!(partial, acc::V, op::O) where {V, O}
    r = block_reduce(acc, op)
    if threadIdx().x == Int32(1)
        @inbounds partial[blockIdx().x] = r
    end
    return nothing
end

# Every thread reduces a strided subset of the elements into a register, the
# block then reduces and writes a single value.
function reduce_kernel!(partial, f::F, op::O, init, n::Int32, args...) where {F, O}
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    acc = init
    while i <= n
        acc = op(acc, f(i, args...))
        i += stride
    end
    block_reduce_store!(partial, acc, op)
    return nothing
end

"""
    finish_reduction(ws, op, init) -> V

Copy the per block partial results to the host (synchronizes the device) and
combine them. Used after a kernel that called `block_reduce_store!` with
`ws.nblocks` blocks.
"""
function finish_reduction(ws::ReductionWorkspace{V}, op::O, init::V) where {V, O}
    nblocks = ws.nblocks
    copyto!(ws.host, 1, ws.partial, 1, nblocks)
    acc = init
    @inbounds for b in 1:nblocks
        acc = op(acc, ws.host[b])
    end
    return acc
end

"""
    launch_reduce!(ws, f, op, init, n, args...)

Launch the reduction of `op` over `f(i, args...)` for `i in 1:n` into the
per block partial results `ws.partial` without waiting for it. Combine them
with `finish_reduction` (host) or a device side consumer.
"""
function launch_reduce!(ws::ReductionWorkspace{V}, f::F, op::O, init::V, n::Integer,
                        args...) where {V, F, O}
    if n == 0
        fill!(ws.partial, init)
        return nothing
    end
    @cuda threads=REDUCE_THREADS blocks=ws.nblocks reduce_kernel!(ws.partial, f, op, init,
                                                                   Int32(n), args...)
    return nothing
end

"""
    reduce_svector(ws, f, op, init, n, args...)

Compute `op` over `f(i, args...)` for `i in 1:n` on the GPU. `f` receives the
device arrays in `args` and must return the same `SVector` type as `init`.
Returns the final accumulator.
"""
function reduce_svector(ws::ReductionWorkspace{V}, f::F, op::O, init::V, n::Integer,
                        args...) where {V, F, O}
    n == 0 && return init
    launch_reduce!(ws, f, op, init, n, args...)
    return finish_reduction(ws, op, init)
end

end # module
