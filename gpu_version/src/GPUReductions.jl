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

export ReductionWorkspace, reduce_svector, finish_reduction, block_reduce_store!,
       REDUCE_THREADS

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
    block_reduce_store!(partial, acc, op)

Reduce the per thread accumulators `acc` of the current block with `op`
through shared memory and store the block result in `partial[blockIdx().x]`.
Must be called by all threads of a block of exactly `REDUCE_THREADS` threads.
"""
@inline function block_reduce_store!(partial, acc::V, op::O) where {V, O}
    tid = threadIdx().x
    shmem = CuStaticSharedArray(V, REDUCE_THREADS)
    @inbounds shmem[tid] = acc
    sync_threads()

    s = Int32(REDUCE_THREADS ÷ 2)
    while s >= Int32(1)
        if tid <= s
            @inbounds shmem[tid] = op(shmem[tid], shmem[tid + s])
        end
        sync_threads()
        s ÷= Int32(2)
    end

    if tid == Int32(1)
        @inbounds partial[blockIdx().x] = shmem[1]
    end
    return nothing
end

# Every thread reduces a strided subset of the elements into a register, the
# block then reduces through shared memory and writes a single value.
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
    reduce_svector(ws, f, op, init, n, args...)

Compute `op` over `f(i, args...)` for `i in 1:n` on the GPU. `f` receives the
device arrays in `args` and must return the same `SVector` type as `init`.
Returns the final accumulator.
"""
function reduce_svector(ws::ReductionWorkspace{V}, f::F, op::O, init::V, n::Integer,
                        args...) where {V, F, O}
    n == 0 && return init
    @cuda threads=REDUCE_THREADS blocks=ws.nblocks reduce_kernel!(ws.partial, f, op, init,
                                                                   Int32(n), args...)
    return finish_reduction(ws, op, init)
end

end # module
