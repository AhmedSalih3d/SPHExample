#https://jenni-westoby.github.io/Julia_GPU_examples/dev/Vector_dot_product/

using CUDA
using Test
using BenchmarkTools

function dot(a,b,c, N, threadsPerBlock)

    # Set up shared memory cache for this current block.
    cache = @cuDynamicSharedMem(eltype(a), threadsPerBlock)

    # Initialise some variables.
    tid = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x
    totalThreads = blockDim().x * gridDim().x
    cacheIndex = threadIdx().x - 1
    temp = 0

    # Iterate over vector to do dot product in parallel way
    while tid < N
        temp += a[tid + 1] * b[tid + 1]
        tid += totalThreads
    end

    # set cache values
    cache[cacheIndex + 1] = temp

    # synchronise threads
    sync_threads()

    # In the step below, we add up all of the values stored in the cache
    i = blockDim().x ÷ 2
    while i!=0
        if cacheIndex < i
            cache[cacheIndex + 1] += cache[cacheIndex + i + 1]
        end
        sync_threads()
        i = i ÷ 2
    end

    # cache[1] now contains the sum of vector dot product calculations done in
    # this block, so we write it to c
    if cacheIndex == 0
        c[blockIdx().x] = cache[1]
    end

    return nothing
end


let
    # Initialise variables
    N                    = 33 * 1024
    threadsPerBlock      = 256
    blocksPerGrid::Int   = min(32, (N + threadsPerBlock - 1) / threadsPerBlock)

    # Create a,b and c
    a = CuArray(fill(0, N))
    b = CuArray(fill(0, N))
    c = CuArray(fill(0, blocksPerGrid))

    # Fill a and b
    CUDA.@allowscalar for i in 1:N
        a[i] = i
        b[i] = 2*i
    end

    # Execute the kernel. Note the shmem argument - this is necessary to allocate
    # space for the cache we allocate on the gpu with @cuDynamicSharedMem
    @cuda blocks = blocksPerGrid threads = threadsPerBlock shmem = (threadsPerBlock * sizeof(eltype(a))) dot(a,b,c, N, threadsPerBlock)

    # Copy c back from the gpu (device) to the host
    c = Array(c)

    local result = 0

    # Sum the values in c
    for i in 1:blocksPerGrid
        result += c[i]
    end

    # Check whether output is correct
    # println("Does GPU value ", result, " = ", 2 * sum_squares(N - 1)) sum_squares was not defined in original code
    # so compare with cpu directly
    cpu_sum = sum(Array(a) .* Array(b))
    println("Does GPU value ", result, " = ", cpu_sum)
    @test result == cpu_sum
end

# For benchmark
let 
    # Initialise variables
    N                    = 33 * 1024
    threadsPerBlock      = 256
    blocksPerGrid::Int   = min(32, (N + threadsPerBlock - 1) / threadsPerBlock)

    # Create a,b and c
    a = CuArray(fill(0, N))
    b = CuArray(fill(0, N))
    c = CuArray(fill(0, blocksPerGrid))

    # Fill a and b
    CUDA.@allowscalar for i in 1:N
        a[i] = i
        b[i] = 2*i
    end

    # Execute the kernel. Note the shmem argument - this is necessary to allocate
    # space for the cache we allocate on the gpu with @cuDynamicSharedMem
    println(CUDA.@allocated @cuda blocks = blocksPerGrid threads = threadsPerBlock shmem = (threadsPerBlock * sizeof(eltype(a))) dot(a,b,c, N, threadsPerBlock))
    println(CUDA.@profile trace=true @cuda blocks = blocksPerGrid threads = threadsPerBlock shmem = (threadsPerBlock * sizeof(eltype(a))) dot(a,b,c, N, threadsPerBlock))
    bench = @benchmark CUDA.@sync @cuda  blocks = $blocksPerGrid threads = $threadsPerBlock shmem = ($threadsPerBlock * sizeof(eltype($a))) dot($a,$b,$c, $N, $threadsPerBlock)
    display(bench)
end