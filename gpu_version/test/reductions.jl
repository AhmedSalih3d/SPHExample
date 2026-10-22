module ReductionTests

using Test, SPHExampleGPU, CUDA, StaticArrays
using SPHExampleGPU.GPUReductions
using SPHExampleGPU.GPUKernels: step_reduce

@inline input_value(i, values) = @inbounds values[i]

function broadcast_kernel!(output, values, op, passes)
    tid = threadIdx().x
    i = (blockIdx().x - Int32(1)) * blockDim().x + tid
    acc = @inbounds values[i]
    for pass in 1:passes
        acc = block_reduce(acc, op)
        @inbounds output[i, pass] = acc
        # All lanes consume the result, then contribute different next inputs.
        acc = acc + oftype(acc, ntuple(_ -> tid + pass, length(acc)))
    end
    return nothing
end

@inline tuple_reduce(a, b) = (max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))

@testset "block reductions" begin
    @test CUDA.functional()
    @testset "max/min/max, both precisions and all launch shapes" begin
        for T in (Float32, Float64)
            init = SVector{3,T}(0, Inf, 0)
            for n in (0, 1, 7, 31, 32, 33, 63, 64, 65, 255, 256, 257, 511, 513, 4099,
                      REDUCE_THREADS * 512 + 1)
                values = [SVector{3,T}(i % 97, i % 5 == 0 ? Inf : 1 / (i % 101 + 1), i % 79 / 1024) for i in 1:n]
                expected = foldl(step_reduce, values; init)
                ws = ReductionWorkspace{typeof(init)}(n)
                @test isequal(reduce_svector(ws, input_value, step_reduce, init, n, CuArray(values)), expected)
                launch_reduce!(ws, input_value, step_reduce, init, n, CuArray(values))
                @test isequal(finish_reduction(ws, step_reduce, init), expected)
            end
            for special in (SVector{3,T}(Inf, -Inf, Inf), SVector{3,T}(NaN, NaN, NaN),
                            SVector{3,T}(-0.0, -0.0, -0.0))
                values = fill(init, 257)
                values[33] = special
                ws = ReductionWorkspace{typeof(init)}(length(values))
                expected = foldl(step_reduce, values; init)
                @test isequal(reduce_svector(ws, input_value, step_reduce, init, length(values), CuArray(values)), expected)
            end
            ws = ReductionWorkspace{typeof(init)}(2)
            example = CuArray([SVector{3,T}(3, 0.8, 0.02), SVector{3,T}(5, 0.4, 0.01)])
            @test reduce_svector(ws, input_value, step_reduce, init, 2, example) == SVector{3,T}(5, 0.4, 0.02)
            @test reduce_svector(ws, input_value, step_reduce, init, 33, CUDA.fill(init, 33)) == init
        end
    end

    @testset "all-thread broadcast and consecutive shared-storage reuse" begin
        for T in (Float32, Float64, Int32), threads in (1, 2, 7, 16, 31, 32, 33, 63, 64, 96, 128, 224, 255, 256)
            values = [SVector{3,T}(i % 97, 500 - i, i % 79) for i in 1:2threads]
            passes = 32
            output = CuArray{SVector{3,T}}(undef, 2threads, passes)
            @cuda threads=threads blocks=2 broadcast_kernel!(output, CuArray(values), step_reduce, passes)
            actual = Array(output)
            for block in 0:1
                range = block * threads + 1:(block + 1) * threads
                inputs = values[range]
                for pass in 1:passes
                    expected = foldl(step_reduce, inputs)
                    @test all(v -> isequal(v, expected), actual[range, pass])
                    inputs = [expected + SVector{3,T}(ntuple(_ -> tid + pass, 3)) for tid in 1:threads]
                end
            end
        end
    end

    @testset "generic non-shuffle accumulators" begin
        init = (Int32(0), typemax(Int32), Int32(0))
        values = [(Int32(i % 97), Int32(500 - i), Int32(i % 79)) for i in 1:257]
        ws = ReductionWorkspace{typeof(init)}(length(values))
        @test reduce_svector(ws, input_value, tuple_reduce, init, length(values), CuArray(values)) ==
            foldl(tuple_reduce, values; init)
        init2 = SVector(-Inf, Inf)
        values2 = [SVector(Float64(i), -Float64(i)) for i in 1:513]
        op(a, b) = SVector(max(a[1], b[1]), min(a[2], b[2]))
        ws2 = ReductionWorkspace{typeof(init2)}(length(values2))
        @test reduce_svector(ws2, input_value, op, init2, length(values2), CuArray(values2)) == SVector(513.0, -513.0)
    end
end

end
