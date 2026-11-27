using Test
using SPHExampleGPU
using CUDA
using StaticArrays

@testset "Combined kernel value and gradient" begin
    kernels = SPHExampleGPU.SPHKernels
    for T in (Float32, Float64), D in (2, 3), shape in (WendlandC2(), CubicSpline{T}())
        @testset "$T, $(D)D, $(typeof(shape))" begin
            kernel = SPHKernelInstance{D,T}(shape; h = T(0.037))
            qs = T[0, eps(T), 0.4, prevfloat(one(T)), 1, nextfloat(one(T)), prevfloat(T(2)), 2]
            displacements = [SVector{D,T}(ntuple(i -> i == 1 ? q * kernel.h : zero(T), D)) for q in qs]
            evaluate = kernels.kernel_value_and_gradient
            host = evaluate.(Ref(kernel), qs, displacements)
            results = CUDA.functional() ?
                (host, Array(evaluate.(Ref(kernel), CuArray(qs), CuArray(displacements)))) : (host,)
            for result in results, (q, x, (value, gradient)) in zip(qs, displacements, result)
                @test isapprox(value, Wᵢⱼ(kernel, q); rtol = 8eps(T), atol = 0)
                @test isapprox(gradient[1], ∇Wᵢⱼ(kernel, q, x)[1]; rtol = 8eps(T), atol = 0)
                @test all(iszero, gradient[2:end])
            end
        end
    end
end
