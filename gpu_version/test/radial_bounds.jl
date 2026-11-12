using Test, SPHExampleGPU, CUDA, StaticArrays

const DD = SPHExampleGPU.SPHDensityDiffusionModels

@testset "radial diffusion standalone bounds and gating" begin
    for D in (2, 3), T in (Float32, Float64)
        constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(42), δᵩ = T(0.1))
        kernel = SPHKernelInstance{D,T}(WendlandC2(); dx = constants.dx)
        x = SVector{D,T}(ntuple(_ -> T(0.01), D))
        d2 = sum(abs2, x)
        args = (kernel, constants, x, d2, T(1007), T(995), inv(T(1007)), inv(T(995)))
        factor = SPHExampleGPU.SPHKernels.radial_gradient_factor(kernel, sqrt(d2) * kernel.h⁻¹)
        inverse_distance = inv(d2 + kernel.η²)
        for model in (LinearDensityDiffusion(), ZeroGravityLinearDensityDiffusion())
            call(types, i, j) = DD.radial_density_diffusion(model, args..., i, j, types, factor, inverse_distance)
            if model isa LinearDensityDiffusion
                @test_throws BoundsError call([Fluid, Fluid], 0, 1)
                @test_throws BoundsError call([Fluid, Fluid], 1, 3)
            end
            value = call([Fluid, Fluid], 1, 2)
            @test value[2] == -value[1]
            if model isa LinearDensityDiffusion
                @test call([Fluid, Fixed], 1, 2) == (zero(T), zero(T))
                @test call([Fixed, Fluid], 1, 2) == (zero(T), zero(T))
            else
                @test call([Fluid, Fixed], 1, 2) == value
                @test call([Fixed, Fluid], 1, 2) == value
            end
        end
    end
end
