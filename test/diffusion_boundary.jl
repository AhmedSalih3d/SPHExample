using Test
using SPHExample
using StaticArrays
using StructArrays

@testset "diffusion boundary exclusions" begin
    for D in (2, 3), T in (Float32, Float64)
        Constants = SimulationConstants{T}(dx=T(0.02), c₀=T(30))
        Kernel = SPHKernelInstance{D,T}(WendlandC2(); dx=Constants.dx)
        Displacement = SVector{D,T}(ntuple(Axis -> Axis == 1 ? T(0.01) : zero(T), D))
        Gradient = -Displacement
        DistanceSquared = sum(abs2, Displacement)
        Particles = StructArray((Density=T[1001, 1003], Type=ParticleType[Fluid, Fluid]))
        # The existing inverse-hydrostatic estimator used by the complex model
        # relies on Float64 bit layout. Cover its supported precision here.
        Models = T === Float64 ? (LinearDensityDiffusion(), ComplexDensityDiffusion()) : (LinearDensityDiffusion(),)
        for Model in Models
            for Typeᵢ in (Fluid, Fixed, Moving), Typeⱼ in (Fluid, Fixed, Moving)
                Particles.Type .= (Typeᵢ, Typeⱼ)
                Dᵢ, Dⱼ = compute_density_diffusion(Model, Kernel, Constants, Particles,
                    Displacement, Gradient, DistanceSquared, 1, 2, Particles.Type)
                @test typeof(Dᵢ) === T
                @test Dⱼ == -Dᵢ
                @test iszero(Dᵢ) == (Typeᵢ != Fluid || Typeⱼ != Fluid)
            end
            @test_throws BoundsError compute_density_diffusion(Model, Kernel, Constants, Particles,
                Displacement, Gradient, DistanceSquared, 1, 3, Particles.Type)
        end
        # This separate model intentionally applies diffusion at boundaries.
        @test !iszero(first(compute_density_diffusion(ZeroGravityLinearDensityDiffusion(),
            Kernel, Constants, Particles, Displacement, Gradient, DistanceSquared, 1, 2, Particles.Type)))
    end
end
