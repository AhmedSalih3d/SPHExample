using Test
using SPHExampleGPU

@testset "dimension-aware simulation constants" begin
    for T in (Float32, Float64)
        legacy = SimulationConstants{T}()
        planar = @inferred SimulationConstants{T}(2)
        spatial = @inferred SimulationConstants{T}(3)
        @test planar === legacy
        @test planar.m₀ === T(1000 * 0.02^2)
        @test spatial.m₀ === T(1000 * 0.02^3)
        @test SimulationConstants{T}(Int32(3)) === spatial

        for D in (2, 3), (density, spacing) in ((1050, 0.03), (T(1017), T(0.04)))
            constants = @inferred SimulationConstants{T}(D; ρ₀ = density, dx = spacing,
                g = 0, c₀ = 30, γ = 5, α = 0.02, ν₀ = 2e-6)
            @test constants isa SimulationConstants{T}
            @test constants.m₀ === T(density * spacing^D)
            @test constants.ρ₀ === T(density)
            @test constants.dx === T(spacing)
            @test constants.g === zero(T)
            @test constants.α === T(0.02)
            @test constants.ν₀ === T(2e-6)
            @test constants.γ⁻¹ ≈ inv(constants.γ)
            @test constants.Cb ≈ constants.c₀^2 * constants.ρ₀ / constants.γ
            @test constants.Cb⁻¹ ≈ inv(constants.Cb)
            if D == 2
                @test constants === SimulationConstants{T}(; ρ₀ = density, dx = spacing,
                    g = 0, c₀ = 30, γ = 5, α = 0.02, ν₀ = 2e-6)
            end

            explicit = SimulationConstants{T}(D; ρ₀ = density, dx = spacing, m₀ = 0.125,
                Cb = 120000, γ⁻¹ = 0.25)
            @test explicit.m₀ === T(0.125)
            @test explicit.Cb === T(120000)
            @test explicit.Cb⁻¹ ≈ inv(explicit.Cb)
            @test explicit.γ⁻¹ === T(0.25)
        end

        for D in (-1, 0, 1, 4)
            @test_throws ArgumentError SimulationConstants{T}(D)
            @test_throws ArgumentError SimulationConstants{T}(D; m₀ = 0.125)
        end
    end
end
