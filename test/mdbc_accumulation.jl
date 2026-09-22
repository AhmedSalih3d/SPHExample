using Test
using SPHExample
using StaticArrays
using LinearAlgebra: dot, norm

@testset "MDBC accumulates accepted fluid neighbors" begin
    for D in (2, 3), T in (Float32, Float64), Model in (WendlandC2(), CubicSpline{T}())
        Constants = SimulationConstants{T}(dx=T(0.02), m₀=T(1000) * T(0.02)^D)
        Kernel = SPHKernelInstance{D,T}(Model; dx=Constants.dx)
        MetaData = SimulationMetaData{D,T,NoShifting,NoKernelOutput,SimpleMDBC}(
            SimulationName="mdbc_accumulation", SaveLocation=".", OutputTimes=T(0.01))
        N = D + 1
        GhostPoints = [SVector{D,T}(ntuple(d -> T(d) / T(10), D))]
        Direction = SVector{D,T}(ntuple(d -> T(d), D))
        Direction /= norm(Direction)
        Position = copy(GhostPoints)
        Density = T[1010]
        Types = ParticleType[Fluid]
        StartB = SVector{N,T}(ntuple(i -> T(i) / T(7), N))
        StartA = SMatrix{N,N,T}(ntuple(i -> T(i) / T(11), N * N))
        Interact = SPHExample.SPHCellList.ComputeInteractionsMDBC!

        for Scale in (T(0.25), T(0.75))
            Position[1] = GhostPoints[1] + Scale * Kernel.H * Direction
            Separation = GhostPoints[1] - Position[1]
            q = sqrt(dot(Separation, Separation)) * Kernel.h⁻¹
            W = SPHExample.SPHKernels.Wᵢⱼ(Kernel, q)
            Gradient = SPHExample.SPHKernels.∇Wᵢⱼ(Kernel, q, Separation)
            ExpectedB = SVector{N,T}(Constants.m₀ * W, (Constants.m₀ * Gradient)...)
            FirstColumn = (Constants.m₀ / Density[1]) * SVector{N,T}(W, Gradient...)
            # Construct matrix entries directly, independently of the production outer product.
            ExpectedA = SMatrix{N,N,T}(ntuple(N * N) do Index
                Row, Column = mod1(Index, N), cld(Index, N)
                Column == 1 ? FirstColumn[Row] : -Separation[Column - 1] * FirstColumn[Row]
            end)
            Contribution = Interact(Kernel, MetaData, Constants, Position, Density, Types, GhostPoints, 1, 1)
            Accumulated = Interact(Kernel, MetaData, Constants, Position, Density, Types, GhostPoints, 1, 1, StartB, StartA)
            @test isapprox(Contribution[1], ExpectedB; rtol=T(16) * eps(T))
            @test isapprox(Contribution[2], ExpectedA; rtol=T(16) * eps(T))
            @test isequal(Accumulated, (StartB + Contribution[1], StartA + Contribution[2]))
        end

        for Type in (Fixed, Moving, Fluid)
            Types[1] = Type
            Scale = Type == Fluid ? T(1.5) : T(0.25)
            Position[1] = GhostPoints[1] + Scale * Kernel.H * Direction
            Result = Interact(Kernel, MetaData, Constants, Position, Density, Types, GhostPoints, 1, 1, StartB, StartA)
            @test isequal(Result, (StartB, StartA))
            @test isequal(Interact(Kernel, MetaData, Constants, Position, Density, Types, GhostPoints, 1, 1),
                (zero(StartB), zero(StartA)))
        end
    end
end
