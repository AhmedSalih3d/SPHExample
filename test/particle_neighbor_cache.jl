using Test
using SPHExample
using StaticArrays
using LinearAlgebra: norm

ParticleCacheNeighbors(Cache, i) = view(Cache.Neighbors, Cache.Offsets[i]:(Cache.Offsets[i + 1] - 1))

function CheckParticleCacheReference(Case, Cache, Diffusion, Viscosity,
                                     Shifting=NoShifting, KernelOutput=NoKernelOutput)
    Expected = CheckedInteractionReference(Case, Diffusion, Viscosity, Shifting, KernelOutput)
    (; Constants, Kernel, MetaData, Particles, ParticleRanges, CellListIndices,
       NeighborCellLists, Position, Density, Velocity, Pressure) = Case
    T = eltype(Density)
    VectorType = eltype(Position)
    Count = length(Position)
    DensityRate = fill(T(-9), Count)
    Acceleration = fill(VectorType(ntuple(_ -> T(-9), length(VectorType))), Count)
    AccelerationMax = fill(T(-9), Count)
    ShiftC = copy(Acceleration)
    ShiftR = copy(DensityRate)
    SPHExample.SPHCellList.EvaluateInteractions!(
        Cache, Diffusion, Viscosity, Kernel, MetaData, Constants, Particles,
        ParticleRanges, CellListIndices, NeighborCellLists, DensityRate,
        Acceleration, ShiftC, ShiftR, AccelerationMax;
        Position, Density, Pressure, Velocity,
    )
    for (Actual, Reference) in ((DensityRate, Expected.DensityRate),
                                (Acceleration, Expected.Acceleration),
                                (Particles.Kernel, Expected.KernelValues),
                                (Particles.KernelGradient, Expected.KernelGradients),
                                (ShiftC, Expected.ShiftC), (ShiftR, Expected.ShiftR))
        @test all(isapprox.(Actual, Reference; rtol=8eps(T), atol=8eps(T) * maximum(norm, Reference)))
    end
    @test AccelerationMax ≈ norm.(Acceleration) rtol=8eps(T)
    return (; DensityRate, Acceleration)
end

@testset "particle neighbor cache matches scalar interactions" begin
    Cases = (
        (2, Float32, WendlandC2(), ZeroDensityDiffusion(), ZeroViscosity()),
        (2, Float64, CubicSpline{Float64}(), LinearDensityDiffusion(), ArtificialViscosity()),
        (3, Float32, CubicSpline{Float32}(), ZeroGravityLinearDensityDiffusion(), Laminar()),
        (3, Float64, WendlandC2(), ComplexDensityDiffusion(), LaminarSPS()),
    )
    for (D, T, KernelModel, Diffusion, Viscosity) in Cases
        @testset "$D dimensions, $T" begin
            Case = InteractionReferenceCase(Val(D), T, KernelModel, NoShifting, NoKernelOutput;
                                            Packed=D == 3, Copies=D == 3 && T === Float64 ? 23 : 1)
            Cache = SPHExample.SPHCellList.ParticleNeighborCache(Case.Position)
            CheckParticleCacheReference(Case, Cache, Diffusion, Viscosity)
            ReferencePosition = copy(Cache.ReferencePosition)
            OriginalPosition = copy(Case.Particles.Position)
            # Evaluate a distinct midpoint buffer while retaining the grid and
            # cached particle indices from the accepted particle state.
            Position = [X + SVector{D,T}(ntuple(d -> T(isodd(i + d) ? 1 : -1) * Case.Kernel.H / 128, D))
                        for (i, X) in enumerate(Case.Position)]
            CheckParticleCacheReference((; Case..., Position), Cache, Diffusion, Viscosity)
            @test Cache.ReferencePosition == ReferencePosition
            @test Case.Particles.Position == OriginalPosition
            Position .+= Ref(SVector{D,T}(ntuple(d -> d == 1 ? Case.Kernel.H / 4 : zero(T), D)))
            CheckParticleCacheReference((; Case..., Position), Cache, Diffusion, Viscosity)
            @test Cache.ReferencePosition == Position
            @test Case.Particles.Position == OriginalPosition
        end
    end
end

@testset "particle cache cutoff, motion, and invalidation" begin
    for D in (2, 3), T in (Float32, Float64)
        @testset "$D dimensions, $T" begin
            Kernel = SPHKernelInstance{D,T}(WendlandC2(); h=T(0.5))
            Point(X) = SVector{D,T}(ntuple(d -> d == 1 ? T(X) : zero(T), D))
            Position = [Point(X) for X in (0, prevfloat(one(T)), one(T), nextfloat(one(T)),
                                           prevfloat(T(1.125)), T(1.125), nextfloat(T(1.125)), 3)]
            Ranges = [1, 1, length(Position) + 1, length(Position) + 1]
            Indices = fill(2, length(Position))
            Lists = [Int[], [1, 3], Int[]]
            Cache = SPHExample.SPHCellList.ParticleNeighborCache(Position)
            Prepare!() = SPHExample.SPHCellList.PrepareParticleNeighborCache!(Cache, Position, Kernel, Ranges, Indices, Lists)
            Prepare!()
            @test Int.(ParticleCacheNeighbors(Cache, 1)) == [2, 3, 4, 5, 6]
            @test Cache.ReferencePosition !== Position
            ReferencePosition = copy(Cache.ReferencePosition)
            Position[1] = Point(prevfloat(T(1 / 32)))
            Prepare!()
            @test Cache.ReferencePosition == ReferencePosition
            Position[1] = Point(nextfloat(T(1 / 32)))
            Prepare!()
            @test Cache.ReferencePosition == Position

            # Rebuilding the cell map must invalidate even an unmoved system.
            Lists[2] = [2, 2, 1, 3]
            SPHExample.SPHCellList.InvalidateParticleNeighborCache!(Cache)
            @test !Cache.Valid
            Prepare!()
            @test count(==(1), ParticleCacheNeighbors(Cache, 1)) == 2
            @test count(==(2), ParticleCacheNeighbors(Cache, 1)) == 3
            LargerKernel = SPHKernelInstance{D,T}(WendlandC2(); h=T(2))
            SPHExample.SPHCellList.PrepareParticleNeighborCache!(Cache, Position, LargerKernel, Ranges, Indices, Lists)
            @test 8 in ParticleCacheNeighbors(Cache, 1)
            @test Cache.CutoffSquared == LargerKernel.H²

            # The same cache supports a changed particle count and no particles.
            resize!(Position, 2)
            resize!(Indices, 2)
            Ranges .= [1, 1, 3, 3]
            Prepare!()
            @test length(Cache.ReferencePosition) == 2
            @test length(Cache.Offsets) == 3
            @test all(Index -> Index <= 2, Cache.Neighbors)
            empty!(Position)
            empty!(Indices)
            Ranges .= 1
            Prepare!()
            @test isempty(Cache.Neighbors)
            @test isempty(Cache.ReferencePosition)
        end
    end
end

@testset "particle cache preserves custom neighbor order" begin
    Case = InteractionReferenceCase(Val(2), Float64, WendlandC2(), NoShifting, NoKernelOutput)
    # The sentinel and cell 4 are empty; the other cells deliberately have
    # repeated and reversed neighbors, including a repeated owning cell.
    ParticleRanges = [1, 1, 3, 6, 6, 9, 13]
    CellListIndices = [2, 2, 3, 3, 3, 5, 5, 5, 6, 6, 6, 6]
    Lists = [Int[], [6, 5, 3, 3, 2, 1, 4], [5, 2, 6, 2], Int[], [6, 3, 2], [5, 3, 2]]
    Offsets = cumsum([1; length.(Lists)])
    Packed = PackedNeighborCellLists(Offsets, UInt8.(reduce(vcat, Lists)))
    Position = [SVector(i / 1000, isodd(i) ? 0.002 : -0.002) for i in eachindex(Case.Position)]
    for NeighborCellLists in (Lists, Packed)
        CustomCase = (; Case..., Position, ParticleRanges, CellListIndices, NeighborCellLists)
        Cache = SPHExample.SPHCellList.ParticleNeighborCache(Position)
        CheckParticleCacheReference(CustomCase, Cache, ZeroDensityDiffusion(), ZeroViscosity())
        for i in eachindex(Position)
            Cell = CellListIndices[i]
            Expected = [j for j in ParticleRanges[Cell]:(ParticleRanges[Cell + 1] - 1) if j != i]
            for Neighbor in Lists[Cell]
                append!(Expected, ParticleRanges[Neighbor]:(ParticleRanges[Neighbor + 1] - 1))
            end
            @test Int.(ParticleCacheNeighbors(Cache, i)) == Expected
        end
    end
end

@testset "particle cache reevaluates entering and nonfinite neighbors" begin
    Case = InteractionReferenceCase(Val(2), Float64, WendlandC2(), NoShifting, NoKernelOutput)
    Kernel = SPHKernelInstance{2,Float64}(WendlandC2(); h=0.5)
    ParticleRanges = [1, 1, length(Case.Position) + 1]
    CellListIndices = fill(2, length(Case.Position))
    NeighborCellLists = [Int[], [1]]
    Position = [SVector(4.0i, 0.0) for i in eachindex(Case.Position)]
    Position[1] = SVector(0.0, 0.0)
    Position[2] = SVector(1.025, 0.0)
    fill!(Case.Particles.Pressure, 10.0)
    Case = (; Case..., Kernel, Position, ParticleRanges, CellListIndices, NeighborCellLists)
    Cache = SPHExample.SPHCellList.ParticleNeighborCache(Position)
    Initial = CheckParticleCacheReference(Case, Cache, ZeroDensityDiffusion(), ZeroViscosity())
    @test iszero(Initial.Acceleration[1])
    ReferencePosition = copy(Cache.ReferencePosition)
    Position[1] += SVector(0.02, 0.0)
    Position[2] -= SVector(0.02, 0.0)
    Entered = CheckParticleCacheReference(Case, Cache, ZeroDensityDiffusion(), ZeroViscosity())
    @test !iszero(Entered.Acceleration[1])
    @test Cache.ReferencePosition == ReferencePosition
    @test 3 ∉ ParticleCacheNeighbors(Cache, 1)
    Position[3] = SVector(0.5, 0.0)
    CheckParticleCacheReference(Case, Cache, ZeroDensityDiffusion(), ZeroViscosity())
    @test 3 in ParticleCacheNeighbors(Cache, 1)
    @test Cache.ReferencePosition == Position
    for Nonfinite in (NaN, Inf, -Inf)
        Position[3] = SVector(Nonfinite, 0.0)
        Result = CheckParticleCacheReference(Case, Cache, ZeroDensityDiffusion(), ZeroViscosity())
        @test all(isfinite, Result.DensityRate)
        @test all(Value -> all(isfinite, Value), Result.Acceleration)
        Position[3] = SVector(0.5, 0.0)
        CheckParticleCacheReference(Case, Cache, ZeroDensityDiffusion(), ZeroViscosity())
        @test Cache.ReferencePosition == Position
    end
end

@testset "optional interaction outputs use the original traversal" begin
    for (Shifting, KernelOutput) in ((NoShifting, StoreKernelOutput),
                                     (PlanarShifting, NoKernelOutput),
                                     (PlanarShifting, StoreKernelOutput))
        Case = InteractionReferenceCase(Val(2), Float64, WendlandC2(), Shifting, KernelOutput; Midpoint=true)
        Cache = SPHExample.SPHCellList.MakeParticleNeighborCache(Case.MetaData, Case.Position)
        @test Cache === nothing
        @test SPHExample.SPHCellList.InvalidateParticleNeighborCache!(Cache) === nothing
        CheckParticleCacheReference(Case, Cache, ZeroDensityDiffusion(), ZeroViscosity(), Shifting, KernelOutput)
    end
end
