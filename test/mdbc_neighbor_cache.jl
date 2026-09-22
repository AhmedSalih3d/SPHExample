using Test
using SPHExample
using StaticArrays
using StructArrays
using LinearAlgebra: norm

@testset "fixed MDBC ghost cell cache preserves traversal" begin
    SPH = SPHExample.SPHCellList
    for D in (2, 3), T in (Float32, Float64), Model in (WendlandC2(), CubicSpline{T}())
        Constants = SimulationConstants{T}(dx=T(0.02), m₀=T(1000) * T(0.02)^D)
        Kernel = SPHKernelInstance{D,T}(Model; dx=Constants.dx)
        MetaData = SimulationMetaData{D,T,NoShifting,NoKernelOutput,SimpleMDBC}(
            SimulationName="mdbc_cache", SaveLocation=".", OutputTimes=T(0.01))
        N = 240
        Position = [SVector{D,T}(ntuple(d -> T(0.02) * mod(div(i - 1, 6^(d - 1)), 6), D)) for i in 1:N]
        Types = ParticleType[mod(i, 3) == 0 ? Fixed : Fluid for i in 1:N]
        Offset = SVector{D,T}(ntuple(d -> T(0.007) * d, D))
        Particles = StructArray((
            Cells=fill(zero(CartesianIndex{D}), N), Position=Position,
            Density=T[T(1000) + T(i) / T(10) for i in 1:N], Type=Types,
            ID=collect(1:N),
            GhostPoints=[Types[i] == Fixed ? Position[i] + Offset : zero(Offset) for i in 1:N],
            GhostNormals=fill(Offset, N),
        ))
        Cache = SPH.MakeMDBCNeighborCache(MetaData, Particles)
        @test Cache isa SPH.MDBCNeighborCache
        Ranges = zeros(Int, N + 2)
        Cells = zeros(CartesianIndex{D}, N + 1)
        Indices = zeros(Int, N)
        Scratch = NeighborSortScratch(N)
        CellMap = Dict{CartesianIndex{D},Int}()
        Lists = PackedNeighborCellLists(N + 1)
        b = fill(SVector{D+1,T}(ntuple(_ -> T(-1), D + 1)), N)
        A = fill(SMatrix{D+1,D+1,T}(ntuple(_ -> T(-1), (D + 1)^2)), N)
        ReferenceB, ReferenceA = copy(b), copy(A)
        for Round in 1:3
            if Round > 1
                # Force new cells and a different particle order before rebuilding.
                for i in eachindex(Particles)
                    Particles.Position[i] = SVector{D,T}(ntuple(d -> T(0.03) * mod(Particles.ID[i] * (d + Round), 13), D))
                    Particles.GhostPoints[i] = Particles.Type[i] == Fixed ? Particles.Position[i] + Offset : zero(Offset)
                end
            end
            Count = UpdateNeighbors!(Particles, Kernel.H⁻¹, Scratch, Ranges, Cells, Indices)
            MetaData.IndexCounter = Count
            CellView = view(Cells, 1:Count)
            BuildNeighborCellLists!(Lists, ConstructStencil(Val(D)), CellView, Ranges, CellMap)
            SPH.UpdateMDBCNeighborCache!(Cache, Kernel, Particles, CellMap)
            @test Cache.GhostIndices == findall(Point -> !iszero(Point), Particles.GhostPoints)
            for (GhostIndex, i) in enumerate(Cache.GhostIndices)
                CachedNeighbors = Int[]
                ReferenceNeighbors = Int[]
                for Cell in Cache.NeighborCells[GhostIndex]
                    append!(CachedNeighbors, Ranges[Cell]:(Ranges[Cell + 1] - 1))
                end
                GhostCell = SPH.f(Kernel, Particles.GhostPoints[i])
                for Offset in ConstructStencil(Val(D))
                    Cell = get(CellMap, GhostCell + Offset, 1)
                    append!(ReferenceNeighbors, Ranges[Cell]:(Ranges[Cell + 1] - 1))
                end
                @test CachedNeighbors == ReferenceNeighbors
            end
            for Evaluation in 1:2
                # Densities change between evaluations without a cell rebuild.
                Particles.Density .+= T(0.125)
                SPH.NeighborLoopMDBC!(Kernel, MetaData, Constants, Ranges, CellView, CellMap, Particles, b, A, Cache)
                SPH.NeighborLoopMDBC!(Kernel, MetaData, Constants, Ranges, CellView, CellMap, Particles, ReferenceB, ReferenceA)
                # As in interaction_reference.jl, allow final-bit rounding from
                # different compiler specializations; traversal above is exact.
                Tolerance = 8eps(T)
                @test all(isapprox.(b, ReferenceB; rtol=Tolerance, atol=Tolerance * maximum(norm, ReferenceB)))
                @test all(isapprox.(A, ReferenceA; rtol=Tolerance, atol=Tolerance * maximum(norm, ReferenceA)))
                CachedParticles, ReferenceParticles = deepcopy(Particles), deepcopy(Particles)
                SPH.ApplyMDBCBeforeHalf!(MetaData, Kernel, Constants, CachedParticles, Ranges, Cells, CellMap, Cache)
                SPH.ApplyMDBCBeforeHalf!(MetaData, Kernel, Constants, ReferenceParticles, Ranges, Cells, CellMap)
                # Reconstruction also solves the small boundary matrix, amplifying
                # final-bit differences in its inputs slightly.
                @test all(isapprox.(CachedParticles.Density, ReferenceParticles.Density;
                    rtol=32eps(T), atol=zero(T)))
            end
        end
        Ghost = first(Cache.GhostIndices)
        Particles.Type[Ghost] = Moving
        @test SPH.MakeMDBCNeighborCache(MetaData, Particles) === nothing
        Particles.Type[Ghost] = Fluid
        @test SPH.MakeMDBCNeighborCache(MetaData, Particles) === nothing
        fill!(Particles.GhostPoints, zero(Offset))
        SPH.UpdateMDBCNeighborCache!(Cache, Kernel, Particles, CellMap)
        @test isempty(Cache.GhostIndices) && isempty(Cache.NeighborCells)
        BeforeB, BeforeA = copy(b), copy(A)
        SPH.NeighborLoopMDBC!(Kernel, MetaData, Constants, Ranges, view(Cells, 1:MetaData.IndexCounter), CellMap, Particles, b, A, Cache)
        @test isequal(b, BeforeB) && isequal(A, BeforeA)
        EmptyParticles = Particles[1:0]
        SPH.UpdateMDBCNeighborCache!(Cache, Kernel, EmptyParticles, CellMap)
        @test isempty(Cache.GhostIndices)
    end
    NoBoundaryMetaData = SimulationMetaData{2,Float64}(SimulationName="no_mdbc", SaveLocation=".")
    @test SPH.MakeMDBCNeighborCache(NoBoundaryMetaData, nothing) === nothing
    @test SPH.UpdateMDBCNeighborCache!(nothing, nothing, nothing, nothing) === nothing
end
