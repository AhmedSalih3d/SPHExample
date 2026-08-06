using Test
using SPHExample
using StaticArrays
using StructArrays
using LinearAlgebra: norm

@testset "time stepping" begin
    pos = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 0.0)]
    vel = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(0.0, 0.0)]
    acc = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(0.0, -9.81)]
    sc = SimulationConstants{Float64}()
    ker = SPHKernelInstance{2, Float64}(WendlandC2(); dx=sc.dx)
    max_acc = maximum(norm, acc)
    dt  = Δt(max_acc, sc, ker)
    @test dt > 0
    alloc = @allocated Δt(max_acc, sc, ker)
    @test alloc == 0
end

@testset "isolated particle" begin
    D = 2
    T = Float64
    sc = SimulationConstants{T}()
    ker = SPHKernelInstance{D,T}(WendlandC2(); dx=sc.dx)
    meta = SimulationMetaData{D,T}(SimulationName="iso", SaveLocation=".")

    pos = [SVector{D,T}(0, 0)]
    vel = [SVector{D,T}(0, 0)]
    acc = [SVector{D,T}(0, 0)]
    dens = [sc.ρ₀]
    press = [0.0]
    gf = [-1.0]
    limiter = [1.0]
    bound = UInt8[0]
    id = [1]
    typ = [Fluid]
    group = UInt[1]
    kernel_val = [0.0]
    kernel_grad = [SVector{D,T}(0, 0)]
    cell = [CartesianIndex(0, 0)]
    gpoint = [SVector{D,T}(0, 0)]
    gnorm = [SVector{D,T}(0, 0)]

    particles = StructArray((
        Cells=cell, Kernel=kernel_val, KernelGradient=kernel_grad,
        Position=pos, Acceleration=acc, Velocity=vel, Density=dens, Pressure=press,
        GravityFactor=gf, MotionLimiter=limiter, BoundaryBool=bound, ID=id,
        Type=typ, GroupMarker=group, GhostPoints=gpoint, GhostNormals=gnorm,
    ))

    dρdtI, vel_n, pos_n, ρ_n, ∇C, ∇r =
        AllocateSupportDataStructures(meta, particles.Position)

    for _ in 1:1000
        ResetArrays!(dρdtI, particles.Acceleration)
        dt = Δt(maximum(norm, particles.Acceleration), sc, ker)
        dt2 = dt / 2

        SPHExample.SPHCellList.HalfTimeStep(meta, sc, particles, pos_n, vel_n,
                                           ρ_n, dρdtI, dt2)
        LimitDensityAtBoundary!(ρ_n, sc.ρ₀, particles.MotionLimiter)
        Pressure!(press, ρ_n, sc)
        SPHExample.SPHCellList.FullTimeStep(meta, ker, sc, particles, vel_n, ∇C, ∇r, dt)
        DensityEpsi!(dens, dρdtI, ρ_n, dt)
        LimitDensityAtBoundary!(dens, sc.ρ₀, particles.MotionLimiter)
        SPHExample.SPHCellList.UpdateMetaData!(meta, dt)

        @test isapprox(dens[1], sc.ρ₀; atol=1e-10)
        @test isapprox(press[1], 0; atol=1e-10)
    end

    @test particles.Position[1][1] == 0
    @test particles.Velocity[1][1] == 0
    @test particles.Velocity[1][2] < 0
end

@testset "neighbor rebuild with all unique cells" begin
    D = 2
    T = Float64
    N = 3
    particles = StructArray((
        Cells = [CartesianIndex(0, 0) for _ in 1:N],
        Position = [SVector{D,T}(0, 0), SVector{D,T}(10, 0), SVector{D,T}(20, 0)],
    ))
    ParticleRanges = zeros(Int, N + 2)
    UniqueCells = zeros(CartesianIndex{D}, N + 1)
    CellListIndices = zeros(Int, N)
    _, SortingScratchSpace = Base.Sort.make_scratch(nothing, eltype(particles), N)

    for _ in 1:10
        IndexCounter = UpdateNeighbors!(particles, one(T), SortingScratchSpace, ParticleRanges, UniqueCells, CellListIndices)
        UniqueCellsView = view(UniqueCells, 1:IndexCounter)
        NeighborCellLists = [Int[] for _ in 1:IndexCounter]

        @test IndexCounter == N + 1
        @test ParticleRanges[IndexCounter + 1] == N + 1
        @test CellListIndices == collect(2:(N + 1))
        @test allunique(UniqueCellsView[2:end])

        BuildNeighborCellLists!(NeighborCellLists, ConstructStencil(Val(D)), UniqueCellsView, ParticleRanges)
        @test length(NeighborCellLists) == IndexCounter
    end
end
