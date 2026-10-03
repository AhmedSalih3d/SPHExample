using Test
using SPHExample
using StaticArrays
using StructArrays
using LinearAlgebra

@testset "laminar viscosity" begin
    T = Float64
    sc = SimulationConstants{T}(m₀=2, ν₀=0.3)
    ker = SPHKernelInstance{2,T}(WendlandC2(); dx=sc.dx)
    particles = (Density = T[1000, 1200],)
    x = SVector{2,T}(0.5, 0)
    v = SVector{2,T}(1, 2)
    grad = SVector{2,T}(-3, 0)
    d² = sum(abs2, x)
    force, reaction = compute_viscosity(Laminar(), ker, sc, particles,
                                        x, v, grad, d², 1, 2)
    factor = (4 * sc.m₀ * sc.ν₀ * dot(x, grad)) /
             ((particles.Density[1] + particles.Density[2]) * (d² + ker.η²))
    @test force ≈ factor * v
    @test reaction == -force
end

@testset "time stepping" begin
    pos = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(1.0, 0.0)]
    vel = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(0.0, 0.0)]
    acc = [SVector{2,Float64}(0.0, 0.0), SVector{2,Float64}(0.0, -9.81)]
    sc = SimulationConstants{Float64}()
    ker = SPHKernelInstance{2, Float64}(WendlandC2(); dx=sc.dx)
    dt  = Δt(pos, vel, acc, sc, ker)
    @test dt > 0
    alloc = @allocated Δt(pos, vel, acc, sc, ker)
    @test alloc == 0
end

@testset "prescribed wall velocity without translation" begin
    meta = SimulationMetaData{2,Float64}(SimulationName="motion", SaveLocation=".")
    positions = [SVector{2,Float64}(0.5, 1.0)]
    velocities = [SVector{2,Float64}(0.0, 0.0)]
    types = [Moving]
    groups = UInt[1]
    stationary = [MotionDetails{2,Float64}(
        Velocity=1.0, StartTime=0.0, Duration=1.0,
        Direction=SVector{2,Float64}(1.0, 0.0), MoveParticles=false,
    )]

    SPHExample.SPHCellList.ProgressMotion(positions, velocities, types, groups,
                                          0.01, stationary, meta)
    @test positions[1] == SVector{2,Float64}(0.5, 1.0)
    @test velocities[1] == SVector{2,Float64}(1.0, 0.0)

    translating = [MotionDetails{2,Float64}(
        Velocity=1.0, StartTime=0.0, Duration=1.0,
        Direction=SVector{2,Float64}(1.0, 0.0),
    )]
    SPHExample.SPHCellList.ProgressMotion(positions, velocities, types, groups,
                                          0.01, translating, meta)
    @test isapprox(positions[1][1], 0.51; atol=eps(Float64))
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
    chunk = [0]
    gpoint = [SVector{D,T}(0, 0)]
    gnorm = [SVector{D,T}(0, 0)]

    particles = StructArray((
        Cells=cell, ChunkID=chunk, Kernel=kernel_val, KernelGradient=kernel_grad,
        Position=pos, Acceleration=acc, Velocity=vel, Density=dens, Pressure=press,
        GravityFactor=gf, MotionLimiter=limiter, BoundaryBool=bound, ID=id,
        Type=typ, GroupMarker=group, GhostPoints=gpoint, GhostNormals=gnorm,
    ))

    dρdtI, vel_n, pos_n, ρ_n, ∇C, ∇r =
        AllocateSupportDataStructures(particles.Position)

    for _ in 1:1000
        ResetArrays!(dρdtI, particles.Acceleration)
        dt = Δt(particles.Position, particles.Velocity, particles.Acceleration,
                 sc, ker)
        dt2 = dt / 2

        SPHExample.SPHCellList.HalfTimeStep(meta, sc, particles, pos_n, vel_n,
                                           ρ_n, dρdtI, dt2)
        LimitDensityAtBoundary!(ρ_n, sc.ρ₀, particles.MotionLimiter)
        Pressure!(press, ρ_n, sc)
        SPHExample.SPHCellList.FullTimeStep(meta, ker, sc, particles, ∇C, ∇r, dt)
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