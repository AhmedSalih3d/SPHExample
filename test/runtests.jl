using Test
using SPHExample
using StaticArrays
using StructArrays

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


@testset "SimulationMetaData Float32 keyword conversion" begin
    meta = SimulationMetaData{2,Float32}(
        SimulationName="typed",
        SaveLocation=".",
        OutputEach=0.05,
        OutputTimes=[0.05, 0.1],
        CurrentTimeStep=0.001,
        TotalTime=1.0,
        SimulationTime=2.0,
        TimeSteps=[0.001, 0.002],
        Δx=0.01,
    )

    @test meta.OutputEach isa Float32
    @test meta.OutputTimes isa Vector{Float32}
    @test meta.CurrentTimeStep isa Float32
    @test meta.TotalTime isa Float32
    @test meta.SimulationTime isa Float32
    @test meta.TimeSteps isa Vector{Float32}
    @test meta.Δx isa Float32
    @test meta.OutputEach == Float32(0.05)
    @test meta.OutputTimes == Float32[0.05, 0.1]
    @test meta.CurrentTimeStep == Float32(0.001)
    @test meta.TotalTime == Float32(1.0)
    @test meta.SimulationTime == Float32(2.0)
    @test meta.TimeSteps == Float32[0.001, 0.002]
    @test meta.Δx == Float32(0.01)

    scalar_output_meta = SimulationMetaData{2,Float32}(
        SimulationName="scalar",
        SaveLocation=".",
        OutputTimes=0.2,
    )

    @test scalar_output_meta.OutputTimes isa Float32
    @test scalar_output_meta.TimeSteps isa Vector{Float32}
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
