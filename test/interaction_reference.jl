using Test
using SPHExample
using StaticArrays
using StructArrays
using LinearAlgebra: norm

@testset "laminar viscosity uses density and distance product" begin
    for T in (Float32, Float64)
        Kernel = (; η²=T(1//4))
        Constants = (; m₀=T(2), ν₀=T(1//2))
        Particles = (; Density=T[4, 6])
        Displacement = SVector{2,T}(1, 0)
        VelocityDifference = SVector{2,T}(2, 3)
        Gradient = SVector{2,T}(-1//2, 0)

        ViscosityI, ViscosityJ = compute_viscosity(
            Laminar(), Kernel, Constants, Particles, Displacement,
            VelocityDifference, Gradient, T(1), 1, 2,
        )
        Expected = SVector{2,T}(-8//25, -12//25)
        @test ViscosityI ≈ Expected rtol=8eps(T)
        @test ViscosityJ == -ViscosityI
    end
end

function InteractionReferenceCase(::Val{D}, ::Type{T}, KernelModel,
                                  Shifting, KernelOutput; Midpoint=false, Copies=1, Packed=false) where {D,T}
    dx = T(0.02)
    Constants = SimulationConstants{T}(dx=dx, m₀=T(1000) * dx^D, c₀=T(30))
    Kernel = SPHKernelInstance{D,T}(KernelModel; dx=dx)
    MetaData = SimulationMetaData{D,T,Shifting,KernelOutput}(
        SimulationName="interaction_reference",
        SaveLocation=".",
        OutputTimes=T(0.02),
    )
    Coordinates = (
        (0, 0, 0), (0, 0, 0), (1.1, 0, -0.2), (-2.1, 0.3, 0.4),
        (2.1, 0, 0.2), (3.7, 0.4, 0), (-3.8, 0, 0.4), (4, 0, 0),
        (0, 3.8, 1), (0.2, 4.2, 1.5), (9, 0, 0), (30, 0, 0),
    )
    Count = length(Coordinates) * Copies
    Positions = [SVector{D,T}(ntuple(d -> dx * T(Coordinates[mod1(i, length(Coordinates))][d]) +
        (d == 1 ? T(2 * div(i - 1, length(Coordinates))) : zero(T)), D)) for i in 1:Count]
    Particles = StructArray((
        Cells=fill(CartesianIndex(ntuple(_ -> 0, D)), Count),
        Position=Positions,
        Velocity=[SVector{D,T}(ntuple(d -> T(0.03) * sin(T(i + 2d)), D)) for i in 1:Count],
        Acceleration=fill(zero(SVector{D,T}), Count),
        Density=[T(1000) + T(0.4) * cos(T(i)) for i in 1:Count],
        Pressure=zeros(T, Count),
        Type=ParticleType[mod(i, 5) == 0 ? Moving : mod(i, 4) == 0 ? Fixed : Fluid for i in 1:Count],
        ID=collect(1:Count),
        Kernel=fill(T(-9), Count),
        KernelGradient=fill(SVector{D,T}(ntuple(_ -> T(-9), D)), Count),
    ))
    Pressure!(Particles.Pressure, Particles.Density, Constants)
    ParticleRanges = zeros(Int, Count + 2)
    UniqueCells = zeros(CartesianIndex{D}, Count + 1)
    CellListIndices = zeros(Int, Count)
    NeighborCellLists = Packed ? PackedNeighborCellLists(length(UniqueCells)) : [Int[] for _ in eachindex(UniqueCells)]
    _, Scratch = Base.Sort.make_scratch(nothing, eltype(Particles), Count)
    CellCount = UpdateNeighbors!(Particles, Kernel.H⁻¹, Scratch, ParticleRanges, UniqueCells, CellListIndices)
    BuildNeighborCellLists!(NeighborCellLists, ConstructStencil(Val(D)),
                            view(UniqueCells, 1:CellCount), ParticleRanges)

    Position = Particles.Position
    Density = Particles.Density
    Velocity = Particles.Velocity
    Pressure = Particles.Pressure
    if Midpoint
        Position = [X + SVector{D,T}(ntuple(d -> T(0.0001) * sin(T(i + d)), D)) for (i, X) in enumerate(Position)]
        Density = [Rho + T(0.2) * sin(T(i)) for (i, Rho) in enumerate(Density)]
        Velocity = [T(-0.9) * V for V in Velocity]
        Pressure = similar(Density)
        Pressure!(Pressure, Density, Constants)
    end
    return (; Constants, Kernel, MetaData, Particles, ParticleRanges, CellListIndices,
            NeighborCellLists, Position, Density, Velocity, Pressure)
end

function CheckedInteractionReference(Case, Diffusion, Viscosity, Shifting, KernelOutput)
    (; Constants, Kernel, MetaData, Particles, ParticleRanges, CellListIndices,
       NeighborCellLists, Position, Density, Velocity, Pressure) = Case
    T = eltype(Density)
    VectorType = eltype(Position)
    Count = length(Position)
    DensityRate = zeros(T, Count)
    Acceleration = zeros(VectorType, Count)
    KernelValues = copy(Particles.Kernel)
    KernelGradients = copy(Particles.KernelGradient)
    ShiftC = fill(VectorType(ntuple(_ -> T(-9), length(first(Position)))), Count)
    ShiftR = fill(T(-9), Count)

    for i in eachindex(Position)
        Accumulators = (zero(T), zero(VectorType))
        if KernelOutput === StoreKernelOutput
            Accumulators = (Accumulators..., zero(T), zero(VectorType))
        end
        if Shifting === PlanarShifting
            Accumulators = (Accumulators..., zero(VectorType), zero(T))
        end
        Cell = CellListIndices[i]
        # Match the production summation order while leaving bounds checks enabled.
        Neighbors = vcat(collect(ParticleRanges[Cell]:(i - 1)),
                         collect((i + 1):(ParticleRanges[Cell + 1] - 1)))
        for NeighborCell in NeighborCellLists[Cell]
            append!(Neighbors, ParticleRanges[NeighborCell]:(ParticleRanges[NeighborCell + 1] - 1))
        end
        for j in Neighbors
            Accumulators = SPHExample.SPHCellList.ComputeInteractionsPerParticle!(
                Diffusion, Viscosity, Kernel, MetaData, Constants, Particles,
                Position, Density, Pressure, Velocity, Particles.Type,
                Accumulators..., i, j,
            )
        end
        DensityRate[i], Acceleration[i] = Accumulators[1:2]
        if KernelOutput === StoreKernelOutput
            KernelValues[i], KernelGradients[i] = Accumulators[3:4]
        end
        if Shifting === PlanarShifting
            ShiftC[i], ShiftR[i] = Accumulators[(end - 1):end]
        end
    end
    return (; DensityRate, Acceleration, KernelValues, KernelGradients, ShiftC, ShiftR)
end

@testset "threaded interactions match checked scalar traversal" begin
    Cases = (
        (2, Float64, WendlandC2(), NoShifting, NoKernelOutput, ZeroDensityDiffusion(), ZeroViscosity(), false),
        (2, Float32, CubicSpline{Float32}(), NoShifting, StoreKernelOutput, LinearDensityDiffusion(), ArtificialViscosity(), true),
        (3, Float64, WendlandC2(), PlanarShifting, NoKernelOutput, ComplexDensityDiffusion(), Laminar(), true),
        (3, Float32, CubicSpline{Float32}(), PlanarShifting, StoreKernelOutput, ZeroGravityLinearDensityDiffusion(), LaminarSPS(), false),
        (2, Float64, CubicSpline{Float64}(), PlanarShifting, StoreKernelOutput, ComplexDensityDiffusion(), ArtificialViscosity(), true),
        (2, Float32, WendlandC2(), PlanarShifting, NoKernelOutput, ZeroGravityLinearDensityDiffusion(), Laminar(), false),
        (3, Float64, CubicSpline{Float64}(), NoShifting, StoreKernelOutput, LinearDensityDiffusion(), LaminarSPS(), true),
        (3, Float32, WendlandC2(), NoShifting, NoKernelOutput, ZeroDensityDiffusion(), ZeroViscosity(), true),
    )
    for (D, T, KernelModel, Shifting, KernelOutput, Diffusion, Viscosity, Midpoint) in Cases, Copies in (1, 23)
        @testset "$D dimensions, $T, $(typeof(KernelModel)), $Shifting, $KernelOutput, $Copies clusters" begin
            Case = InteractionReferenceCase(Val(D), T, KernelModel, Shifting, KernelOutput; Midpoint, Copies, Packed=Copies > 1)
            Expected = CheckedInteractionReference(Case, Diffusion, Viscosity, Shifting, KernelOutput)
            (; Constants, Kernel, MetaData, Particles, ParticleRanges, CellListIndices,
               NeighborCellLists, Position, Density, Velocity, Pressure) = Case
            Count = length(Particles)
            DensityRate = fill(T(-9), Count)
            Acceleration = fill(SVector{D,T}(ntuple(_ -> T(-9), D)), Count)
            AccelerationMax = fill(T(-9), Count)
            ShiftC = fill(SVector{D,T}(ntuple(_ -> T(-9), D)), Count)
            ShiftR = fill(T(-9), Count)
            OriginalState = deepcopy((Particles.Position, Particles.Density, Particles.Velocity, Particles.Pressure))
            OriginalEvaluationState = deepcopy((Position, Density, Velocity, Pressure))
            SPHExample.SPHCellList.NeighborLoopPerParticle!(
                Diffusion, Viscosity, Kernel, MetaData, Constants, Particles,
                ParticleRanges, CellListIndices, NeighborCellLists, DensityRate,
                Acceleration, ShiftC, ShiftR, AccelerationMax;
                Position, Density, Pressure, Velocity,
            )
            # Scalar calls and compiled threaded loops can differ by a few ulps,
            # including with the original solver. Allow a few ulps of the field
            # scale for particles whose contributions nearly cancel;
            # keep state preservation and isolated-particle zeros exact below.
            Tolerance = 8eps(T)
            @test all(isapprox.(DensityRate, Expected.DensityRate; rtol=Tolerance, atol=Tolerance * maximum(norm, Expected.DensityRate)))
            @test all(isapprox.(Acceleration, Expected.Acceleration; rtol=Tolerance, atol=Tolerance * maximum(norm, Expected.Acceleration)))
            @test all(isapprox.(Particles.Kernel, Expected.KernelValues; rtol=Tolerance, atol=Tolerance * maximum(norm, Expected.KernelValues)))
            @test all(isapprox.(Particles.KernelGradient, Expected.KernelGradients; rtol=Tolerance, atol=Tolerance * maximum(norm, Expected.KernelGradients)))
            @test all(isapprox.(ShiftC, Expected.ShiftC; rtol=Tolerance, atol=Tolerance * maximum(norm, Expected.ShiftC)))
            @test all(isapprox.(ShiftR, Expected.ShiftR; rtol=Tolerance, atol=Tolerance * maximum(norm, Expected.ShiftR)))
            @test AccelerationMax ≈ norm.(Acceleration) rtol=8eps(T)
            @test all(isfinite, DensityRate)
            @test all(V -> all(isfinite, V), Acceleration)
            @test any(X -> !iszero(X), Acceleration)
            @test OriginalState == (Particles.Position, Particles.Density, Particles.Velocity, Particles.Pressure)
            @test OriginalEvaluationState == (Position, Density, Velocity, Pressure)
            Isolated = findfirst(==(12), Particles.ID)
            @test iszero(DensityRate[Isolated])
            @test iszero(Acceleration[Isolated])

            Accumulators = (zero(T), zero(SVector{D,T}))
            if KernelOutput === StoreKernelOutput
                Accumulators = (Accumulators..., zero(T), zero(SVector{D,T}))
            end
            if Shifting === PlanarShifting
                Accumulators = (Accumulators..., zero(SVector{D,T}), zero(T))
            end
            @test_throws BoundsError SPHExample.SPHCellList.ComputeInteractionsPerParticle!(
                Diffusion, Viscosity, Kernel, MetaData, Constants, Particles,
                Position, Density, Pressure, Velocity, Particles.Type,
                Accumulators..., 1, Count + 1,
            )
            Displacement = Position[1] - Position[2]
            Gradient = zero(SVector{D,T})
            if !(Diffusion isa ZeroDensityDiffusion)
                @test_throws BoundsError compute_density_diffusion(
                    Diffusion, Kernel, Constants, Particles, Displacement, Gradient,
                    Constants.dx^2, 0, 1, Particles.Type,
                )
            end
            if !(Viscosity isa ZeroViscosity)
                @test_throws BoundsError compute_viscosity(
                    Viscosity, Kernel, Constants, Particles, Displacement,
                    zero(SVector{D,T}), Gradient, Constants.dx^2, 0, 1,
                )
            end
        end
    end
end

@testset "pressure pair conserves equal-mass momentum" begin
    for D in (2, 3), T in (Float32, Float64)
        Case = InteractionReferenceCase(Val(D), T, WendlandC2(), NoShifting, NoKernelOutput)
        (; Constants, Kernel, MetaData, Particles) = Case
        fill!(Particles.Density, Constants.ρ₀)
        fill!(Particles.Pressure, T(10))
        fill!(Particles.Velocity, zero(SVector{D,T}))
        i = findfirst(==(1), Particles.ID)
        j = findfirst(==(3), Particles.ID)
        function PairContribution(i, j)
            return SPHExample.SPHCellList.ComputeInteractionsPerParticle!(
                ZeroDensityDiffusion(), ZeroViscosity(), Kernel, MetaData, Constants,
                Particles, Particles.Position, Particles.Density, Particles.Pressure,
                Particles.Velocity, Particles.Type, zero(T), zero(SVector{D,T}), i, j,
            )
        end
        DensityI, AccelerationI = PairContribution(i, j)
        DensityJ, AccelerationJ = PairContribution(j, i)
        @test iszero(DensityI) && iszero(DensityJ)
        @test !iszero(AccelerationI)
        @test AccelerationI == -AccelerationJ
    end
end
