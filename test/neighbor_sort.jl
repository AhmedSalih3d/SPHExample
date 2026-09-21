using Test
using SPHExample
using StaticArrays
using StructArrays

@testset "stable neighbor sorting preserves all particle fields" begin
    for D in (2, 3), T in (Float32, Float64), ParticleCount in (1, 73)
        Particles = StructArray((
            Cells=fill(zero(CartesianIndex{D}), ParticleCount),
            Position=[SVector{D,T}(ntuple(Axis -> T(mod(17 * Index + 7 * Axis, 23) - 11) / T(4), Val(D))) for Index in 1:ParticleCount],
            Velocity=[SVector{D,T}(ntuple(Axis -> T(Index + Axis), Val(D))) for Index in 1:ParticleCount],
            Acceleration=[SVector{D,T}(ntuple(Axis -> T(Index - Axis), Val(D))) for Index in 1:ParticleCount],
            Density=T.(1000 .+ (1:ParticleCount)),
            Pressure=T.(1:ParticleCount),
            ID=collect(1:ParticleCount),
            Type=fill(Fluid, ParticleCount),
            GroupMarker=UInt.(1:ParticleCount),
            Kernel=T.(2 .* (1:ParticleCount)),
            KernelGradient=[SVector{D,T}(ntuple(Axis -> T(3 * Index + Axis), Val(D))) for Index in 1:ParticleCount],
            GhostPoints=[SVector{D,T}(ntuple(Axis -> T(4 * Index + Axis), Val(D))) for Index in 1:ParticleCount],
            GhostNormals=[SVector{D,T}(ntuple(Axis -> T(5 * Index + Axis), Val(D))) for Index in 1:ParticleCount],
        ))
        Reference = deepcopy(Particles)
        Scratch = SPHExample.SPHNeighborList.NeighborSortScratch(ParticleCount)
        _, ReferenceScratch = Base.Sort.make_scratch(nothing, eltype(Reference), ParticleCount)
        Ranges = zeros(Int, ParticleCount + 2)
        ReferenceRanges = similar(Ranges)
        Cells = zeros(CartesianIndex{D}, ParticleCount + 1)
        ReferenceCells = similar(Cells)
        Indices = zeros(Int, ParticleCount)
        ReferenceIndices = similar(Indices)

        # Reusing the buffers tests both fixed points and cycles after motion,
        # with repeated cell keys to check stable ordering.
        for Round in 1:4
            if Round > 2
                for Index in eachindex(Particles)
                    NewPosition = SVector{D,T}(ntuple(Axis -> T(mod((13 + Round) * Particles.ID[Index] + 5 * Axis, 19) - 9) / T(3), Val(D)))
                    Particles.Position[Index] = NewPosition
                    Reference.Position[Index] = NewPosition
                end
            end
            Count = UpdateNeighbors!(Particles, T(2), Scratch, Ranges, Cells, Indices)
            ReferenceCount = UpdateNeighbors!(Reference, T(2), ReferenceScratch, ReferenceRanges, ReferenceCells, ReferenceIndices)

            @test Count == ReferenceCount
            @test collect(Particles) == collect(Reference)
            @test Ranges == ReferenceRanges
            @test Cells[1:Count] == ReferenceCells[1:Count]
            @test Indices == ReferenceIndices
        end
    end
end
