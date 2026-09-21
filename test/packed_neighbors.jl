using Test
using SPHExample

@testset "packed neighbor IDs preserve cells and stencil order" begin
    for (Limit, IndexType) in ((0, UInt8), (255, UInt8), (256, UInt16),
                              (65535, UInt16), (65536, UInt32),
                              (Int(typemax(UInt32)) + 1, UInt64))
        Lists = PackedNeighborCellLists(Limit)
        @test eltype(Lists.Neighbors) === IndexType
        @test length(Lists) == 0
    end
    @test_throws ArgumentError PackedNeighborCellLists(-1)

    for I in (UInt8, UInt16, UInt32, UInt64)
        LargestID = min(typemax(I), UInt64(typemax(Int)))
        Lists = PackedNeighborCellLists(Int[1, 4, 4], I[1, 255, LargestID])
        @test collect(Lists[1]) == Int[1, 255, LargestID]
        @test isempty(Lists[2])
        @test_throws BoundsError Lists[1][0]
        @test_throws BoundsError Lists[1][4]

        Runs = PackedNeighborCellLists(Int[1, 3, 3], I[1, LargestID - 2], I[3, LargestID])
        @test collect(Runs[1]) == Int[1, 2, 3, LargestID - 2, LargestID - 1, LargestID]
        @test [Runs[1][Index] for Index in eachindex(Runs[1])] == collect(Runs[1])
        @test isempty(Runs[2])
        @test_throws BoundsError Runs[1][7]
    end

    for D in (2, 3), Width in (3, 7)
        Sentinel = SPHExample.SPHNeighborList.MinCell(CartesianIndex{D})
        AllCells = [Sentinel; vec(collect(CartesianIndices(ntuple(_ -> -Width:Width, D))))]
        # Exercise sentinel/empty cells, missing cells, shrinking, and regrowth.
        Packed = PackedNeighborCellLists(length(AllCells))
        Legacy = Vector{Int}[]
        Map = Dict{CartesianIndex{D},Int}()
        for Stride in (1, 3, 1)
            Cells = AllCells[1:Stride:end]
            Ranges = ones(Int, length(Cells) + 1)
            for Index in eachindex(Cells)
                Ranges[Index + 1] = Ranges[Index] + (Index == 1 || Index % 11 == 0 ? 0 : Index % 3 + 1)
            end
            BuildNeighborCellLists!(Packed, ConstructStencil(Val(D)), Cells, Ranges, Map)
            BuildNeighborCellLists!(Legacy, ConstructStencil(Val(D)), Cells, Ranges)
            @test length(Packed) == length(Legacy)
            @test length(Packed.Neighbors) == length(Packed.RunEnds)
            @test all(Packed.Neighbors .<= Packed.RunEnds)
            @test all(Index -> Packed[Index] == Legacy[Index], eachindex(Legacy))
            for Lists in (Legacy, Packed)
                @test all(eachindex(Legacy)) do Index
                    Expected = [Particle for Cell in Legacy[Index] for Particle in Ranges[Cell]:(Ranges[Cell + 1] - 1)]
                    Spans = SPHExample.SPHNeighborList.NeighborParticleRanges(Lists[Index], Ranges)
                    collect(Iterators.flatten(Spans)) == Expected
                end
            end
            @test ComputeCellNeighborCounts(Ranges, Packed, length(Cells)) ==
                  ComputeCellNeighborCounts(Ranges, Legacy, length(Cells))
            @test_throws BoundsError Packed[0]
            @test_throws BoundsError Packed[length(Packed) + 1]
        end
    end

    TooNarrow = PackedNeighborCellLists(255)
    Cells = [CartesianIndex(Index, 0) for Index in 1:256]
    @test_throws ArgumentError BuildNeighborCellLists!(TooNarrow, ConstructStencil(Val(2)), Cells, collect(1:257))
    @test length(TooNarrow) == 0

    # A non-Cartesian stencil can contain repeated and reversed offsets.
    # Preserve that order too, and never join runs across different owner cells.
    Cells = [CartesianIndex(Index, 0) for Index in 1:6]
    Ranges = collect(1:7)
    Stencil = CartesianIndex{2}[CartesianIndex(Index, 0) for Index in (2, 1, -1, -1, 0)]
    Packed = PackedNeighborCellLists(6)
    Legacy = Vector{Int}[]
    BuildNeighborCellLists!(Packed, Stencil, Cells, Ranges)
    BuildNeighborCellLists!(Legacy, Stencil, Cells, Ranges)
    @test all(Index -> collect(Packed[Index]) == Legacy[Index], eachindex(Legacy))
end

@testset "neighbor particle spans preserve arbitrary cell order" begin
    Ranges = [1, 1, 3, 6, 6, 7, 9]
    for Cells in (Int[], [1], [2, 3], [2, 4, 5], [5, 3, 2], [2, 2, 3], [2, 3, 4, 5, 6])
        Expected = [Particle for Cell in Cells for Particle in Ranges[Cell]:(Ranges[Cell + 1] - 1)]
        Packed = PackedNeighborCellLists([1, length(Cells) + 1], UInt8.(Cells))
        for Indices in (Cells, Packed[1])
            Spans = SPHExample.SPHNeighborList.NeighborParticleRanges(Indices, Ranges)
            @test collect(Iterators.flatten(Spans)) == Expected
        end
    end
    Spans = SPHExample.SPHNeighborList.NeighborParticleRanges([2, 3, 5, 6], Ranges)
    @test collect(Spans) == [1:5, 6:8]
end
