using StaticArrays
using Chairmarks
using BenchmarkTools
using Polyester
using Base.Threads
using Test


function BatchExtractCellsFloorOriginal!(Cells, Points, CutOff)
    @batch per=thread for i ∈ eachindex(Cells)
        Cells[i] = CartesianIndex(@. floor(Int, 4 + Points[i] / CutOff)...)
    end
end

function BatchExtractCellsFloorVal!(Cells, Points, ::Val{CutOff}) where CutOff
    @batch per=thread for i ∈ eachindex(Cells)
        Cells[i] = CartesianIndex(@. floor(Int, 4 + Points[i] / CutOff)...)
    end
end

# CutOff fixed to 0.5
function BatchExtractCellsFloorFixed!(Cells, Points, _)
    @batch per=thread for i ∈ eachindex(Cells)
        Cells[i]   = CartesianIndex(@. floor(Int, 4 + Points[i] / 0.5)...)
    end
end

# Number of points/cells to test
let
    N         = 100_000
    Dims      = 2 
    FloatType = Float64
    CutOff    = 0.5

    # Initialize Points with random SVector values
    Points = [SVector{Dims, FloatType}(rand(Dims)...) for _ = 1:N]

    # Initialize Cells as an array of CartesianIndex{3}, dummy initialization
    Cells = [zero(CartesianIndex{Dims}) for _ = 1:N]

    benchmark_result = @b BatchExtractCellsFloorOriginal!($Cells, $Points, $CutOff)
    println("BatchExtractCellsFloorOriginal!"); display(benchmark_result)

    benchmark_result = @b BatchExtractCellsFloorVal!($Cells, $Points, $(Val(CutOff)))
    println("BatchExtractCellsFloorVal!"); display(benchmark_result)

    benchmark_result = @b BatchExtractCellsFloorFixed!($Cells, $Points, $CutOff)
    println("BatchExtractCellsFloorFixed!"); display(benchmark_result)

    CellsA = [zero(CartesianIndex{Dims}) for _ = 1:N]
    CellsB = [zero(CartesianIndex{Dims}) for _ = 1:N]
    CellsC = [zero(CartesianIndex{Dims}) for _ = 1:N]

    BatchExtractCellsFloorOriginal!(CellsA, Points, CutOff)
    BatchExtractCellsFloorVal!(CellsB, Points, Val(CutOff))
    BatchExtractCellsFloorFixed!(CellsC, Points, CutOff)

    @test CellsA == CellsB == CellsC
end