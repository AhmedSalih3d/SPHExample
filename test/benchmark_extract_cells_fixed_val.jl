using StaticArrays
using Chairmarks
using BenchmarkTools
using Polyester
using Base.Threads
using Test

function BatchExtractCellsOriginal!(Cells, Points, CutOff)
    @batch per=thread for i ∈ eachindex(Cells)
        ci = CartesianIndex(@. Int(fld(Points[i], CutOff))...)
        Cells[i] = ci + 2 * one(ci) 
    end
end

function BatchExtractCellsFloorOriginal!(Cells, Points, CutOff)
    @batch per=thread for i ∈ eachindex(Cells)
        Cells[i] = CartesianIndex(@. floor(Int, 2 + Points[i] / CutOff)...)
    end
end

function BatchExtractCellsFloorVal!(Cells, Points, ::Val{CutOff}) where CutOff
    @batch per=thread for i ∈ eachindex(Cells)
        Cells[i] = CartesianIndex(@. floor(Int, 2 + Points[i] / CutOff)...)
    end
end

function BatchExtractCellsFloorMap!(Cells, Points, ::Val{CutOff}) where CutOff
    function map_floor(x)
        floor(Int, x / CutOff) + 2
    end
    
    @batch per=thread for i ∈ eachindex(Cells)
        t = map(map_floor, Tuple(Points[i]))
        Cells[i] = CartesianIndex(t)
    end
end

# CutOff fixed to 0.5
function BatchExtractCellsFloorFixed!(Cells, Points, _)
    @batch per=thread for i ∈ eachindex(Cells)
        Cells[i]   = CartesianIndex(@. floor(Int, 2 + Points[i] / 0.5)...)
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

    # Benchmarks
    benchmark_result = @b BatchExtractCellsOriginal!($Cells, $Points, $CutOff)
    println("BatchExtractCellsOriginal!"); display(benchmark_result)

    benchmark_result = @b BatchExtractCellsFloorOriginal!($Cells, $Points, $CutOff)
    println("BatchExtractCellsFloorOriginal!"); display(benchmark_result)

    benchmark_result = @b BatchExtractCellsFloorVal!($Cells, $Points, $(Val(CutOff)))
    println("BatchExtractCellsFloorVal!"); display(benchmark_result)

    benchmark_result = @b BatchExtractCellsFloorMap!($Cells, $Points, $(Val(CutOff)))
    println("BatchExtractCellsFloorMap!"); display(benchmark_result)

    benchmark_result = @b BatchExtractCellsFloorFixed!($Cells, $Points, $CutOff)
    println("BatchExtractCellsFloorFixed!"); display(benchmark_result)

    Cells0 = [zero(CartesianIndex{Dims}) for _ = 1:N]
    CellsA = [zero(CartesianIndex{Dims}) for _ = 1:N]
    CellsB = [zero(CartesianIndex{Dims}) for _ = 1:N]
    CellsC = [zero(CartesianIndex{Dims}) for _ = 1:N]
    CellsD = [zero(CartesianIndex{Dims}) for _ = 1:N]

    BatchExtractCellsOriginal!(Cells0, Points, CutOff)
    BatchExtractCellsFloorOriginal!(CellsA, Points, CutOff)
    BatchExtractCellsFloorVal!(CellsB, Points, Val(CutOff))
    BatchExtractCellsFloorMap!(CellsC, Points, Val(CutOff))
    BatchExtractCellsFloorFixed!(CellsD, Points, CutOff)

    @test Cells0 == CellsA == CellsB == CellsC == CellsD
end