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

function BatchExtractCellsMuladd!(Cells, Points, ::Val{InverseCutOff}) where InverseCutOff
    function map_floor(x)
        unsafe_trunc(Int, muladd(x,InverseCutOff,2))
    end

    @batch per=thread for i ∈ eachindex(Cells)
        t = map(map_floor, Tuple(Points[i]))
        Cells[i] = CartesianIndex(t)
    end
end

# Number of points/cells to test
let
    N         = 100_000
    Dims      = 2 
    FloatType = Float64
    CutOff    = 0.5
    InverseCutOff = 1/CutOff
    ValInverseCutOff = Val(InverseCutOff)

    # Initialize Points with random SVector values
    Points = [SVector{Dims, FloatType}(rand(Dims)...) for _ = 1:N]

    # Initialize Cells as an array of CartesianIndex{3}, dummy initialization
    Cells = [zero(CartesianIndex{Dims}) for _ = 1:N]

    # Benchmarks
    benchmark_result = @benchmark BatchExtractCellsOriginal!($Cells, $Points, $CutOff)
    println("BatchExtractCellsOriginal!"); display(benchmark_result)
    
    benchmark_result = @benchmark BatchExtractCellsMuladd!($Cells, $Points, $ValInverseCutOff)
    println("BatchExtractCellsMuladd!"); display(benchmark_result)


    Cells0 = [zero(CartesianIndex{Dims}) for _ = 1:N]
    CellsA = [zero(CartesianIndex{Dims}) for _ = 1:N]

    BatchExtractCellsOriginal!(Cells0, Points, CutOff)
    BatchExtractCellsMuladd!(CellsA, Points, ValInverseCutOff)

    @test Cells0 == CellsA
end
