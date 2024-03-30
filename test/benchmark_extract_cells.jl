using StaticArrays
using Chairmarks
using BenchmarkTools
using Polyester
using Base.Threads
using Test

function BatchExtractCells!(Cells, Points, CutOff)
    @batch per=thread for i ∈ eachindex(Cells)
        Cells[i] = CartesianIndex(@. Int(fld(Points[i] + 2, CutOff))...)
    end
end

function BatchExtractCellsFloor!(Cells, Points, CutOff)
    @batch per=thread for i ∈ eachindex(Cells)
        Cells[i]   = CartesianIndex(@. floor(Int, 4 + Points[i] / CutOff)...)
    end
end

function BatchExtractCells3!(Cells, Points, CutOff)
    # @batch per=thread
    for i ∈ eachindex(Cells)
        t = map(Tuple(Points[i])) do x
            floor(Int, x/CutOff)+2
        end
        Cells[i] = CartesianIndex(t)
    end
end


# Number of points/cells to test
N         = 100_000
Dims      = 2 
FloatType = Float64
CutOff    = 0.5

# Initialize Points with random SVector values
Points = [SVector{Dims, FloatType}(rand(Dims)...) for _ = 1:N]

# Initialize Cells as an array of CartesianIndex{3}, dummy initialization
Cells = [zero(CartesianIndex{Dims}) for _ = 1:N]

benchmark_result = @b BatchExtractCells!($Cells, $Points, $CutOff)
println("BatchExtractCells!"); display(benchmark_result)

benchmark_result = @b BatchExtractCellsFloor!($Cells, $Points, $CutOff)
println("BatchExtractCellsFloor!"); display(benchmark_result)

BatchExtractCells3!(Cells, Points, CutOff)
benchmark_result = @b BatchExtractCells3!($Cells, $Points, $CutOff)
println("BatchExtractCells3!"); display(benchmark_result)

# Initialize Cells as an array of CartesianIndex{3}, dummy initialization
CellsA = [zero(CartesianIndex{Dims}) for _ = 1:N]
CellsB = [zero(CartesianIndex{Dims}) for _ = 1:N]
CellsC = [zero(CartesianIndex{Dims}) for _ = 1:N]

BatchExtractCells!(CellsA, Points, CutOff)
BatchExtractCellsFloor!(CellsB, Points, CutOff)
BatchExtractCellsFloor!(CellsC, Points, CutOff)

@test CellsA == CellsB == CellsC


