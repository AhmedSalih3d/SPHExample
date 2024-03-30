using StaticArrays
using Chairmarks
using BenchmarkTools
using Polyester
using Base.Threads

function BatchExtractCells!(Cells, Points, CutOff)
    @batch per=thread for i ∈ eachindex(Cells)
        Cells[i] = CartesianIndex(@. Int(fld(Points[i] + 2, CutOff))...)
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
println("BatchExtractCells"); display(benchmark_result)

# @benchmark BatchExtractCells!($Cells, $Points, $CutOff)

