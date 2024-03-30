using StaticArrays
using Chairmarks
using BenchmarkTools
using Polyester
using Base.Threads

function BatchExtractCells!(Cells, Points, CutOff)
    @batch per=thread for i ∈ eachindex(Cells)
        ci = CartesianIndex(@. Int(fld(Points[i], CutOff))...)
        Cells[i] = ci + 2 * one(ci) 
    end
end

function BatchExtractCellsManual!(Cells, Points, CutOff)
    @batch per=thread for i in eachindex(Cells)
        p = Points[i]
        x, y = p[1], p[2]
        
        ci_x = Int(fld(x, CutOff))
        ci_y = Int(fld(y, CutOff))

        ci = CartesianIndex(ci_x, ci_y)
        Cells[i] = ci + 2 * one(ci)
    end
end



# function ThreadsExtractCells!(Cells, Points, CutOff)
#     @threads for i ∈ eachindex(Cells)
#         ci = CartesianIndex(@. Int(fld(Points[i], CutOff))...)
#         Cells[i] = ci + 2 * one(ci) 
#     end
# end

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

@benchmark BatchExtractCells!($Cells, $Points, $CutOff)

benchmark_result = @b BatchExtractCellsManual!($Cells, $Points, $CutOff)
println("BatchExtractCellsManual"); display(benchmark_result)

# benchmark_result = @b ThreadsExtractCells!($Cells, $Points, $CutOff)
# println("ThreadsExtractCells"); display(benchmark_result)
