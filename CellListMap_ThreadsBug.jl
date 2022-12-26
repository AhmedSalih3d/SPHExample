# CellListMap is not thread safe
using CellListMap
using StaticArrays

N      = 100
points = rand(SVector{3,Float64},N)
system = InPlaceNeighborList(x=points, cutoff=0.5, parallel=false)
list   = neighborlist!(system)

result_single_thread = zeros(N)
for L in list
    i = L[1]; j = L[2]; d = L[3]

    result_single_thread[i] += d
    result_single_thread[j] += d
end

result_multi_thread = zeros(N)
Threads.@threads for L in list
    i = L[1]; j = L[2]; d = L[3]

    result_multi_thread[i] += d
    result_multi_thread[j] += d
end

deviance_between_single_and_multi_thread = sum(result_single_thread .- result_multi_thread)

println("Deviance between single and multi thread")
println(deviance_between_single_and_multi_thread)
println("Sum SINGLE: $(sum(result_single_thread))")
println("Sum MULTI: $(sum(result_multi_thread))")
