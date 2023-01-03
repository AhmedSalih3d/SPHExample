module AuxillaryFunctions

# This function is never used in this project but added as a courtesy
# if you wish to convert a NeighbourList of Tuples into a format
# in which all indices realted to a particles neighbours is stored
function ListToIndex(points, list)
    out  = [ Int[] for _ in points ]
    rout = [ Float64[] for _ in points]

    for i in eachindex(out,rout)
            push!(out[i],i)
            push!(rout[i],0)
        end

        for (i,j,d) in list
            push!(out[i], j)
            push!(out[j], i)
            push!(rout[i],d)
        end
        return out,rout
end

end