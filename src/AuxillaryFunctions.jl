module AuxillaryFunctions

function ListToIndex(points, list)
    out  = [ Int[] for _ in points ]
    rout = [ Float64[] for _ in points]

    # HERE YOU ADD THE ORIGINAL INDEX
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