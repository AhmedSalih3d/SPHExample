module PreProcess

export LoadParticlesFromCSV

using CSV
using DataFrames
using StaticArrays

function LoadParticlesFromCSV(fluid_csv,boundary_csv)
    DF_FLUID = CSV.read(fluid_csv, DataFrame)
    DF_BOUND = CSV.read(boundary_csv, DataFrame)

    P1F = DF_FLUID[!,"Points:0"]
    P2F = DF_FLUID[!,"Points:1"]
    P3F = DF_FLUID[!,"Points:2"]
    P1B = DF_BOUND[!,"Points:0"]
    P2B = DF_BOUND[!,"Points:1"]
    P3B = DF_BOUND[!,"Points:2"]

    points = Vector{SVector{3,Float64}}()

    for i = 1:length(P1F)
        push!(points,SVector(P1F[i],P3F[i],P2F[i]))
    end

    for i = 1:length(P1B)
        push!(points,SVector(P1B[i],P3B[i],P2B[i]))
    end

    return points,DF_FLUID,DF_BOUND
end

end