using BenchmarkTools
import CellListMap
using StaticArrays
using LinearAlgebra
using Parameters
using Plots; using Measures

@with_kw struct CLL{I,T,D}
    Points::Vector{SVector{D,T}}
    MaxValidIndex::Base.RefValue{I} = Ref(0)
    CutOff::T
    CutOffSquared::T                         = CutOff^2
    Padding::I                               = 2
    HalfPad::I                               = convert(typeof(Padding),Padding//2)
    ZeroOffset::I                            = 1 #Since we start from 0 when generating cells

    ListOfInteractions::Vector{Tuple{I,I,T}} = Vector{Tuple{Int,Int,getsvecT(eltype(Points))}}(undef,length(Points)^2)
    Stencil::Vector{NTuple{D, I}}            = neighbors(getsvecD(eltype(Points))) 
    
    Cells::Vector{NTuple{D, I}}              = ExtractCells(Points,CutOff,getsvecD(eltype(Points)))
    UniqueCells::Vector{NTuple{D, I}}        = unique(Cells)
    Nmax::I                                  = maximum(reinterpret(Int,@view(Cells[:]))) + ZeroOffset
    Layout::Array{Vector{I}, D}              = GenerateM(Nmax,ZeroOffset,HalfPad,Padding,Cells,getsvecDT(eltype(Points)))
    
end
@inline getspecs(::Type{CLL{I,T,D}}) where {I,T,D} = (typeINT = I, typeFLT = T, dimensions=D)
@inline getsvecDT(::Type{SVector{d,T}}) where {d,T} = (dimensions=d, type=T)
@inline getsvecD(::Type{SVector{d,T}}) where {d,T} = d
@inline getsvecT(::Type{SVector{d,T}}) where {d,T} = T

@inline function distance_condition(p1::AbstractVector{T}, p2::AbstractVector{T}) where T <: AbstractFloat
    d2 = sum(@. (p1 - p2)^2)
    return d2
end

# https://jaantollander.com/post/searching-for-fixed-radius-near-neighbors-with-cell-lists-algorithm-in-julia-language/#definition
@inline function neighbors(d::Int)
    n_ = CartesianIndices((fill(-1:1, d)...,))
    n  = n_[1:fld(length(n_), 2)]

    n_svec = Vector{NTuple{d,Int}}(undef,length(n)) #zeros(SVector{d,eltype(d)},length(n))

    for i ∈ eachindex(n_svec)
        val       = n[i]
        n_svec[i] = (val.I)
    end

    return n_svec
end



function ExtractCells(p,R,d)
    n = length(p)
    cells = Vector{NTuple{d,Int}}(undef,n)

    for i = 1:n
        v = Int.(fld.(p[i],R))
        cells[i] = tuple(v...)
    end

    return cells
end

function GenerateM(Nmax,ZeroOffset,HalfPad,Padding,cells,PointsTD)

    Msize = tuple(repeat([Nmax+Padding],PointsTD.dimensions)...)
    M     = Array{Vector{Int}}(undef,Msize)

    #sizehint! is a genius function
    # but it actually does not improve performance anymore lol
    for i = 1:prod(size(M))
        arr  = Vector{Int}()
        #sizehint!(arr,100)
        @inbounds M[i] = arr
    end

    iter = 0

    for ind ∈ cells
        iter += 1
        @inbounds push!(M[(ind .+ ZeroOffset .+ HalfPad)...],iter)
    end

    return M
end

function CustomCLL(p,TheCLL)
    nl = 0

    @inbounds for Cind_ ∈ TheCLL.UniqueCells
            
        Cind = (Cind_ .+ 1 .+ TheCLL.HalfPad)

            # The indices in the cell are:
            @inbounds indices_in_cell = TheCLL.Layout[Cind...]

            n_idx_cells = length(indices_in_cell)
            @inbounds for ki = 1:n_idx_cells-1
                @inbounds  k_idx = indices_in_cell[ki]
                  @inbounds for kj = (ki+1):n_idx_cells
                    @inbounds  k_1up = indices_in_cell[kj]
                    @inbounds  d2 = distance_condition(p[k_idx],p[k_1up])

                    cond = d2 <= TheCLL.CutOffSquared

                    # If cond true, we use nl + 1 as new index
                    ind = ifelse(cond,nl+1,length(TheCLL.ListOfInteractions))
                    @inbounds TheCLL.ListOfInteractions[ind] = (k_idx,k_1up,sqrt(d2))
                    # Then if cond true, update nl
                    nl  = ifelse(cond,ind,nl)
                end
            end

            for Sind ∈ TheCLL.Stencil
                Sind = (Cind .+ Sind)
               @inbounds indices_in_cell_plus  = TheCLL.Layout[Sind...]

                # Here a double loop to compare indices_in_cell[k] to all possible neighbours
                @inbounds for k1 ∈ eachindex(indices_in_cell)
                    @inbounds k1_idx = indices_in_cell[k1]
                    @inbounds for k2 ∈ eachindex(indices_in_cell_plus)
                        @inbounds k2_idx = indices_in_cell_plus[k2]
                        @inbounds d2 = distance_condition(p[k1_idx],p[k2_idx])

                        cond = d2 <= TheCLL.CutOffSquared

                        # If cond true, we use nl + 1 as new index
                        ind = ifelse(cond,nl+1,length(TheCLL.ListOfInteractions))
                        @inbounds TheCLL.ListOfInteractions[ind] = (k1_idx,k2_idx,sqrt(d2))
                        # Then if cond true, update nl
                        nl  = ifelse(cond,ind,nl)
                    end
                end
            end
    end

    @inbounds TheCLL.MaxValidIndex[] = nl

    #resize!(TheCLL.ListOfInteractions,nl)
end

## Plotting CLL

function plot_cube(x, y, z, l, opacity=0.1)
    w = h = l
    x = x - l/2
    y = y - h/2
    z = z - w/2
    xp = [x, x, x, x, x+l, x+l, x+l, x+l];
    yp = [y, y+h, y, y+h, y, y, y+h, y+h];
    zp = [z, z, z+w, z+w, z+w, z, z, z+w];
    connections = [(1,2,3), (4,2,3), (4,7,8), (7,5,6), (2,4,7), (1,6,2), (2,7,6), (7,8,5), (4,8,5), (4,5,3), (1,6,3), (6,3,5)];
    mesh3d!(xp, yp, zp; connections, xlabel="x", proj_type = :persp, linecolor=RGBA(0,0,0,0), opacity=opacity)
end

function PlotCLL(TheCLL)
    I,T,D = getspecs(typeof(TheCLL))
    # define a function that returns a Plots.Shape
    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

    # To visualize cell midpoints
    p = scatter(Tuple.(TheCLL.Points), aspectrato=1, size=(500,500))
    #scatter!(getindex.(unique(TheCLL.Cells),1)*TheCLL.CutOff .+ TheCLL.CutOff,getindex.(unique(TheCLL.Cells),2)*TheCLL.CutOff .+ TheCLL.CutOff)
    #plot!(rectangle.(TheCLL.CutOff,TheCLL.CutOff,getindex.(unique(TheCLL.Cells),1)*TheCLL.CutOff,getindex.(unique(TheCLL.Cells),2)*TheCLL.CutOff), legend=false,opacity=.5, color=:blue, aspectrato=1)

    if D == 2
        plot!(rectangle.(TheCLL.CutOff,TheCLL.CutOff,getindex.(unique(TheCLL.Cells),1)*TheCLL.CutOff,getindex.(unique(TheCLL.Cells),2)*TheCLL.CutOff), legend=false,opacity=.5, color=:blue, aspectrato=1)
    elseif D == 3
        map(x->plot_cube(x[1],x[2],x[3],TheCLL.CutOff),TheCLL.Points)
    end

    return p
end




## Plotting CellListMap
function PlotCLM(Points,TheCLM)
    # define a function that returns a Plots.Shape
    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

    cellpoints = map(x->x.center,TheCLM.cl.cells)

    # To visualize cell midpoints

    p = CellListMap.draw_computing_cell(Points,TheCLM.box)

    #p      = scatter(getindex.(Points,1),getindex.(Points,2), aspectrato=1, size=(500,500))
    #scatter!(getindex.(cellpoints,1),getindex.(cellpoints,2))
    #scatter!(getindex.(unique(TheCLL.Cells),1)*TheCLL.CutOff .+ TheCLL.CutOff,getindex.(unique(TheCLL.Cells),2)*TheCLL.CutOff .+ TheCLL.CutOff)
    #plot!(rectangle.(TheCLL.CutOff,TheCLL.CutOff,getindex.(unique(TheCLL.Cells),1)*TheCLL.CutOff,getindex.(unique(TheCLL.Cells),2)*TheCLL.CutOff), legend=false,opacity=.5, color=:blue, aspectrato=1)
    return p
end

function BenchmarkIteration(;n=20,d=2,r=0.1,T=Float64)
    R = 2*r
    p = rand(SVector{d,T},n)

    S=CellListMap.InPlaceNeighborList(x=p, cutoff=R, parallel=false)
    CLMB = @benchmark CellListMap.neighborlist!($S)

    TheCLL = CLL(Points=p,CutOff=R)
    CLLB = @benchmark CustomCLL($p,$TheCLL)
    

    println("CellListMap")
    display(CLMB)
    println("CLL")
    display(CLLB)

    return nothing
end


function SingleIteration(;n=20,d=2,r=0.1,T=Float64)
    R = 2*r
    p = rand(SVector{d,T},n)

    S=CellListMap.InPlaceNeighborList(x=p, cutoff=R, parallel=false)
    CellListMap.neighborlist!(S)

    TheCLL = CLL(Points=p,CutOff=R)
    CustomCLL(p,TheCLL)

    return S,TheCLL
end

function SingleIterationCLL(;n=20,d=2,r=0.1,T=Float64)
    R = 2*r
    p = rand(SVector{d,T},n)

    TheCLL = CLL(Points=p,CutOff=R)
    CustomCLL(p,TheCLL)

    return TheCLL
end
println("If the solution is correct then the sum of all values (ijd) should be very close to equal")

function PlotTest(;n=10,d=2,NSIM = 10)
    CLMResults  = Vector{CellListMap.InPlaceNeighborList}(undef,NSIM)
    CLLResults  = Vector{CLL}(undef,NSIM)
    VerifyArray = Vector{Bool}(undef,NSIM)
    for i = 1:NSIM
        S,TheCLL    = SingleIteration(n=n,d=d);

        CLMsum      = sum(sum.(S.nb.list))
        CLLsum      = sum(sum.(TheCLL.ListOfInteractions[1:TheCLL.MaxValidIndex[]]))


        CLMResults[i]  = S
        CLLResults[i]  = TheCLL

        VerifyArray[i] = isapprox(CLMsum,CLLsum; rtol=1e-3)
    end

    for i ∈ eachindex(VerifyArray)
        p1 = PlotCLL(CLLResults[i])
        p2 = PlotCLM(CLLResults[i].Points,CLMResults[i])
        pt = plot(p1,p2, layout=(1,2), legend=false,size=(1000,500),aspect_ratio=1,margin=10mm)
        title!(pt,"Result. Iteration: $i | Success? : $(VerifyArray[i])")
        ylabel!("Y Axis")
        xlabel!("X Axis")
        display(pt)
    end

    
    if all(VerifyArray)
        println("Success! All comparisons matched.")
    else
        println("Fail! Not all comparisons match.")
        false_matches = findall(VerifyArray .== false);
        println("Failures at: $(reduce(hcat,false_matches))")
        

        for f in false_matches

            println("|===============================================================================|")
            println("ITERATION $f")
            #println("CLM Results:",CLMResults[f].nb.list)
            #println("CLL Results:",CLLResults[f].ListOfInteractions[1:TheCLL.MaxValidIndex[]])
            println("|===============================================================================|")

            p1 = PlotCLL(CLLResults[f])
            p2 = PlotCLM(CLLResults[f].Points,CLMResults[f])
            pt = plot(p1,p2, layout=(1,2), legend=false,size=(1000,500),aspect_ratio=1,margin=10mm)
            title!(pt,"FAIL. Iteration: $f")
            ylabel!("Y Axis")
            xlabel!("X Axis")
            display(pt)
        end

    end

    return CLMResults,CLLResults
end

#TheCLM,TheCLL = SingleIteration(n=100,d=2);
CLMResults,CLLResults = PlotTest(n=7000,d=2);

BenchmarkIteration(n=7000)

@profview a = SingleIterationCLL(n=7000,d=2);

nothing