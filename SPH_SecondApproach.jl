using CellListMap
using WriteVTK
using StaticArrays
using LinearAlgebra
using CSV
using DataFrames

function Wᵢⱼ(αD,q)
    return αD*(1-q/2)^4*(2*q + 1)
end

function ∑ⱼWᵢⱼ(list,points,αD,h)
    N    = length(points)

    sumWI = zeros(N)
    sumWL = zeros(length(list))
    for (iter,L) in collect(enumerate(list))
        i = L[1]; j = L[2]; d = L[3]

        q = d/h

        W = Wᵢⱼ(αD,q)

        sumWI[i] += W
        sumWI[j] += W

        sumWL[iter] = W
    end

    return sumWI,sumWL
end

function ∇ᵢWᵢⱼ(αD,q,xᵢⱼ,h)
    # Skip distances outside the support of the kernel:
    if q < 0.0 || q > 2.0
        return SVector(0.0,0.0,0.0)
    end

    gradWx = αD * 1/h * (5*(q-2)^3*q)/8 * (xᵢⱼ[1] / (q*h+1e-6))
    gradWy = αD * 1/h * (5*(q-2)^3*q)/8 * (xᵢⱼ[2] / (q*h+1e-6))
    gradWz = αD * 1/h * (5*(q-2)^3*q)/8 * (xᵢⱼ[3] / (q*h+1e-6)) 

    return SVector(gradWx,gradWy,gradWz)
end

function ∑ⱼ∇ᵢWᵢⱼ(list,points,αD,h)
    N    = length(points)

    sumWgI = zeros(SVector{3,Float64},N)
    sumWgL = zeros(SVector{3,Float64},length(list))
    for (iter,L) in collect(enumerate(list))
        i = L[1]; j = L[2]; d = L[3]

        xᵢⱼ = points[i] - points[j]

        q = d/h

        Wg = ∇ᵢWᵢⱼ(αD,q,xᵢⱼ,h)

        sumWgI[i] +=  Wg
        sumWgI[j] += -Wg

        sumWgL[iter] = Wg
    end

    return sumWgI,sumWgL
end

function Pressure(ρ,c₀,γ,ρ₀)
    return ((c₀^2*ρ₀)/γ) * ((ρ/ρ₀)^γ - 1)
end

# Equation 2.5
function ∂ρᵢ∂t(system,points,m,ρ,v,WgL)
    list = system.nb.list
    N    = length(points)

    dρdtI = zeros(N)
    dρdtL = zeros(length(list))
    for (iter,L) in collect(enumerate(list))
        i = L[1]; j = L[2]

        ρᵢ    = ρ[i]
        ρⱼ    = ρ[j]
        vᵢⱼ   = v[i] - v[j]
        ∇ᵢWᵢⱼ = WgL[iter]

        dρdtI[i] += ρᵢ*dot((m/ρⱼ)*vᵢⱼ,∇ᵢWᵢⱼ)
        dρdtI[j] += ρⱼ*dot((m/ρᵢ)*-vᵢⱼ,-∇ᵢWᵢⱼ)

        dρdtL[iter] = ρᵢ*dot((m/ρⱼ)*vᵢⱼ,∇ᵢWᵢⱼ)
    end

    return dρdtI,dρdtL
end

# Equation 2.6
function ∂vᵢ∂t(system,points,m,ρ,WgL,c₀,γ,ρ₀)
    list = system.nb.list
    N    = length(points)

    dvdtI = fill(SVector(0.0,0.0,0.0),N)
    dvdtL = fill(SVector(0.0,0.0,0.0),length(list))
    for (iter,L) in collect(enumerate(list))
        i = L[1]; j = L[2]

        ρᵢ    = ρ[i]
        ρⱼ    = ρ[j]
        Pᵢ    = Pressure(ρᵢ,c₀,γ,ρ₀)
        Pⱼ    = Pressure(ρⱼ,c₀,γ,ρ₀)
        ∇ᵢWᵢⱼ = WgL[iter]

        Pfac  = (Pᵢ+Pⱼ)/(ρᵢ*ρⱼ)


        dvdt  = - m * Pfac *  ∇ᵢWᵢⱼ

        dvdtI[i]    +=  dvdt
        dvdtI[j]    +=  -dvdt
        
        dvdtL[iter] =   dvdt
    end

    return dvdtI,dvdtL
end

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

function create_vtp_file(filename,points,Wi,Wg,dρdt,dvdt,v)
    # Convert the particle positions and densities into the format required by the vtk_grid function:
    points = reduce(hcat,points)  # Concatenate the particle positions into a single matrix
    polys = empty(MeshCell{WriteVTK.PolyData.Polys,UnitRange{Int64}}[])
    verts = empty(MeshCell{WriteVTK.PolyData.Verts,UnitRange{Int64}}[])

    # Note: the order of verts, lines, polys and strips is not important.
    # One doesn't even need to pass all of them.
    all_cells = (verts, polys)

    # Create a .vtp file with the particle positions and densities:
    vtk_grid(filename, points, all_cells..., compress = true, append = false) do vtk

        # Add the particle densities as a point data array:
        vtk_point_data(vtk, Wi, "Wi")
        vtk_point_data(vtk, Wg, "Wg")
        vtk_point_data(vtk, dρdt, "Density Derivative")
        vtk_point_data(vtk, dvdt, "Acceleration")
        vtk_point_data(vtk, v, "Velocity")
    end
end

# ρ₀ = 1000
# dx = 0.125/2
# H  = sqrt(2)*dx
# m₀ = ρ₀*dx'dx
# mᵢ = mⱼ = m₀
# αD = (7/(4*π*H^2))
# c₀ = 100
# γ  = 7

# points   =  Iterators.product(-1.:dx:0., -1.:dx:0., 0:1:0.);
# points   = vec(collect.(points))
# points   = reinterpret.(SVector{3,Float64},points)
# points   = map(x->x[1],points)

### Play with code
DF_FLUID = CSV.read("FluidPoints_Dp0.02.csv", DataFrame)
DF_BOUND = CSV.read("BoundaryPoints_Dp0.02.csv", DataFrame)

P1F = DF_FLUID[!,"Points:0"]
P2F = DF_FLUID[!,"Points:1"]
P3F = DF_FLUID[!,"Points:2"]
P1B = DF_BOUND[!,"Points:0"]
P2B = DF_BOUND[!,"Points:1"]
P3B = DF_BOUND[!,"Points:2"]

points = SVector[]

for i = 1:length(P1F)
    push!(points,SVector(P1F[i],P3F[i],P2F[i]))
end

for i = 1:length(P1B)
    push!(points,SVector(P1B[i],P3B[i],P2B[i]))
end

GravityFactor = [Int64(-1) .+ 0*collect(1:size(DF_FLUID,1));Int64(1) .+ 0*collect(1:size(DF_BOUND,1))]
MotionLimiter = [Int64(1)  .+ 0*collect(1:size(DF_FLUID,1));Int64(0) .+ 0*collect(1:size(DF_BOUND,1))]

ρ₀ = 1000
dx = 0.02
H  = sqrt(2)*dx
m₀ = ρ₀*dx'dx
mᵢ = mⱼ = m₀
αD = (7/(4*π*H^2))
c₀ = 81#85.89
γ  = 7
g  = 9.81
dt = 1e-5

density  = Array([DF_FLUID.Rhop;DF_BOUND.Rhop])
velocity = zeros(SVector{3,Float64},length(points))
acceleration = zeros(SVector{3,Float64},length(points))



foreach(rm, filter(endswith(".vtp"), readdir("./second_approach/",join=true)))
neighborlist!(system)
for big_iter = 1:100001
    system  = InPlaceNeighborList(x=points, cutoff=2*H, parallel=false)
    update!(system,points)
    list = neighborlist!(system)

    WiI,WiL = ∑ⱼWᵢⱼ(list,points,αD,H)
    WgI,WgL = ∑ⱼ∇ᵢWᵢⱼ(list,points,αD,H)

    dρdtI,dρdtL = ∂ρᵢ∂t(system,points,m₀,density,velocity,WgL)

    dvdtI,dvdtL = ∂vᵢ∂t(system,points,m₀,density,WgL,c₀,γ,ρ₀)
    # We add gravity as a final step for the i particles, not the L ones, since we do not split the contribution, that is unphysical!
    dvdtI .= map((x,y)->x+y*SVector(0,g,0),dvdtI,GravityFactor)


    density_n_half  = density  .+ dρdtI * (dt/2)
    velocity_n_half = velocity .+ dvdtI * (dt/2) .* MotionLimiter
    points_n_half   = points   .+ velocity_n_half * (dt/2) .* MotionLimiter

    dρdtI_n_half,dρdtL_n_half = ∂ρᵢ∂t(system,points_n_half,m₀,density_n_half,velocity_n_half,WgL)

    dvdtI_n_half,dvdtL_n_half = ∂vᵢ∂t(system,points_n_half,m₀,density_n_half,WgL,c₀,γ,ρ₀)
    dvdtI_n_half .= map((x,y)->x+y*SVector(0,g,0),dvdtI_n_half,GravityFactor) 


    epsi = -( dρdtI_n_half ./ density_n_half)*dt

    density_new   = density .* (2 .- epsi)./(2 .+ epsi)
    velocity_new  = velocity .+ dvdtI_n_half * dt .* MotionLimiter
    points_new    = points .+ ((velocity_new .+ velocity)/2) * dt .* MotionLimiter

    density  = density_new
    velocity = velocity_new
    points   = points_new
    acceleration = dvdtI_n_half

    if big_iter % 50 == 0
        create_vtp_file("./second_approach/PlayAround_"*lpad(big_iter,4,"0"),points,WiI,WgI,density,acceleration,velocity)
    end
end


