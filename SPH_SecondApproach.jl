using CellListMap
using WriteVTK
using StaticArrays
using LinearAlgebra
using CSV
using DataFrames
using Printf

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

# function ∇ᵢWᵢⱼ(αD,q,xᵢⱼ,h)
#     # Skip distances outside the support of the kernel:
#     if q < 0.0 || q > 2.0
#         return SVector(0.0,0.0,0.0)
#     end

#     gradWx = αD * 1/h * (5*(q-2)^3*q)/8 * (xᵢⱼ[1] / (q*h+1e-6))
#     gradWy = αD * 1/h * (5*(q-2)^3*q)/8 * (xᵢⱼ[2] / (q*h+1e-6))
#     gradWz = αD * 1/h * (5*(q-2)^3*q)/8 * (xᵢⱼ[3] / (q*h+1e-6)) 

#     return SVector(gradWx,gradWy,gradWz)
# end

# This is a much faster version of ∇ᵢWᵢⱼ
function Optim∇ᵢWᵢⱼ(αD,q,xᵢⱼ,h) 
    # Skip distances outside the support of the kernel:
    if 0 < q < 2
        Fac = αD*5*(q-2)^3*q / (8h*(q*h+1e-6)) 
    else
        Fac = 0.0 # or return zero(xᵢⱼ) 
    end
    return Fac .* xᵢⱼ
end

function ∑ⱼ∇ᵢWᵢⱼ(list,points,αD,h)
    N    = length(points)

    sumWgI = zeros(SVector{3,Float64},N)
    sumWgL = zeros(SVector{3,Float64},length(list))
    for (iter,L) in collect(enumerate(list))
        i = L[1]; j = L[2]; d = L[3]

        xᵢⱼ = points[i] - points[j]

        q = d/h

        Wg = Optim∇ᵢWᵢⱼ(αD,q,xᵢⱼ,h)

        sumWgI[i] +=  Wg
        sumWgI[j] += -Wg

        sumWgL[iter] = Wg
    end

    return sumWgI,sumWgL
end

function Pressure(ρ,c₀,γ,ρ₀)
    return ((c₀^2*ρ₀)/γ) * ((ρ/ρ₀)^γ - 1)
end

function ∂Πᵢⱼ∂t(list,points,h,ρ,α,v,c₀,m₀,WgL)
    N    = length(points)

    η²    = (0.1*h)*(0.1*h)

    viscI = zeros(SVector{3,Float64},N)
    viscL = zeros(SVector{3,Float64},length(list))
    for (iter,L) in collect(enumerate(list))
        i = L[1]; j = L[2];
        
        ρᵢ    = ρ[i]
        ρⱼ    = ρ[j]
        vᵢⱼ   = v[i] - v[j]
        xᵢⱼ   = points[i] - points[j]
        ρᵢⱼ   = (ρᵢ+ρⱼ)*0.5

        cond      = dot(vᵢⱼ,xᵢⱼ)

        cond_bool = cond < 0

        μᵢⱼ = h*cond/(dot(xᵢⱼ,xᵢⱼ)+η²)
        Πᵢⱼ = cond_bool*(-α*c₀*μᵢⱼ)/ρᵢⱼ
        
        viscI[i] += -Πᵢⱼ*m₀*WgL[iter]
        viscI[j] +=  Πᵢⱼ*m₀*WgL[iter]

        viscL[iter] = -Πᵢⱼ*m₀*WgL[iter]
    end

    return viscI,viscL
end

# function ∂Ψᵢⱼ∂t(list,points,h,m₀,δᵩ,c₀,γ,g,ρ₀,ρ,WgL)
#     N    = length(points)

#     η²    = (0.1*h)*(0.1*h)

#     dpsiI = zeros(N)
#     dpsiL = zeros(length(list))
#     for (iter,L) in collect(enumerate(list))
#         i = L[1]; j = L[2]; d = L[3] #norm(xᵢⱼ)

#         xᵢⱼ   = points[i] - points[j]
#         ρᵢ    = ρ[i]
#         ρⱼ    = ρ[j]
        
#         Cb    = (c₀^2*ρ₀)/γ

#         r²    = dot(xᵢⱼ,xᵢⱼ)
#         # For particle i
#         dz    = xᵢⱼ[2]
#         Pᵢⱼᴴ  = ρ₀*g*dz
#         ρᵢⱼᴴ  = ρ₀*(((Pᵢⱼᴴ/Cb) + 1)^(1/γ) - 1)
#         ρⱼᵢᵀ  = ρⱼ-ρᵢ
#         Ψᵢⱼ   = 2*(ρⱼᵢᵀ+ρᵢⱼᴴ) * (xᵢⱼ/(r²+η²)) #Should be + ?

#         delta_i = δᵩ*h*c₀*dot(Ψᵢⱼ,WgL[i])*(m₀/ρⱼ)

#         # For particle j
#         Pⱼᵢᴴ  = ρ₀*g*(-dz) #!
#         ρⱼᵢᴴ  = ρ₀*(((Pⱼᵢᴴ/Cb)+1)^(1/γ) - 1)
#         ρᵢⱼᵀ  = -ρⱼᵢᵀ
#         Ψⱼᵢ   = 2*(ρᵢⱼᵀ+ρⱼᵢᴴ) * (-xᵢⱼ/(r²+η²)) #Should be + ?

#         delta_j = δᵩ*h*c₀*dot(Ψⱼᵢ,WgL[j])*(m₀/ρᵢ)

#         dpsiI[i]  += delta_i
#         dpsiI[j]  += delta_j

#         dpsiL[iter] = delta_i
#     end
#     return dpsiI,dpsiL
# end

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

# FOr use with ddt
function ∂ρᵢ∂tDDT(list,points,h,m₀,δᵩ,c₀,γ,g,ρ₀,ρ,v,WgL,MotionLimiter)
    N    = length(points)

    η²   = (0.1*h)*(0.1*h)

    dρdtI = zeros(N)
    dρdtL = zeros(length(list))
    for (iter,L) in collect(enumerate(list))
        i = L[1]; j = L[2]

        xᵢⱼ   = points[i] - points[j]
        ρᵢ    = ρ[i]
        ρⱼ    = ρ[j]
        vᵢⱼ   = v[i] - v[j]
        ∇ᵢWᵢⱼ = WgL[iter]


        Cb    = (c₀^2*ρ₀)/γ

        r²    = dot(xᵢⱼ,xᵢⱼ)

        # DDTgz = ρ₀*g/Cb
        # DDTkh = 2*h*δᵩ
        # # For particle i
        # drz   = xᵢⱼ[2]
        # rh    = 1 + DDTgz*drz
        # drhop = ρ₀* ^(rh,1/γ) - ρ₀
        # visc_densi = DDTkh*c₀*(-(ρⱼ-ρᵢ)-drhop)/(r²+η²) #DEVIATED WITH MINUS HERE
        # dot3  = dot(xᵢⱼ,∇ᵢWᵢⱼ)
        # delta_i = visc_densi*dot3*m₀/ρⱼ

        # # For particle j
        # drz   = -xᵢⱼ[2]
        # rh    = 1 + DDTgz*drz
        # drhop = ρ₀* ^(rh,1/γ) - ρ₀
        # visc_densi = DDTkh*c₀*(-(ρᵢ-ρⱼ)-drhop)/(r²+η²) #DEVIATED WITH MINUS HERE
        # dot3  = dot(-xᵢⱼ,-∇ᵢWᵢⱼ)
        # delta_j = visc_densi*dot3*m₀/ρᵢ

        # dρdtI[i] += (dot(m₀*vᵢⱼ,∇ᵢWᵢⱼ))+delta_i*MotionLimiter[i]
        # dρdtI[j] += (dot(m₀*-vᵢⱼ,-∇ᵢWᵢⱼ))+delta_j*MotionLimiter[j]

        # dρdtL[iter] = (dot(m₀*vᵢⱼ,∇ᵢWᵢⱼ)+delta_i)*MotionLimiter[i]
        # # For particle i
        
        # dz    = xᵢⱼ[2]
        # Pᵢⱼᴴ  = ρ₀*g*dz
        # ρᵢⱼᴴ  = ρ₀*(((Pᵢⱼᴴ/Cb) + 1)^(1/γ) - 1)
        # ρⱼᵢᵀ  = ρⱼ-ρᵢ
        # Ψᵢⱼ   = 2*(ρⱼᵢᵀ+ρᵢⱼᴴ) * (xᵢⱼ/(r²+η²)) #Should be + ?

        # delta_i = δᵩ*h*c₀*dot(Ψᵢⱼ,∇ᵢWᵢⱼ)

        # # For particle j
        # Pⱼᵢᴴ  = ρ₀*g*(-dz) #!
        # ρⱼᵢᴴ  = ρ₀*(((Pⱼᵢᴴ/Cb)+1)^(1/γ) - 1)
        # ρᵢⱼᵀ  = -ρⱼᵢᵀ
        # Ψⱼᵢ   = 2*(ρᵢⱼᵀ+ρⱼᵢᴴ) * (-xᵢⱼ/(r²+η²)) #Should be + ?

        # delta_j = δᵩ*h*c₀*dot(Ψⱼᵢ,-∇ᵢWᵢⱼ)

        # dρdtI[i] += ρᵢ*(dot(m₀*vᵢⱼ,∇ᵢWᵢⱼ))*(m₀/ρⱼ) +delta_i*(m₀/ρⱼ)
        # dρdtI[j] += ρⱼ*(dot(m₀*-vᵢⱼ,-∇ᵢWᵢⱼ))*(m₀/ρⱼ)+delta_j*(m₀/ρᵢ)

        # dρdtL[iter] = ρᵢ*(dot(m₀*vᵢⱼ,∇ᵢWᵢⱼ))*(m₀/ρⱼ) +delta_i*(m₀/ρⱼ)

        # For particle i
        volᵢ  = 1/ρᵢ
        volⱼ  = 1/ρⱼ

        Ψᵢⱼ   = 2*((volᵢ/volⱼ)-1) * (xᵢⱼ/(r²)) #Should be + ?

        delta_i = δᵩ*h*c₀*dot(Ψᵢⱼ,∇ᵢWᵢⱼ)*(m₀/ρⱼ)

        # For particle j
        Ψⱼᵢ   = 2*((volⱼ/volᵢ)-1) * (-xᵢⱼ/(r²)) #Should be + ?

        delta_j = δᵩ*h*c₀*dot(Ψⱼᵢ,-∇ᵢWᵢⱼ)*(m₀/ρᵢ)

        dρdtI[i]    += (dot(m₀*vᵢⱼ,∇ᵢWᵢⱼ))+delta_i*MotionLimiter[i]
        dρdtI[j]    += (dot(m₀*-vᵢⱼ,-∇ᵢWᵢⱼ))+delta_j*MotionLimiter[j]
        dρdtL[iter] =  (dot(m₀*vᵢⱼ,∇ᵢWᵢⱼ))+delta_i*MotionLimiter[i]
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


function Δt(α,points,v,c₀,h,CFL)
    eta2  = (0.01)h * (0.01)h
    visc  = maximum(abs.(h* dot.(v,points) ./ (dot.(points,points) .+ eta2)))
    dt1   = minimum(sqrt.(h ./ norm.(α)))
    dt2   = h / (c₀+visc)

    dt    = CFL*min(dt1,dt2)

    return dt
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

function create_vtp_file(filename,points,Wi,Wg,ρ,dvdt,v)
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
        vtk_point_data(vtk, ρ, "Density")
        vtk_point_data(vtk, dvdt, "Acceleration")
        vtk_point_data(vtk, v, "Velocity")
    end
end
### Play with code
#DF_FLUID = CSV.read("FluidPoints_Dp0.02.csv", DataFrame)
#DF_BOUND = CSV.read("BoundaryPoints_Dp0.02.csv", DataFrame)

DF_FLUID = CSV.read("./StillFluid/StillFluid_Water_Dp0.02.csv",DataFrame)
DF_BOUND = CSV.read("./StillFluid/StillFluid_Wall_Dp0.02.csv",DataFrame)

P1F = DF_FLUID[!,"Points:0"]
P2F = DF_FLUID[!,"Points:1"]
P3F = DF_FLUID[!,"Points:2"]
P1B = DF_BOUND[!,"Points:0"]
P2B = DF_BOUND[!,"Points:1"]
P3B = DF_BOUND[!,"Points:2"]

points = SVector[]

for i = 1:length(P1F)
    push!(points,SVector(P1F[i],P3F[i]+4,P2F[i]))
end

for i = 1:length(P1B)
    push!(points,SVector(P1B[i],P3B[i],P2B[i]))
end

GravityFactor = [Int64(-1) .+ 0*collect(1:size(DF_FLUID,1));Int64(1) .+ 0*collect(1:size(DF_BOUND,1))]
MotionLimiter = [Int64(1)  .+ 0*collect(1:size(DF_FLUID,1));Int64(0) .+ 0*collect(1:size(DF_BOUND,1))]


function RunSimulation(points,GravityFactor,MotionLimiter)
    foreach(rm, filter(endswith(".vtp"), readdir("./second_approach/",join=true)))

    ρ₀  = 1000
    dx  = 0.02
    H   = sqrt(2)*dx
    m₀  = ρ₀*dx*dx
    mᵢ  = mⱼ = m₀
    αD  = (7/(4*π*H^2))
    α   = 0.01
    c₀  = sqrt(9.81*2)*20#81#85.89
    γ   = 7
    g   = 9.81
    dt  = 1e-5
    δᵩ  = 0.1
    CFL = 0.1
    

    density  = Array([DF_FLUID.Rhop;DF_BOUND.Rhop])
    velocity = zeros(SVector{3,Float64},length(points))
    acceleration = zeros(SVector{3,Float64},length(points))

    create_vtp_file("./second_approach/PlayAround_"*lpad(0,4,"0"),points,density.*0,acceleration.*0,density,acceleration,velocity)

    system  = InPlaceNeighborList(x=points, cutoff=2*H, parallel=false)
    neighborlist!(system)

    for big_iter = 1:200001
        update!(system,points)
        list = neighborlist!(system)

        WiI,_   = ∑ⱼWᵢⱼ(list,points,αD,H)
        WgI,WgL = ∑ⱼ∇ᵢWᵢⱼ(list,points,αD,H)

        #dρdtI,_ = ∂ρᵢ∂t(system,points,m₀,density,velocity,WgL)
        dρdtI,_ = ∂ρᵢ∂tDDT(list,points,H,m₀,δᵩ,c₀,γ,g,ρ₀,density,velocity,WgL,MotionLimiter)
        #dρdtI  .*= MotionLimiter
     

        viscI,_ = ∂Πᵢⱼ∂t(list,points,H,density,α,velocity,c₀,m₀,WgL)
        dvdtI,_ = ∂vᵢ∂t(system,points,m₀,density,WgL,c₀,γ,ρ₀)
        # We add gravity as a final step for the i particles, not the L ones, since we do not split the contribution, that is unphysical!
        dvdtI .= map((x,y)->x+y*SVector(0,g,0),dvdtI+viscI,GravityFactor)


        density_n_half  = density  .+ dρdtI * (dt/2)
        velocity_n_half = velocity .+ dvdtI * (dt/2) .* MotionLimiter
        points_n_half   = points   .+ velocity_n_half * (dt/2) .* MotionLimiter

        #dρdtI_n_half,_ = ∂ρᵢ∂t(system,points_n_half,m₀,density_n_half,velocity_n_half,WgL)
        dρdtI_n_half,_ = ∂ρᵢ∂tDDT(list,points_n_half,H,m₀,δᵩ,c₀,γ,g,ρ₀,density_n_half,velocity_n_half,WgL,MotionLimiter)
        #dρdtI_n_half .*= MotionLimiter

        viscI_n_half,_ = ∂Πᵢⱼ∂t(list,points_n_half,H,density_n_half,α,velocity_n_half,c₀,m₀,WgL)
        dvdtI_n_half,_ = ∂vᵢ∂t(system,points_n_half,m₀,density_n_half,WgL,c₀,γ,ρ₀)
        dvdtI_n_half  .= map((x,y)->x+y*SVector(0,g,0),dvdtI_n_half+viscI_n_half,GravityFactor) 


        epsi = -( dρdtI_n_half ./ density_n_half)*dt

        density_new   = density .* (2 .- epsi)./(2 .+ epsi)
        velocity_new  = velocity .+ dvdtI_n_half * dt .* MotionLimiter
        points_new    = points .+ ((velocity_new .+ velocity)/2) * dt .* MotionLimiter

        density      = density_new
        velocity     = velocity_new
        points       = points_new
        acceleration = dvdtI_n_half

        # Automatic time stepping probably does not work in non-vicous sim
        #dt = Δt(acceleration,points,velocity,c₀,H,CFL)

        @printf "Iteration %i | dt = %.5e \n" big_iter dt
        if big_iter % 50 == 0
            create_vtp_file("./second_approach/PlayAround_"*lpad(big_iter,4,"0"),points,WiI,WgI,density,acceleration,velocity)
        end
    end
end

RunSimulation(points,GravityFactor,MotionLimiter)
