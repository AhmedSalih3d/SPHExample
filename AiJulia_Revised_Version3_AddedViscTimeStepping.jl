using StaticArrays
using WriteVTK
using LinearAlgebra
using CSV, DataFrames
using Printf
using CellListMap
using FastPow
using NearestNeighbors

DF_FLUID = CSV.read("FluidPoints_Dp0.04.csv", DataFrame)
DF_BOUND = CSV.read("BoundaryPoints_Dp0.04.csv", DataFrame)

mutable struct Particle
    position::SVector{3, Float64}
    acceleration::SVector{3, Float64}
    velocity::SVector{3, Float64}
    density::Float64
    id::Int
    Visc::Float64
    GravityFactor::Float64
    MotionLimiter::Float64
    # For debugging
    W::Float64
    WG::SVector{3,Float64}
    ddt::Float64 #Density Diffusion Term

    function Particle()
        position = SVector(NaN, NaN, NaN)
        acceleration = SVector(NaN, NaN, NaN)
        velocity = SVector(NaN, NaN, NaN)
        density = NaN
        id = -1
        Visc = NaN
        GravityFactor = 0
        MotionLimiter = 0
        W = NaN
        WG = SVector(NaN, NaN, NaN)
        ddt = 0
        new(position, acceleration, velocity, density, id,Visc,GravityFactor,MotionLimiter,W,WG,ddt)
    end

    function Particle(position, acceleration, velocity, density, id, Visc,GravityFactor,MotionLimiter,W,WG,ddt)
        new(position, acceleration, velocity, density, id,Visc,GravityFactor,MotionLimiter,W,WG,ddt)
    end
end

mutable struct Collection
    particles::Vector{Particle}

    function Collection()
        new()
    end

    function Collection(particles)
        new(particles)
    end
end

Base.@kwdef mutable struct Constants
    dx::Float64 = 0.1
    dt_ini::Float64 = 0.0001
    h::Float64  = 1.5*sqrt(2*0.1^2)
    c0::Float64 = 0
    rho0::Float64 = 1000
    gamma::Float64 = 7
    α::Float64     = 0.01
    CFL::Float64   = 0.3
    g::Float64     = -9.81
    mass::Float64  = rho0*dx^2
    Cb::Float64    = (c0^2*rho0)/gamma
    aD::Float64    =  7 / (4 * pi * h^2)
end

Base.@kwdef mutable struct Simulation
    Boundary::Collection = Collection()
    Fluid::Collection    = Collection()
    Constants::Constants = Constants()
    dt::Float64          = 0;
    iter::Int64          = 0;
end



# Define the Wendland kernel function:
function WendlandKernel(q,aD)
    if q < 0 || q > 2
        return 0.0
    end

    return @fastpow aD * (1 - q / 2)^4 * (2 * q + 1)
end

# Define a function to calculate the distance between two particles:
function calcDistanceQ(particle1, particle2, h)
    # Calculate the distance between the two particles:
    d = particle1.position - particle2.position
    r = norm(d)

    # Calculate the normalized distance between the two particles:
    q = r / h

    return q,d
end

# Define a function to calculate the gradient of the Wendland kernel for a particle:
function calcGradientW(h, q, rel)
    # Skip distances outside the support of the kernel:
    if q < 0 || q > 2
        return SVector(0.0,0.0,0.0)
    end

    gradWx = 7 / (4 * pi * h^2) * 1/h * (5*(q-2)^3*q)/8 * (rel[1] / (q*h+1e-6))
    gradWy = 7 / (4 * pi * h^2) * 1/h * (5*(q-2)^3*q)/8 * (rel[2] / (q*h+1e-6))
    gradWz = 7 / (4 * pi * h^2) * 1/h * (5*(q-2)^3*q)/8 * (rel[3] / (q*h+1e-6)) 

    return SVector(gradWx,gradWy,gradWz)
end

# Define the pressure equation of state:
function pressure_eqn_of_state(density, initial_density, gamma, c0)
    # Calculate the pressure using the given equation:
    pressure = (c0^2 * initial_density / gamma) * ((density / initial_density)^gamma - 1)
    return pressure
end

#https://github.com/DualSPHysics/DualSPHysics/blob/2016fa3b63f40b6675b904048af13fe47f0e8af3/src_mphase/DSPH_v5.0_NNewtonian/source/JSphCpu.cpp
#https://forums.dual.sphysics.org/discussion/2173/ddt-fourtakas-hydrostatic-component
function Ψ(Sim,pa,pb,rel,gradW)
    drz   = pa.position[2] - pb.position[2]
    eta2  = (0.1*Sim.Constants.h)*(0.1*Sim.Constants.h)
    rr2   = dot(rel,rel)
    r0    = Sim.Constants.rho0
    DDTgz = r0*abs(Sim.Constants.g)/Sim.Constants.Cb
    rh    = 1 + DDTgz*drz
    drhop = r0 * ^(rh,1/Sim.Constants.gamma) - r0

    cbar  = Sim.Constants.c0

    DDTkh = 2*Sim.Constants.h*0.1

    rhoa  = pa.density
    rhob  = pb.density

    visc_densi = DDTkh*cbar*((rhob-rhoa)-drhop)/(rr2+eta2)

    dot3       = dot(rel,gradW)

    delta      = visc_densi*dot3*Sim.Constants.mass/rhob

    return delta
end

# Define the time step function:
function time_step(Sim,system,idxs_arr)

    parts =  [Sim.Fluid.particles;Sim.Boundary.particles]
    x     =  getfield.([Sim.Fluid.particles;Sim.Boundary.particles],:position)
    
    H  = Sim.Constants.h
    dt = Sim.dt

    list     = neighborlist!(system);
    function convert(x, list)
        out = [ Int[] for _ in x ]
        for (i,j,d) in list
            push!(out[i], j)
            push!(out[j], i)
        end
        return out
     end

     idxs_arr .= convert(parts,list)

    # Loop over all fluid particles:
    Threads.@threads for i  in eachindex(idxs_arr)
        idxs = idxs_arr[i]
        particle_update = parts[i]
        particle = deepcopy(particle_update)

        rel      = (particle.position,) .- x[idxs]
        r        = norm.(rel)
        q        = r ./ H

        Wg_ij     = calcGradientW.(H, q, rel)

        ρ_i  = particle.density
        ρ_j  = getfield.(parts,:density)[idxs]
        v_i  = particle.velocity
        v_j  = getfield.(parts,:velocity)[idxs]

        v_ij = (v_i,) .- v_j
    
        dρ_i_dt_n = ρ_i * sum(dot((Sim.Constants.mass ./ ρ_j) .* v_ij, Wg_ij))
        dv_i_dt_n = particle.acceleration + particle.GravityFactor*SVector(0,Sim.Constants.g,0)

        ρ_i_n_half = particle.density  + dρ_i_dt_n*(dt/2)
        v_i_n_half = particle.velocity + dv_i_dt_n*(dt/2)
        v_ij_n_half = (v_i_n_half,) .- v_j

        Pi_n_half  = pressure_eqn_of_state(ρ_i_n_half,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0)
        Pj_n_half  = pressure_eqn_of_state.(ρ_j,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0)

        cond = dot.(v_ij_n_half,rel)

        visc_bool = cond .< 0

        α = 0.01

        eta2    = (0.01*H)*(0.01*H)
        mu_ab   = (H*cond) ./ (dot.(rel,rel) .+ eta2)
        visc_i  = visc_bool .* (-α*Sim.Constants.c0*mu_ab) ./ ((ρ_i,) .+ ρ_j)

        dv_i_dt_n_half = sum(-(((Sim.Constants.mass)*((Pi_n_half .+ Pj_n_half) ./ ((ρ_i,) .*ρ_j))) .+ visc_i ).* Wg_ij) + particle.GravityFactor*SVector(0,Sim.Constants.g,0)
        dρ_i_dt_n_half = ρ_i_n_half * sum(dot((Sim.Constants.mass ./ ρ_j) .* v_ij_n_half, Wg_ij))

        epsi = -(dρ_i_dt_n_half/ρ_i_n_half) * dt

        particle_update.density  = ρ_i *((2-epsi)/(2+epsi))
        particle_update.velocity = v_i + particle.MotionLimiter*dv_i_dt_n_half*dt
        particle_update.position = particle_update.position + particle.MotionLimiter*((particle_update.velocity + v_i)/2) * dt
        particle_update.acceleration = dv_i_dt_n_half
        particle_update.WG           = sum(Wg_ij)
        particle_update.Visc         = sum(visc_i)
    end

    # for (particle,particle_update,idxs) in zip(Sim.Boundary.particles,Sim.Boundary.particles,idxs_arr)
    #     rel      = (particle.position,) .- balltree.data[idxs]
    #     r        = norm.(rel)
    #     q        = r ./ H

    #     Wg_ij     = calcGradientW.(H, q, rel)

    #     ρ_i  = particle.density
    #     ρ_j  = getfield.(parts,:density)[idxs]
    #     v_i  = particle.velocity
    #     v_j  = getfield.(parts,:velocity)[idxs]

    #     v_ij = (v_i,) .- v_j

    #     dρ_i_dt_n = ρ_i * sum(dot((Sim.Constants.mass ./ ρ_j) .* v_ij, Wg_ij))
    #     dv_i_dt_n = particle.acceleration - SVector(0,Sim.Constants.g,0)

    #     ρ_i_n_half = particle.density  + dρ_i_dt_n*(dt/2)
    #     v_i_n_half = particle.velocity + dv_i_dt_n*(dt/2)
    #     v_ij_n_half = (v_i_n_half,) .- v_j

    #     Pi_n_half  = pressure_eqn_of_state(ρ_i_n_half,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0)
    #     Pj_n_half  = pressure_eqn_of_state.(ρ_j,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0)

    #     dv_i_dt_n_half = sum(-(Sim.Constants.mass)*((Pi_n_half .+ Pj_n_half) ./ ((ρ_i,) .*ρ_j)) .* Wg_ij) - SVector(0,Sim.Constants.g,0)
    #     dρ_i_dt_n_half = ρ_i_n_half * sum(dot((Sim.Constants.mass ./ ρ_j) .* v_ij_n_half, Wg_ij))

    #     epsi = -(dρ_i_dt_n_half/ρ_i_n_half) * dt

    #     particle_update.density  = ρ_i *((2-epsi)/(2+epsi))
    #     particle_update.velocity = v_i
    #     particle_update.position = particle_update.position
    #     particle_update.acceleration = dv_i_dt_n_half
    #     particle_update.WG           = sum(Wg_ij)
    # end


#    if bool
#     density_dt_arr_n_half = zeros(length(parts))
#     density_arr_n_half    = getfield.([Sim.Fluid.particles;Sim.Boundary.particles],:density)

#     # Loop over all fluid particles:
#     for L in list
#         i = L[1]; j = L[2]; d = L[3];
#         p_i_u = parts[i]
#         p_j_u = parts[j]

#         p_i = deepcopy(p_i_u)
#         p_j = deepcopy(p_j_u)

#         # Calculate q
#         q = d/H

#         # Calculate relative distance between them
#         rel = p_i.position - p_j.position

#         # Calculate Wendland Kernel Values
#         Wi = WendlandKernel(q,Sim.Constants.aD)

#         # Calculate Gradient Values
#         Wg_ij     = calcGradientW(H, q, rel)
#         Wg_ji     = calcGradientW(H, q, -rel)

#         # Calculate Continuity Equation
#         v_ij   = p_i.velocity - p_j.velocity
#         v_ji   = -v_ij
#         ρ_i    = p_i.density
#         ρ_j    = p_j.density

#         dρ_i_dt_n = ρ_i * dot((Sim.Constants.mass/ρ_j)*v_ij, Wg_ij)
#         dρ_j_dt_n = ρ_j * dot((Sim.Constants.mass/ρ_i)*v_ji, Wg_ji)

#         # Caclulate Momentum Equation
#         Pi            = pressure_eqn_of_state(ρ_i,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0)
#         Pj            = pressure_eqn_of_state(ρ_j,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0)
#         ρ_ij          = (ρ_i+ρ_j)/2
#         visc_ij       = Π(Sim.Constants.α,Sim.Constants.c0,Sim.Constants.h,v_ij,rel,ρ_ij)*0
#         visc_ji       = Π(Sim.Constants.α,Sim.Constants.c0,Sim.Constants.h,v_ji,-rel,ρ_ij)*0 #############################
#         dv_i_dt_n     = -(Sim.Constants.mass)*((Pi+Pj)/(ρ_i*ρ_j) + visc_ij) * Wg_ij
#         dv_j_dt_n     = (Sim.Constants.mass)*((Pi+Pj)/(ρ_i*ρ_j) + visc_ji)  * Wg_ij

#         # Calculate Half-Step properties for density and momentum derivative to
#         # updated density and acceleration values
#         ρ_i_n_half     = p_i.density  + dρ_i_dt_n*(dt/2)
#         ρ_j_n_half     = p_j.density  + dρ_j_dt_n*(dt/2)

#         v_i_n_half     = p_i.velocity + dv_i_dt_n*(dt/2)
#         v_j_n_half     = p_j.velocity + dv_j_dt_n*(dt/2)
#         v_ij_n_half    = v_i_n_half - v_j_n_half

#         # Based on these new values at n+½ calculate momentum and density derivative again
#         Pi_n_half     = pressure_eqn_of_state(ρ_i_n_half,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0)
#         Pj_n_half     = pressure_eqn_of_state(ρ_j_n_half,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0)

#         dv_i_dt_n_half     = -(Sim.Constants.mass)*((Pi_n_half+Pj_n_half)/(ρ_i_n_half*ρ_j_n_half) + visc_ij) * Wg_ij 
#         dv_j_dt_n_half     = -(Sim.Constants.mass)*((Pi_n_half+Pj_n_half)/(ρ_i_n_half*ρ_j_n_half) + visc_ji) * -Wg_ij

#         dρ_i_dt_n_half     = ρ_i_n_half * dot((Sim.Constants.mass/ρ_j_n_half)*v_ij_n_half, Wg_ij)
#         dρ_j_dt_n_half     = ρ_j_n_half * dot((Sim.Constants.mass/ρ_i_n_half)*(-v_ij_n_half), -Wg_ij)

#         # Now calculate final values at n+1
#         #epsi_i = -(dρ_i_dt_n_half/ρ_i_n_half) * dt
#         #epsi_j = -(dρ_j_dt_n_half/ρ_j_n_half) * dt

#         # Density update --> has to be done outside loop this leads to sky-rocket numbers..
#         # p_i_u.density  += p_i.density*((2-epsi_i)/(2+epsi_i))
#         # p_j_u.density  += p_j.density*((2-epsi_j)/(2+epsi_j))
#         #density_fac_arr[i] += 1-((2-epsi_i)/(2+epsi_i))
#         #density_fac_arr[j] += 1-((2-epsi_j)/(2+epsi_j))

#         density_dt_arr_n_half[i] += dρ_i_dt_n_half
#         density_dt_arr_n_half[j] += dρ_j_dt_n_half
#         density_arr_n_half[i]    += dρ_i_dt_n*(dt/2)
#         density_arr_n_half[j]    += dρ_j_dt_n*(dt/2)

#         # Velocity update
#         p_i_u.velocity  += p_i.MotionLimiter*dv_i_dt_n_half*dt
#         p_j_u.velocity  += p_j.MotionLimiter*dv_j_dt_n_half*dt

#         # Position update
#         p_i_u.position  += p_i.MotionLimiter*((p_i_u.velocity+p_i.velocity)/2)*dt
#         p_j_u.position  += p_j.MotionLimiter*((p_j_u.velocity+p_j.velocity)/2)*dt

#         # For acceleration we just use the half step value as output
#         p_i_u.acceleration += dv_i_dt_n_half
#         p_j_u.acceleration += dv_j_dt_n_half

#         # Add Kernel Data #REMEMBER WE MISS W(0) HERE
#         #p_i_u.W += Wi
#         #p_j_u.W += Wi
#         # Add Kernel Gradient Data
#         p_i_u.WG += Wg_ij
#         p_j_u.WG += -Wg_ij
#         # Add Viscosity Data
#         p_i_u.Visc += visc_ij
#         p_j_u.Visc += visc_ji
#     end

#     #println(density_fac_arr)
#     epsi = -(density_dt_arr_n_half ./ density_arr_n_half) * dt
#     for (p_u,e) in zip(parts,epsi)
#         p_u.density       = p_u.density*((2-e)/(2+e))
#     end

# else
#     T = typeof(Sim.Fluid.particles[1])
#     f = fieldnames(T)

#      # Loop over all fluid particles:
#      for (particle,particle_update) in zip(Sim.Fluid.particles,Sim.Fluid.particles)
#         #
#         pold = deepcopy(particle)

#         # Initial values
#         rho_n = particle.density
#         vel_n = particle.velocity
#         pos_n = particle.position

#         #  n --> n+½
#         drhodt_n = continuity_eqn(particle, Sim, H)
#         dvadt_n  = particle.acceleration

#         rho_n_half = drhodt_n * dt/2
#         v_n_half   = dvadt_n  * dt/2

#         particle.density  += rho_n_half
#         particle.velocity += v_n_half

#         # n+½ --> re-evaluate terms
#         particle.acceleration,_ = inviscid_momentum_eqn(particle, Sim, H, dx)
#         drhodt_n_half = continuity_eqn(particle, Sim, H)

#         # n+½ --> n+1
#         epsi = -(drhodt_n_half/particle.density) * dt

#         particle.density  = rho_n *((2-epsi)/(2+epsi))

#         particle.velocity = vel_n + particle.acceleration*dt

#         particle.position = pos_n + ((particle.velocity+vel_n)/2) * dt

#         # Update
#         for f_ in f
#             setfield!(particle_update,f_,getfield(particle,f_))
#         end

#         particle        = pold
#     end

#     for (particle,particle_update) in zip(Sim.Boundary.particles,Sim.Boundary.particles)
#         #
#         pold = deepcopy(particle)
#         # Initial values
#         rho_n = particle.density
#         vel_n = particle.velocity
#         pos_n = particle.position

#         #  n --> n+½
#         drhodt_n = continuity_eqn(particle, Sim, H)
#         dvadt_n  = particle.acceleration - SVector(0,Sim.Constants.g,0)
    
#         rho_n_half = drhodt_n * dt/2
#         v_n_half   = dvadt_n  * dt/2

#         particle.density  += rho_n_half
#         particle.velocity += v_n_half

#         # n+½ --> re-evaluate terms
#         particle.acceleration,_ = inviscid_momentum_eqn(particle, Sim, H, dx)
#         particle.acceleration -= SVector(0,Sim.Constants.g,0)

#         drhodt_n_half = continuity_eqn(particle, Sim, H)

#         # n+½ --> n+1
#         epsi = -(drhodt_n_half/particle.density) * dt

#         particle.density  = rho_n *((2-epsi)/(2+epsi))

#         particle.velocity = vel_n + particle.acceleration*dt

#         particle.position = pos_n + ((particle.velocity+vel_n)/2) * dt

#         # Boundary particles, are fixed
#         particle.position = pos_n
#         particle.velocity = vel_n

#         # Update
#         for f_ in f
#             setfield!(particle_update,f_,getfield(particle,f_))
#         end
#         particle        = pold
#     end

    acc_arr = 0
#end

    # Extract Info relevant for time stepping
    # Inspired by JSphCpu.cpp from DualSPHysics
    max_acc = maximum(norm.(getfield.(Sim.Fluid.particles,:acceleration)));
    dt1     =  sqrt(H/max_acc)

    dt2     = H / (Sim.Constants.c0 + maximum(getfield.(Sim.Fluid.particles,:Visc)))

    dt      = Sim.Constants.CFL*min(dt1,dt2)

    Sim.dt  = dt;

    if isnan(dt)
        print("Simulation experienced nan time step")
        return 1
    end

    it = lpad(Sim.iter,4,"0")
    @printf "Iteration: %s | dt = %.5e" it dt
end

# Define the create_vtp_file subfunction:
function create_vtp_file(collection::Collection, filename::String)
    # Create a vector of the x, y, and z positions of the particles:
    positions = [
        [particle.position[1], particle.position[2], particle.position[3]]
        for particle in collection.particles
    ]

    # Create a vector of the particle densities:
    densities  = [particle.density for particle in collection.particles]
    ddt        = [particle.ddt for particle in collection.particles]
    accelerations = [particle.acceleration for particle in collection.particles]
    velocities = [particle.velocity for particle in collection.particles]
    kernelW   = [particle.W for particle in collection.particles]
    kernelWG  = [particle.WG for particle in collection.particles]
    viscocities = [particle.Visc for particle in collection.particles]

    # Convert the particle positions and densities into the format required by the vtk_grid function:
    points = hcat(positions...)  # Concatenate the particle positions into a single matrix
    polys = empty(MeshCell{WriteVTK.PolyData.Polys,UnitRange{Int64}}[])
    verts = empty(MeshCell{WriteVTK.PolyData.Verts,UnitRange{Int64}}[])

    # Note: the order of verts, lines, polys and strips is not important.
    # One doesn't even need to pass all of them.
    all_cells = (verts, polys)

    # Create a .vtp file with the particle positions and densities:
    vtk_grid(filename, points, all_cells..., compress = true, append = false) do vtk

        # Add the particle densities as a point data array:
        vtk_point_data(vtk, densities, "density")
        vtk_point_data(vtk, ddt, "density diffusion")
        vtk_point_data(vtk, accelerations, "acceleration")
        vtk_point_data(vtk, velocities, "velocity")
        vtk_point_data(vtk, kernelW, "kernel")
        vtk_point_data(vtk, kernelWG, "kernel_gradient")
        vtk_point_data(vtk, viscocities, "Viscosity")
        vtk_point_data(vtk,pressure_eqn_of_state.(densities,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0),"pressure")
    end
end

function RunSimulation(Sim,max_iter)

    system = InPlaceNeighborList(x=getfield.([Sim.Fluid.particles;Sim.Boundary.particles],:position), cutoff=2*Sim.Constants.h, parallel=true)

    idxs_arr = Vector{Vector{Int64}}(undef,length([Sim.Fluid.particles;Sim.Boundary.particles]))

    # Loop over all iterations:
    while Sim.iter < max_iter
        # Perform an action every 100 iterations:
        if Sim.iter % 50 == 0
            # Create .vtp files for the fluid particles and the wall particles:
            create_vtp_file(Sim.Fluid, "./particles/fluid_particles"*lpad(Sim.iter,4,"0")*".vtp")
            create_vtp_file(Sim.Boundary, "./particles/wall_particles"*lpad(Sim.iter,4,"0")*".vtp")
        end

        update!(system,getfield.([Sim.Fluid.particles;Sim.Boundary.particles],:position))
        
        # Increment the counter:
        Sim.iter += 1;
        stats = @timed time_step(Sim,system,idxs_arr)
        @printf " | Execution Time: %.5e [s] \n" stats.time
    end
end

# Run

#Sim = Simulation(dt=1e-4,h=0.141421,c0=81.675,dx=0.1,rho0=1000)
Consts = Constants(dt_ini=1e-4,h=0.056569,c0=85.89,dx=0.04,rho0=1000)
Sim = Simulation(Constants=Consts)
Sim.dt = Sim.Constants.dt_ini

#Sim = Simulation(dt=1e-4,h=0.028284,c0=87.25,dx=0.02,rho0=1000)

# Create a Collection object for the fluid particles:
fluid_particles = Collection(Vector{Particle}())

for i = 1:size(DF_FLUID)[1]
    idp = DF_FLUID[i,:]["Idp"]
    pos = SVector(0.5,0.2,0)+SVector(DF_FLUID[i,:]["Points:0"],DF_FLUID[i,:]["Points:2"],DF_FLUID[i,:]["Points:1"])
    acc = SVector(0,0,0)
    vel = SVector(0.0, 0.0, 0.0)
    # Create a new Particle object with the calculated position:
    particle = Particle(pos,acc,vel, DF_FLUID[i,:]["Rhop"], idp,0,1,1,0,SVector(0,0,0),0)

    # Add the particle to the wall_particles collection:
    push!(fluid_particles.particles, particle)
end

# Initialize the positions of the wall particles using a regular grid:
wall_particles = Collection(Vector{Particle}())

for i = 1:size(DF_BOUND)[1]
    idp = DF_BOUND[i,:]["Idp"]
    pos = SVector(DF_BOUND[i,:]["Points:0"],DF_BOUND[i,:]["Points:2"],DF_BOUND[i,:]["Points:1"])
    acc = SVector(0,0,0)
    vel = SVector(0.0, 0.0, 0.0)
    # Create a new Particle object with the calculated position:
    particle = Particle(pos,acc,vel, Sim.Constants.rho0, idp,0,-1,0,0,SVector(0,0,0),0)

    # Add the particle to the wall_particles collection:
    push!(wall_particles.particles, particle)

    # Create a new Particle object with the calculated position:
    #particle = Particle(pos-SVector(Sim.dx/2,Sim.dx/2,0.0),acc,vel, Sim.rho0, idp,0,SVector(0,0,0))

    # Add the particle to the wall_particles collection:
    #push!(wall_particles.particles, particle)
end

Sim.Boundary = wall_particles
Sim.Fluid    = fluid_particles

foreach(rm, filter(endswith(".vtp"), readdir("./particles",join=true)))
RunSimulation(Sim,10001)
#println(Sim.Fluid.particles[1])