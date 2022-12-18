using StaticArrays
using WriteVTK
using LinearAlgebra
using CSV, DataFrames
using Printf
using CellListMap
using FastPow

DF_FLUID = CSV.read("FluidPoints_Dp0.04.csv", DataFrame)
DF_BOUND = CSV.read("BoundaryPoints_Dp0.04.csv", DataFrame)

mutable struct Particle
    position::SVector{3, Float64}
    acceleration::SVector{3, Float64}
    velocity::SVector{3, Float64}
    density::Float64
    id::Int
    Visc::Float64
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
        W = NaN
        WG = SVector(NaN, NaN, NaN)
        ddt = 0
        new(position, acceleration, velocity, density, id,Visc,W,WG,ddt)
    end

    function Particle(position, acceleration, velocity, density, id, Visc,W,WG,ddt)
        new(position, acceleration, velocity, density, id,Visc,W,WG,ddt)
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

# Define the continuity equation for SPH:
function continuity_eqn(particle, Sim, h)
    # Initialize the time derivative of the density to zero:
    time_deriv_density = 0
    particle.ddt       = 0
    particle.WG        = SVector(0.0,0.0,0.0)

    # Loop over all fluid particles:
    for p in [Sim.Fluid.particles;Sim.Boundary.particles]

        # Calculate the normalized distance between the two particles:
        q,r = calcDistanceQ(particle, p, h)

        # Skip distances outside the support of the kernel:
        if q < 0 || q > 2
            continue
        end

        # Calculate the density and velocity of each particle:
        rhoa = particle.density
        vab  = particle.velocity - p.velocity

        # Calculate the mass of the other particle:
        mb   = Sim.Constants.mass

        # Calculate the gradient of the kernel for the two particles:
        gradW = calcGradientW(h, q, r)

        ddt_term = Ψ(Sim,particle,p,r,gradW)

        # Calculate the contribution of the other particle to the time derivative of the density:
        time_deriv_density += rhoa * dot((mb/p.density)*vab, gradW) #+ ddt_term

        particle.WG  += gradW;
        particle.ddt += ddt_term
    end
    
    return time_deriv_density
end

function Π(α,c0,h,vab,rab,rhoab)

    cond = dot(vab,rab)

    if cond < 0
        eta2 = (0.01*h)*(0.01*h)
        mu_ab = (h*dot(vab,rab))/(dot(rab,rab)+eta2)
        Piab  = (-α*c0*mu_ab)/rhoab
    else
        Piab = 0
    end

    return Piab
end

# Define the inviscid momentum equation for SPH:
function inviscid_momentum_eqn(particle, Sim, h,dx)
    # Initialize the acceleration to zero:
    g = Sim.Constants.g
    acceleration = SVector(0, g, 0)
    visc         = 0

    # Loop over all fluid particles:
    for p in [Sim.Fluid.particles;Sim.Boundary.particles]
        # Calculate the normalized distance between the two particles:
        q,r = calcDistanceQ(particle, p, h)

        # Skip distances outside the support of the kernel:
        if q < 0 || q > 2
            continue
        end

        # Calculate the density and velocity of each particle:
        rhoa = particle.density
        vab  = particle.velocity - p.velocity

        # Calculate the gradient of the kernel for the particle and the other particle:
        gradW = calcGradientW(h, q, r)

        rhob           = p.density;
        mb             = Sim.Constants.mass;
        Pb             = pressure_eqn_of_state(rhob,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0)
        rhoa           = particle.density

        rhoab          = (rhoa+rhob)/2

        visc           += Π(Sim.Constants.α,Sim.Constants.c0,Sim.Constants.h,vab,r,rhoab)

        Pa             = pressure_eqn_of_state(rhoa,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0)
        acceleration   += -mb*((Pb+Pa)/(rhob*rhoa) + visc) * gradW
    end

    return acceleration,visc
end

# Define the time step function:
function time_step(Sim)
    Sim_ = deepcopy(Sim);
    
    h  = Sim.Constants.h
    dt = Sim.dt
    dx = Sim.Constants.dx

    T = typeof(Sim_.Fluid.particles[1])
    f = fieldnames(T)

    # Loop over all fluid particles:
    for (particle,particle_update) in zip(Sim_.Fluid.particles,Sim.Fluid.particles)
        #
        pold = deepcopy(particle)

        # Start particle velocity and density
        vn = particle.velocity
        rn = particle.density

        # Calculate the acceleration of the particle using the inviscid momentum equation:
        particle.acceleration,particle.Visc = inviscid_momentum_eqn(particle, Sim_, h, dx)

        # Perform the first half of the position update:
        position_half_step = particle.velocity * dt / 2
        particle.position += position_half_step

        # Update the velocity of the particle using the acceleration:
        velocity_half_step = particle.acceleration * dt /2
        particle.velocity  += velocity_half_step

        # Calculate the time derivative of the density of the particle using the continuity equation:
        time_deriv_density = continuity_eqn(particle, Sim_, h)
        
        # Update the density of the particle using the time derivative of the density:
        density_half_step = time_deriv_density * dt/2
        particle.density += density_half_step

        # Calculate the acceleration of the particle using the inviscid momentum equation:
        particle.acceleration,particle.Visc = inviscid_momentum_eqn(particle, Sim_, h, dx)

        # Update the velocity of the particle using the acceleration:
        particle.velocity += particle.acceleration * dt - velocity_half_step

        # Perform the second half of the position update:
        particle.position += dt * ((particle.velocity+vn)/2)  - position_half_step

        # Final update of density
        epsi = -(time_deriv_density/particle.density) * dt

        particle.density = rn * ((2-epsi)/(2+epsi))

        # Update
        for f_ in f
            setfield!(particle_update,f_,getfield(particle,f_))
        end

        particle        = pold
    end    

    for (particle,particle_update) in zip(Sim_.Boundary.particles,Sim.Boundary.particles)
        #
        pold = deepcopy(particle)

        # Start particle velocity and density and position
        pn = particle.position
        vn = particle.velocity
        rn = particle.density

        # Calculate the acceleration of the particle using the inviscid momentum equation:
        particle.acceleration,particle.Visc = inviscid_momentum_eqn(particle, Sim_, h, dx) 
        particle.acceleration -= SVector(0,Sim_.Constants.g,0)

        # Perform the first half of the position update:
        position_half_step = particle.velocity * dt / 2
        particle.position += position_half_step

        # Update the velocity of the particle using the acceleration:
        velocity_half_step = particle.acceleration * dt /2
        particle.velocity  += velocity_half_step

        # Calculate the time derivative of the density of the particle using the continuity equation:
        time_deriv_density = continuity_eqn(particle, Sim_, h)
        
        # Update the density of the particle using the time derivative of the density:
        density_half_step = time_deriv_density * dt/2
        particle.density += density_half_step

        # Calculate the acceleration of the particle using the inviscid momentum equation:
        particle.acceleration,particle.Visc = inviscid_momentum_eqn(particle, Sim_, h, dx)
        particle.acceleration -= SVector(0,Sim_.Constants.g,0)

        # Update the velocity of the particle using the acceleration:
        particle.velocity += particle.acceleration * dt - velocity_half_step

        # Perform the second half of the position update:
        particle.position += dt * ((particle.velocity+vn)/2)  - position_half_step

        # Final update of density
        epsi = -(time_deriv_density/particle.density) * dt

        particle.density = rn * ((2-epsi)/(2+epsi))

        # Since boundary partilces no actual velocity is allowed to be set
        particle.velocity = vn
        particle.position = pn

        # Update
        for f_ in f
            setfield!(particle_update,f_,getfield(particle,f_))
        end
        particle        = pold
    end

    # Extract Info relevant for time stepping
    # Inspired by JSphCpu.cpp from DualSPHysics
    max_acc = maximum(norm.(getfield.(Sim.Fluid.particles,:acceleration)));
    dt1     =  sqrt(Sim.Constants.h/max_acc)

    dt2     = Sim.Constants.h / (Sim.Constants.c0 + maximum(getfield.(Sim.Fluid.particles,:Visc)))

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

function RunSimulation(Sim)
    # Define the maximum number of iterations:
    max_iter = 40000

    # Loop over all iterations:
    while Sim.iter < max_iter
        # Perform an action every 100 iterations:
        if Sim.iter % 50 == 0
            # Create .vtp files for the fluid particles and the wall particles:
            create_vtp_file(Sim.Fluid, "./particles/fluid_particles"*lpad(Sim.iter,4,"0")*".vtp")
            create_vtp_file(Sim.Boundary, "./particles/wall_particles"*lpad(Sim.iter,4,"0")*".vtp")
        end
        
        # Increment the counter:
        Sim.iter += 1;
        stats = @timed time_step(Sim)
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
    acc = SVector(0.0, 0.0, 0.0)
    vel = SVector(0.0, 0.0, 0.0)
    # Create a new Particle object with the calculated position:
    particle = Particle(pos,acc,vel, DF_FLUID[i,:]["Rhop"], idp,0,0,SVector(0,0,0),0)

    # Add the particle to the wall_particles collection:
    push!(fluid_particles.particles, particle)
end

# Initialize the positions of the wall particles using a regular grid:
wall_particles = Collection(Vector{Particle}())

for i = 1:size(DF_BOUND)[1]
    idp = DF_BOUND[i,:]["Idp"]
    pos = SVector(DF_BOUND[i,:]["Points:0"],DF_BOUND[i,:]["Points:2"],DF_BOUND[i,:]["Points:1"])
    acc = SVector(0.0, 0.0, 0.0)
    vel = SVector(0.0, 0.0, 0.0)
    # Create a new Particle object with the calculated position:
    particle = Particle(pos,acc,vel, Sim.Constants.rho0, idp,0,0,SVector(0,0,0),0)

    # Add the particle to the wall_particles collection:
    push!(wall_particles.particles, particle)

    # Create a new Particle object with the calculated position:
    #particle = Particle(pos-SVector(Sim.dx/2,Sim.dx/2,0.0),acc,vel, Sim.rho0, idp,0,SVector(0,0,0))

    # Add the particle to the wall_particles collection:
    #push!(wall_particles.particles, particle)
end

Sim.Boundary = wall_particles
Sim.Fluid    = fluid_particles

#foreach(rm, filter(endswith(".vtp"), readdir("./particles",join=true)))
#RunSimulation(Sim)


###

function calc_loop!(x,y,i,j,d2,H,particles,particles_out,Sim,q_arr)
    # Select particle i and j
    pi = particles[i]
    pj = particles[j]

    # Calculate q
    d = sqrt(d2)
    q = d/H

    # Calculate relative distance between them
    rel = x-y

    # Calculate Wendland Kernel Values
    Wi = WendlandKernel(q,Sim.Constants.aD)

    # Calculate Gradient Values
    Wg_ij     = calcGradientW(H, q, x-y)
    Wg_ji     = calcGradientW(H, q, y-x)

    # Calculate Continuity Equation
    vij   = pi.velocity - pj.velocity
    vji   = -vij
    mj    = Sim.Constants.mass
    ρi    = pi.density
    ρj    = pj.density

    # Caclulate Momentum Equation
    Pi            = pressure_eqn_of_state(ρi,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0)
    Pj            = pressure_eqn_of_state(ρj,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0)
    ρij           = (ρi+ρj)/2
    visc_ij       = Π(Sim.Constants.α,Sim.Constants.c0,Sim.Constants.h,vij,rel,ρij)*0
    visc_ji       = Π(Sim.Constants.α,Sim.Constants.c0,Sim.Constants.h,vji,-rel,ρij)*0 #############################
    acci          = -(mj)*((Pi+Pj)/(ρi*ρj) + visc_ij) * Wg_ij
    accj          = -(mj)*((Pi+Pj)/(ρi*ρj) + visc_ji) * Wg_ji

    # Output particles
    pi_out = particles_out[i]
    pj_out = particles_out[j]

    # Add Kernel Data #REMEMBER WE MISS W(0) HERE
    pi_out.W += Wi
    pj_out.W += Wi
    # Add Kernel Gradient Data
    pi_out.WG += Wg_ij
    pj_out.WG += Wg_ji
    # Add Viscosity Data
    pi_out.Visc += visc_ij
    pj_out.Visc += visc_ji

    # Now do some time stepping and updating data
        # # Perform the first half of the position update:
        # dt = Sim.dt*0
        # position_half_step = pi.velocity * dt / 2
        # pi_out.position  += position_half_step
        # # Update the velocity of the particle using the acceleration:
        # pi_out.velocity  += dt*acci
        # # Update the position
        # pi_out.position += (dt/2)*pi_out.velocity 

        # position_half_step = pj.velocity * dt / 2
        # pj_out.position  += position_half_step
        # # Update the velocity of the particle using the acceleration:
        # pj_out.velocity  += dt*acci
        # # Update the position
        # pj_out.position += (dt/2)*pj_out.velocity
    return q_arr
end

function run_map(Sim,box,cl,parts,cutoff)
    N           = length(parts)
    q_arr       = zeros(N)

    for ii in eachindex(parts)
        parts[ii].W  = 0
        parts[ii].WG = SVector(0,0,0)
    end

    parts_out = deepcopy(parts)

    # Run pairwise computation
    map_pairwise!(
        (x,y,i,j,d2,q_arr) -> calc_loop!(x,y,i,j,d2,cutoff/2,parts,parts_out,Sim,q_arr),
        q_arr,box,cl
    )

    return parts_out
end


function time_step_clm(Sim)
    if Sim.it == 0
    parts  = Sim.Fluid.particles;
    x      = getfield.(parts,:position)
    cutoff = 2*Sim.Constants.h
    box    = Box(limits(x),cutoff)
    cl     = CellList(x,box)
    
    parts_out = run_map(Sim,box,cl,parts,cutoff)

    it = lpad(Sim.iter,4,"0")
    @printf "Iteration: %s | dt = %.5e" it Sim.dt

    return parts_out
end

function RunSimulation_clm(Sim)
    # Define the maximum number of iterations:
    max_iter = 100

    # Loop over all iterations:
    while Sim.iter < max_iter
        # Perform an action every 100 iterations:
        if Sim.iter % 1 == 0
            # Create .vtp files for the fluid particles and the wall particles:
            create_vtp_file(Sim.Fluid, "./particles/fluid_particles"*lpad(Sim.iter,4,"0")*".vtp")
            create_vtp_file(Sim.Boundary, "./particles/wall_particles"*lpad(Sim.iter,4,"0")*".vtp")
        end
        
        # Increment the counter:
        Sim.iter += 1;
        stats = @timed  time_step_clm(Sim)
        Sim.Fluid.particles = stats.value
        @printf " | Execution Time: %.5e [s] \n" stats.time
    end

    return Sim
end

foreach(rm, filter(endswith(".vtp"), readdir("./particles",join=true)))
RunSimulation_clm(Sim)