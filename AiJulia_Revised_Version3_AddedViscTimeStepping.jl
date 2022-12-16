using StaticArrays
using WriteVTK
using LinearAlgebra
using CSV, DataFrames


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

    function Particle()
        position = SVector(NaN, NaN, NaN)
        acceleration = SVector(NaN, NaN, NaN)
        velocity = SVector(NaN, NaN, NaN)
        density = NaN
        id = -1
        Visc = NaN
        W = NaN
        WG = SVector(NaN, NaN, NaN)
        new(position, acceleration, velocity, density, id,Visc,W,WG)
    end

    function Particle(position, acceleration, velocity, density, id, Visc,W,WG)
        new(position, acceleration, velocity, density, id,Visc,W,WG)
    end
end

mutable struct Collection
    particles::Vector{Particle}

    function Collection(particles)
        new(particles)
    end
end

Base.@kwdef mutable struct Simulation
    dx::Float64 = 0.1
    dt::Float64 = 0.0001
    h::Float64  = 1.5*sqrt(2*0.1^2)
    c0::Float64 = 0
    rho0::Float64 = 1000
    gamma::Float64 = 7
    α::Float64     = 0.01
    Boundary::Collection
    Fluid::Collection
end

# Define the Wendland kernel function:
function WendlandKernel(q,h)
    # Define aD:
    aD = 7 / (4 * pi * h^2)

    if q < 0 || q > 2
        return 0.0
    end

    return aD * (1 - q / 2)^4 * (2 * q + 1)
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
function calcGradientW(particle, h, q,r)
    gradWx = 0
    gradWy = 0
    gradWz = 0

    # Skip distances outside the support of the kernel:
    if q < 0 || q > 2
        return SVector(0.0,0.0,0.0)
    end
    # Remember
    #           q = r/h = sqrt(x^2+y^2+z^2)
    #           r = q*h.
    # We add a small number to avoid dividing by zero.
    #println("q*h: $(q*h) | r = $(norm(r))")

    gradWx += 7 / (4 * pi * h^2) * 1/h * (5*(q-2)^3*q)/8 * (r[1] / (q*h+1e-6))
    gradWy += 7 / (4 * pi * h^2) * 1/h * (5*(q-2)^3*q)/8 * (r[2] / (q*h+1e-6))
    gradWz += 7 / (4 * pi * h^2) * 1/h * (5*(q-2)^3*q)/8 * (r[3] / (q*h+1e-6)) 

    return SVector(gradWx,gradWy,gradWz)
end

# Define the pressure equation of state:
function pressure_eqn_of_state(density, initial_density, gamma, c0)
    # Calculate the pressure using the given equation:
    pressure = (c0^2 * initial_density / gamma) * ((density / initial_density)^gamma - 1)
    return pressure
end

# https://www.symbolab.com/solver/step-by-step/solve%20for%20d%2C%20%20%20p%20%3D%20%5Cleft(c%5E%7B2%7D%20%5Ccdot%20k%20%2F%207%5Cright)%20%5Ccdot%5Cleft(%5Cleft(d%20%2F%20k%5Cright)%5E%7B7%7D%20-%201%5Cright)?or=input
function density_eqn_of_state(particle, mwl, initial_density, gamma, c0)
    height = mwl - particle.position[2]
    pressure = height*initial_density*9.81;
    # Calculate the pressure using the given equation:
    density = initial_density * ((gamma*pressure+initial_density*c0^2)/(initial_density*c0^2))^(1/gamma)
    return density
end

# Define the continuity equation for SPH:
function continuity_eqn(particle, Sim, h, dx)
    # Initialize the time derivative of the density to zero:
    time_deriv_density = 0
    particle.WG =        SVector(0.0,0.0,0.0)

    # Loop over all fluid particles:
    #for p in Sim.Fluid.particles
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
        mb   = p.density * dx^2

        # Calculate the gradient of the kernel for the two particles:
        gradW = calcGradientW(particle, h, q, r)

        # Calculate the contribution of the other particle to the time derivative of the density:
        time_deriv_density += rhoa * dot((mb/p.density)*vab, gradW)

        particle.WG += gradW;
    end
    
    return time_deriv_density
end

function Π(α,c0,h,vab,rab,rhoab)

    cond = dot(vab,rab)

    if cond < 0
        eta2 = 0.01*h^2
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
    g = -9.81
    acceleration = SVector(0, g, 0)

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
        gradW = calcGradientW(particle, h, q, r)

        rhob           = p.density;
        mb             = rhob*dx^2;
        Pb             = pressure_eqn_of_state(rhob,Sim.rho0,Sim.gamma,Sim.c0)
        rhoa           = particle.density

        rhoab          = (rhoa+rhob)/2

        visc           = Π(Sim.α,Sim.c0,Sim.h,vab,r,rhoab)
        particle.Visc  = visc;

        Pa             = pressure_eqn_of_state(rhoa,Sim.rho0,Sim.gamma,Sim.c0)
        acceleration  += -mb*((Pb+Pa)/(rhob*rhoa) + visc) * gradW
    end

    return acceleration
end



# Define the time step function:
function time_step(Sim)
    h  = Sim.h
    dt = Sim.dt
    dx = Sim.dx

    # Loop over all fluid particles:
    for particle in Sim.Fluid.particles
        # Start particle velocity and density
        vn = particle.velocity
        rn = particle.density

        # Calculate the acceleration of the particle using the inviscid momentum equation:
        particle.acceleration = inviscid_momentum_eqn(particle, Sim, h, dx)

        # Perform the first half of the position update:
        position_half_step = particle.velocity * dt / 2
        particle.position += position_half_step

        # Update the velocity of the particle using the acceleration:
        velocity_half_step = particle.acceleration * dt /2
        particle.velocity  += velocity_half_step

        # Calculate the time derivative of the density of the particle using the continuity equation:
        time_deriv_density = continuity_eqn(particle, Sim, h, dx)
        
        # Update the density of the particle using the time derivative of the density:
        density_half_step = time_deriv_density * dt/2
        particle.density += density_half_step

        # Calculate the acceleration of the particle using the inviscid momentum equation:
        particle.acceleration = inviscid_momentum_eqn(particle, Sim, h, dx)

        # Update the velocity of the particle using the acceleration:
        particle.velocity += particle.acceleration * dt - velocity_half_step

        # Perform the second half of the position update:
        particle.position += dt * ((particle.velocity+vn)/2)  - position_half_step

        # Final update of density
        epsi = -(time_deriv_density/particle.density) * dt

        particle.density = rn * ((2-epsi)/(2+epsi))
    end    

    for particle in Sim.Boundary.particles
        # Start particle velocity and density and position
        pn = particle.position
        vn = particle.velocity
        rn = particle.density

        # Calculate the acceleration of the particle using the inviscid momentum equation:
        particle.acceleration = inviscid_momentum_eqn(particle, Sim, h, dx) - SVector(0,-9.81,0)

        # Perform the first half of the position update:
        position_half_step = particle.velocity * dt / 2
        particle.position += position_half_step

        # Update the velocity of the particle using the acceleration:
        velocity_half_step = particle.acceleration * dt /2
        particle.velocity  += velocity_half_step

        # Calculate the time derivative of the density of the particle using the continuity equation:
        time_deriv_density = continuity_eqn(particle, Sim, h, dx)
        
        # Update the density of the particle using the time derivative of the density:
        density_half_step = time_deriv_density * dt/2
        particle.density += density_half_step

        # Calculate the acceleration of the particle using the inviscid momentum equation:
        particle.acceleration = inviscid_momentum_eqn(particle, Sim, h, dx) - SVector(0,-9.81,0)

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
    end

    # Extract Info relevant for time stepping
    # Inspired by JSphCpu.cpp from DualSPHysics
    CFL     = 0.3
    max_acc = maximum(norm.(getfield.(Sim.Fluid.particles,:acceleration)));
    dt1     =  sqrt(Sim.h/max_acc)

    dt2     = Sim.h / (Sim.c0 + maximum(getfield.(Sim.Fluid.particles,:Visc)))

    dt      = CFL*min(dt1,dt2)

    Sim.dt  = dt;

    if isnan(dt)
        println("Simulation experienced nan time step")
        return 1
    end

    println("Iteration: | dt = $dt")
end

#Sim = Simulation(dt=1e-4,h=0.141421,c0=81.675,dx=0.1,rho0=1000)
Sim = Simulation(dt=1e-4,h=0.056569,c0=85.89,dx=0.04,rho0=1000)
#Sim = Simulation(dt=1e-4,h=0.028284,c0=87.25,dx=0.02,rho0=1000)

# Create a Collection object for the fluid particles:
fluid_particles = Collection(Vector{Particle}())

for i = 1:size(DF_FLUID)[1]
    idp = DF_FLUID[i,:]["Idp"]
    pos = SVector(DF_FLUID[i,:]["Points:0"],DF_FLUID[i,:]["Points:2"],DF_FLUID[i,:]["Points:1"])
    acc = SVector(0.0, 0.0, 0.0)
    vel = SVector(0.0, 0.0, 0.0)
    # Create a new Particle object with the calculated position:
    particle = Particle(pos,acc,vel, DF_FLUID[i,:]["Rhop"], idp,0,0,SVector(0,0,0))

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
    particle = Particle(pos,acc,vel, Sim.rho0, idp,0,0,SVector(0,0,0))

    # Add the particle to the wall_particles collection:
    push!(wall_particles.particles, particle)

    # Create a new Particle object with the calculated position:
    #particle = Particle(pos-SVector(Sim.dx/2,Sim.dx/2,0.0),acc,vel, Sim.rho0, idp,0,SVector(0,0,0))

    # Add the particle to the wall_particles collection:
    #push!(wall_particles.particles, particle)
end

Sim.Boundary = wall_particles
Sim.Fluid    = fluid_particles



# Define the create_vtp_file subfunction:
function create_vtp_file(collection::Collection, filename::String)
    # Create a vector of the x, y, and z positions of the particles:
    positions = [
        [particle.position[1], particle.position[2], particle.position[3]]
        for particle in collection.particles
    ]

    # Create a vector of the particle densities:
    densities  = [particle.density for particle in collection.particles]
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
        vtk_point_data(vtk, accelerations, "acceleration")
        vtk_point_data(vtk, velocities, "velocity")
        vtk_point_data(vtk, kernelW, "kernel")
        vtk_point_data(vtk, kernelWG, "kernel_gradient")
        vtk_point_data(vtk, viscocities, "Viscosity")
        vtk_point_data(vtk,pressure_eqn_of_state.(densities,Sim.rho0,Sim.gamma,Sim.c0),"pressure")
    end
end

function RunSimulation(Sim)
    # Define the maximum number of iterations:
    max_iter = 40000

    # Define the counter for the number of iterations:
    counter = 0
    create_vtp_file(Sim.Fluid, "./particles/fluid_particles"*lpad(counter,4,"0")*".vtp")
    create_vtp_file(Sim.Boundary, "./particles/wall_particles"*lpad(counter,4,"0")*".vtp")

    # Loop over all iterations:
    while counter < max_iter
        # Increment the counter:
        counter += 1
        time_step(Sim)
        # Perform an action every 100 iterations:
        if counter % 50 == 0
            # Create .vtp files for the fluid particles and the wall particles:
            create_vtp_file(Sim.Fluid, "./particles/fluid_particles"*lpad(counter,4,"0")*".vtp")
            create_vtp_file(Sim.Boundary, "./particles/wall_particles"*lpad(counter,4,"0")*".vtp")
        end
    end
end

    
