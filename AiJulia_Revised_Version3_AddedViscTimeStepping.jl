using StaticArrays
using WriteVTK
using LinearAlgebra
using CSV, DataFrames
using Printf
using CellListMap
using FastPow
using NearestNeighbors
using CUDA
CUDA.allowscalar(true)

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

function P(density,Sim)
    return pressure_eqn_of_state(density,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0)
end

function GPU_SIM(Sim,parts,idxs_arr)

    dt = Sim.dt

    r_cpu  = getfield.(parts,:position)
    v_cpu  = getfield.(parts,:velocity)
    ρ_cpu  = getfield.(parts,:density)
    α_cpu  = getfield.(parts,:acceleration)
    WG_cpu = getfield.(parts,:WG)


    MotionLimiter = CuArray(getfield.(parts,:MotionLimiter))
    GravityVector = CuArray(getfield.(parts,:GravityFactor) .* fill(SVector(0,Sim.Constants.g,0),length(idxs_arr),))
    g = CuArray.(idxs_arr)
    r_    = CuArray(r_cpu)
    v_    = CuArray(v_cpu)
    ρ_    = CuArray(ρ_cpu)
    α_    = CuArray(α_cpu)

    r    = map(ind->r_[ind],g)
    v    = map(ind->v_[ind],g)
    ρ    = map(ind->ρ_[ind],g)

    #- infront of x, to be allowed to use view inside of map, really important for performance it seems. To get p_i - p_j = -(p_j-p_i)
    # Cannot put minus inside since it becomes -q, which break calcGradientW
    rel_r  = map(x->-broadcast(-,x,@view x[1]),r)
    rel_v  = map(x->-broadcast(-,x,@view x[1]),v)
    r_norm = map(x->norm.(x),rel_r);
    q      = broadcast(/,r_norm,Sim.Constants.h)
    Wg     = map((x,y)->calcGradientW.(Sim.Constants.h, x, y),q,rel_r);

    dρdt_n =   map((x,y,z) -> dot.(broadcast(dot,(Sim.Constants.mass ./ x) .* y,z),@view x[1]),ρ,rel_v,Wg)

    #ρij     = map(x->broadcast(*,x[2:end],@view x[1]),ρ);
    #Pij     = map(x->broadcast(+,x[2:end],@view x[1]),P);
    dvdt_n  = α_ + GravityVector                      #@views sum.(dvdt_n_); #SLOWEST LINE..

    
    ρ_i_n_half =  ρ_ .+ CuArray(sum.(dρdt_n))*(dt/2)
    v_i_n_half =  v_ .+ CuArray(dvdt_n) * (dt/2)


    ρ_n_half      = map(ind->ρ_i_n_half[ind],g)
    v_n_half      = map(ind->v_i_n_half[ind],g)
    
    rel_v_n_half  = map(x->-broadcast(-,x,@view x[1]),v_n_half) #Different from CPU implementation
    #rel_v_n_half  = map((x,y) -> -broadcast(-,x,@view y[1]),v_n_half,v) #Similar to CPU implementation
   
    ρij_n_half    = map(x->broadcast(*,x[2:end],@view x[1]),ρ_n_half);

    P_n_half      = map(x->pressure_eqn_of_state.(x,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0),ρ_n_half)
    Pij_n_half    = map(x->broadcast(+,x[2:end],@view x[1]),P_n_half);


    # DON'T FORGET MINIS IN MOMENTUM EQUATION
    # IT IS NOT DOT IN MOMENTUM EQUATION
    dvdt_n_half_ = map((x,y,z)-> -Sim.Constants.mass * (x./y) .* z[2:end],Pij_n_half,ρij_n_half,Wg)
    dvdt_n_half  = CuArray(sum.(dvdt_n_half_)) + GravityVector
    
    dρdt_n_half   = map((x,y,z) -> dot.(broadcast(dot,(Sim.Constants.mass ./ x) .* y,z),@view x[1]),ρ_n_half,rel_v_n_half,Wg) #NOTE WE LACK TO MULTPLY WITH RHO_I WE DO THAT LATER

    epsi = - dt*map((x,y)->sum(x./y),dρdt_n_half,ρ_n_half) 

    # # TEMP
    # ρ_n_final    = ρ_i_n_half
    # v_n_final    = v_i_n_half
    # α_n_final    = dvdt_n_half

    ρ_n_final  = ρ_ .* CuArray((2 .- epsi)./(2 .+ epsi))
    v_n_final  = v_ .+  MotionLimiter.*dvdt_n_half*dt
    r_n_final  = r_ .+  MotionLimiter.*(v_n_final .+ v_)*0.5*dt
    α_n_final  = dvdt_n_half
    WG_n_final = sum.(Wg)

    r_cpu  .= Array(r_n_final)
    v_cpu  .= Array(v_n_final)
    ρ_cpu  .= Array(ρ_n_final)
    α_cpu  .= Array(α_n_final)
    WG_cpu .= Array(WG_n_final)

    for i in eachindex(parts)
        p_update = parts[i]
        p_update.position      = r_cpu[i]
        p_update.velocity      = v_cpu[i]
        p_update.density       = ρ_cpu[i]
        p_update.acceleration  = α_cpu[i]
        p_update.WG            = WG_cpu[i]
    end
end

# Define the time step function:

function time_step2(Sim,list)
    parts   = [Sim.Fluid.particles;Sim.Boundary.particles]
    dt      = Sim.dt
    H       = Sim.Constants.h
    mi = mj = Sim.Constants.mass

    ρ_ini   = deepcopy(getfield.(parts,:density))
    v_ini   = deepcopy(getfield.(parts,:velocity))
    x_ini   = deepcopy(getfield.(parts,:position))

    N      = length(parts)

    # Loop 1
    dρdt_n = zeros(N)
    dvdt_n = getfield.(parts,:GravityFactor) .* fill(SVector(0,Sim.Constants.g,0),N)
    
    Threads.@threads for L in list
        i = L[1]; j = L[2]; d = L[3];

        ρi = parts[i].density
        ρj = parts[j].density
        Pi = P(ρi,Sim)
        Pj = P(ρj,Sim)
        vi = parts[i].velocity
        vj = parts[j].velocity
        xi = parts[i].position
        xj = parts[j].position

        dxij  = xi - xj
        vij   = vi - vj
        
        q  = d/H

        wg = calcGradientW(H,q,dxij)

        dρidt_n = dot((mj/ρj) * vij,wg)
        dρjdt_n = dot((mi/ρi) * -vij,-wg)

        dρdt_n[i] += dρidt_n; dρdt_n[j] += dρjdt_n;

        dvidt_n = - mj * (Pi+Pj)/(ρi*ρj) * wg 

        dvdt_n[i] += dvidt_n; dvdt_n[j] += -dvidt_n;
    end

    # Update half time steps
    dρ_n_half = zeros(N)
    dv_n_half = fill(SVector(0.0,0.0,0.0),N)
    dx_n_half = fill(SVector(0.0,0.0,0.0),N) + getfield.(parts,:GravityFactor) .* fill(SVector(0,Sim.Constants.g,0),N)
    for i in eachindex(parts)
        dρ_n_half[i] = dρdt_n[i]*(dt/2)
        dv_n_half[i] = dvdt_n[i]*(dt/2)
        dx_n_half[i] = parts[i].velocity*(dt/2) ####
    end

    # Loop 2
    WG_n_half     = fill(SVector(0.0,0.0,0.0),N)
    dρdt_n_half   = zeros(N)
    dvdt_n_half   = fill(SVector(0.0,0.0,0.0),N)
    
    for L in list
        i = L[1]; j = L[2]; d = L[3];

        ρi = parts[i].density + parts[i].density*dρ_n_half[i]
        ρj = parts[j].density + parts[j].density*dρ_n_half[j]



        Pi = P(ρi,Sim)
        Pj = P(ρj,Sim)
        vi = parts[i].velocity + dv_n_half[i]
        vj = parts[i].velocity + dv_n_half[j]
        xi = parts[i].position + dx_n_half[i]
        xj = parts[j].position + dx_n_half[j]

        dxij  = xi - xj

        vij   = vi - vj
        
        q  = d/H

        wg   = calcGradientW(H,q,dxij)

        WG_n_half[i] += wg; WG_n_half[j] += -wg;

        dρidt_n_half = dot((mj/ρj) * vij,wg)
        dρjdt_n_half = dot((mi/ρi) * -vij,-wg)

        dρdt_n_half[i] += dρidt_n_half; dρdt_n_half[j] += dρjdt_n_half;

        dvidt_n_half = -mj * (Pi+Pj)/(ρi*ρj) * wg

        dvdt_n_half[i] += dvidt_n_half; dvdt_n_half[j] += -dvidt_n_half;
    end

     # Update to final time steps
     # Remember we have to remove contribution from previous time steps
     # before the final calculation.. lets improve this later!
     for i in eachindex(parts)
        GFi = parts[i].GravityFactor
        MLi = parts[i].MotionLimiter

        epsi_i = -(dρdt_n_half[i]/(parts[i].density + dρdt_n[i]))*dt

        parts[i].acceleration  = dvdt_n_half[i] + GFi*SVector(0,Sim.Constants.g,0)
        parts[i].WG            = WG_n_half[i]
        parts[i].density       = ρ_ini[i]*((2+epsi_i)/(2-epsi_i))
        parts[i].velocity      = v_ini[i] + MLi*dvdt_n_half[i]*dt
        parts[i].position      = x_ini[i] + MLi*(parts[i].velocity + v_ini[i])*0.5*dt
    end

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

function time_step(Sim,system,idxs_arr,bool)

    parts =  [Sim.Fluid.particles;Sim.Boundary.particles]

    H  = Sim.Constants.h
    dt = Sim.dt
    

    list     = neighborlist!(system);
    function convert_idx(x, list)
        out  = [ Int[] for _ in x ]
        rout = [ Float64[] for _ in x ]

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
     idxs_arr,rout = convert_idx(parts,list)

     if bool == false
        # Loop over all fluid particles:
        for i  in eachindex(idxs_arr)
            idxs = idxs_arr[i]
            particle_update = parts[i]
            particle = deepcopy(particle_update)

            rel      = (particle.position,) .- @view(getfield.(parts,:position)[idxs])
            r        = norm.(rel)
            q        = r ./ H

            Wg_ij     = calcGradientW.(H, q, rel)

            ρ_i  = particle.density
            ρ_j  = @view(getfield.(parts,:density)[idxs])
            v_i  = particle.velocity
            v_j  = @view(getfield.(parts,:velocity)[idxs])

            v_ij = (v_i,) .- v_j 
        
            # Think there is some bug here, does not look very stable to me :)
            function Ψ(Sim,rel,ρ_i,ρ_j,Wg)
                ρ0    = Sim.Constants.rho0
                g     = Sim.Constants.g
                Cb    = Sim.Constants.Cb
                gamma = Sim.Constants.g
                cbar  = Sim.Constants.c0
                DDTkh = 2*Sim.Constants.h*0.1
                mass  = Sim.Constants.mass

                DDTgz = ρ0*abs(g)/Cb

                drz = rel[2] #y is our z
                rh  = 1 + DDTgz * drz
                dρ  = ρ0 * ^(rh,1/gamma) - ρ0

                rr2   = dot(rel,rel)
                eta2  = (0.1*Sim.Constants.h)*(0.1*Sim.Constants.h)

                visc_densi = DDTkh*cbar*(ρ_j-ρ_i-dρ)/(rr2+eta2)

                dot3  = dot(rel,Wg)

                delta = visc_densi*dot3*(mass/ρ_j)

                return delta
            end

            #ZERO DDT FOR NOW
            ddt_term_n = 0* sum(map((x,y,z) -> Ψ(Sim,z,ρ_i,x,y),ρ_j,Wg_ij,rel))
            
            dρ_i_dt_n = ρ_i * sum(dot((Sim.Constants.mass ./ ρ_j) .* v_ij, Wg_ij)) + ddt_term_n
            dv_i_dt_n = particle.acceleration + particle.GravityFactor*SVector(0,Sim.Constants.g,0)

            ρ_i_n_half = particle.density  + dρ_i_dt_n*(dt/2)

            if i == 1
            print("OLD")
            println(ρ_i_n_half)
            end

            v_i_n_half = particle.velocity + dv_i_dt_n*(dt/2)
            v_ij_n_half = (v_i_n_half,) .- v_j #This is actually wrong, since we do not know v_j_n_half yet

            Pi_n_half  = pressure_eqn_of_state(ρ_i_n_half,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0)
            Pj_n_half  = pressure_eqn_of_state.(ρ_j,Sim.Constants.rho0,Sim.Constants.gamma,Sim.Constants.c0)

            cond = dot.(v_ij_n_half,rel)

            visc_bool = cond .< 0

            # ZERO VISC FOR NOW
            α = 0.00

            eta2    = (0.01*H)*(0.01*H)
            mu_ab   = (H*cond) ./ (dot.(rel,rel) .+ eta2)
            visc_i  = visc_bool .* (-α*Sim.Constants.c0*mu_ab) ./ ((ρ_i,) .+ ρ_j)

            # ZERO DDT FOR NOW
            ddt_term_n_half = 0*sum(map((x,y,z) -> Ψ(Sim,z,ρ_i_n_half,x,y),ρ_j,Wg_ij,rel))
            dv_i_dt_n_half = sum(-(((Sim.Constants.mass)*((Pi_n_half .+ Pj_n_half) ./ ((ρ_i,) .*ρ_j))) .+ visc_i ).* Wg_ij) + particle.GravityFactor*SVector(0,Sim.Constants.g,0)

            dρ_i_dt_n_half = ρ_i_n_half * sum(dot((Sim.Constants.mass ./ ρ_j) .* v_ij_n_half, Wg_ij))

            epsi = -(dρ_i_dt_n_half/ρ_i_n_half) * dt

            particle_update.density  = ρ_i *((2-epsi)/(2+epsi))
            particle_update.velocity = v_i + particle.MotionLimiter*dv_i_dt_n_half*dt
            particle_update.position = particle_update.position + particle.MotionLimiter*((particle_update.velocity + v_i)/2) * dt
            particle_update.acceleration = dv_i_dt_n_half
            # particle_update.Visc         = sum(visc_i)
            # particle_update.ddt          = ddt_term_n_half
            particle_update.WG           = sum(Wg_ij)
        end
    else
        GPU_SIM(Sim,parts,idxs_arr)
    end

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

function RunSimulation(Sim,max_iter,bool,ext)

    system = InPlaceNeighborList(x=getfield.([Sim.Fluid.particles;Sim.Boundary.particles],:position), cutoff=2*Sim.Constants.h, parallel=true)

    idxs_arr = Vector{Vector{Int64}}(undef,length([Sim.Fluid.particles;Sim.Boundary.particles]))

    # Loop over all iterations:
    while Sim.iter < max_iter
        # Perform an action every 100 iterations:
        if Sim.iter % 1 == 0
            # Create .vtp files for the fluid particles and the wall particles:
            create_vtp_file(Sim.Fluid, "./particles/fluid_particles_"*ext*lpad(Sim.iter,4,"0")*".vtp")
            #create_vtp_file(Sim.Boundary, "./particles/wall_particles"*lpad(Sim.iter,4,"0")*".vtp")
        end

        x_new = getfield.([Sim.Fluid.particles;Sim.Boundary.particles],:position)
        update!(system,x_new)
        list = neighborlist!(system);
        # Increment the counter:
        Sim.iter += 1;
        #stats = @timed time_step(Sim,system,idxs_arr,bool)
        stats = @timed time_step2(Sim,list)
        @printf " | Execution Time: %.5e [s] \n" stats.time
    end
end

function RunSimulationOLD(Sim,max_iter,bool,ext)

    system = InPlaceNeighborList(x=getfield.([Sim.Fluid.particles;Sim.Boundary.particles],:position), cutoff=2*Sim.Constants.h, parallel=true)

    idxs_arr = Vector{Vector{Int64}}(undef,length([Sim.Fluid.particles;Sim.Boundary.particles]))

    # Loop over all iterations:
    while Sim.iter < max_iter
        # Perform an action every 100 iterations:
        if Sim.iter % 1 == 0
            # Create .vtp files for the fluid particles and the wall particles:
            create_vtp_file(Sim.Fluid, "./particles/fluid_particles_"*ext*lpad(Sim.iter,4,"0")*".vtp")
            #create_vtp_file(Sim.Boundary, "./particles/wall_particles"*lpad(Sim.iter,4,"0")*".vtp")
        end

        x_new = getfield.([Sim.Fluid.particles;Sim.Boundary.particles],:position)
        update!(system,x_new)
        list = neighborlist!(system);
        # Increment the counter:
        Sim.iter += 1;
        #stats = @timed time_step(Sim,system,idxs_arr,bool)
        stats = @timed time_step(Sim,system,idxs_arr,bool)
        @printf " | Execution Time: %.5e [s] \n" stats.time
    end
end

### Run

#Sim = Simulation(dt=1e-4,h=0.141421,c0=81.675,dx=0.1,rho0=1000)
Consts = Constants(dt_ini=1e-4,h=0.056569,c0=85.89,dx=0.04,rho0=1000)
#Consts  = Constants(dt_ini=1e-4,h=0.028284,c0=87.25,dx=0.02,rho0=1000)
Sim = Simulation(Constants=Consts)
Sim.dt = Sim.Constants.dt_ini

#Sim = Simulation(dt_ini=1e-4,h=0.028284,c0=87.25,dx=0.02,rho0=1000)

# Create a Collection object for the fluid particles:
fluid_particles = Collection(Vector{Particle}())

for i = 1:2#size(DF_FLUID)[1]
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

# for i = 1:size(DF_BOUND)[1]
#     idp = DF_BOUND[i,:]["Idp"]
#     pos = SVector(DF_BOUND[i,:]["Points:0"],DF_BOUND[i,:]["Points:2"],DF_BOUND[i,:]["Points:1"])
#     acc = SVector(0,0,0)
#     vel = SVector(0.0, 0.0, 0.0)
#     # Create a new Particle object with the calculated position:
#     particle = Particle(pos,acc,vel, Sim.Constants.rho0, idp,0,-1,0,0,SVector(0,0,0),0)

#     # Add the particle to the wall_particles collection:
#     push!(wall_particles.particles, particle)
# end

Sim.Boundary = wall_particles
Sim.Fluid    = fluid_particles

Sim_         = deepcopy(Sim)

foreach(rm, filter(endswith(".vtp"), readdir("./particles",join=true)))
iters = 51

Sim = deepcopy(Sim_)
RunSimulationOLD(Sim,iters,false,"CPU_OLD")
println("OLD CPU")
println(Sim.Fluid.particles[1])

Sim = deepcopy(Sim_)
RunSimulation(Sim,iters,false,"CPU_NEW")
println("NEW CPU")
println(Sim.Fluid.particles[1])
