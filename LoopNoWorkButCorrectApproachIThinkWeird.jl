# Loop over all fluid particles:
    for (particle,particle_update) in zip(Sim_.Fluid.particles,Sim.Fluid.particles)
        #
        pold = deepcopy(particle)

        # Start particle velocity and density
        vn = particle.velocity
        rn = particle.density

        # Initially calculate drho/dt at time = n
        # Calculate the time derivative of the density of the particle using the continuity equation:
        time_deriv_density_time_n = continuity_eqn(particle, Sim_, h, dx)

        # Perform the first half of the position update:
        position_half_step = particle.velocity * dt / 2
        particle.position += position_half_step

        # Update the velocity of the particle using the acceleration:
        velocity_half_step = particle.acceleration * dt /2
        particle.velocity  += velocity_half_step

        # And calculate the derivative of density again at vel^n+½
        time_deriv_density_time_n_half = continuity_eqn(particle, Sim_, h, dx)

        # Calculate the acceleration of the particle using the inviscid momentum equation:
        dvadt_halfstep,Visc_halfstep = inviscid_momentum_eqn(particle, Sim_, h, dx)
        particle.velocity += -velocity_half_step + dt*dvadt_halfstep

        # Perform the second half of the position update:
        particle.position += dt * ((particle.velocity+vn)/2)  - position_half_step

        
        # Update the density of the particle using the time derivative of the density:
        density_half_step = time_deriv_density_time_n * dt/2
        particle.density  += density_half_step
       
        # Final update of density
        epsi = -(time_deriv_density_time_n_half/particle.density) * dt

        particle.density = rn * ((2-epsi)/(2+epsi))

        # Update
        for f_ in f
            setfield!(particle_update,f_,getfield(particle,f_))
        end

        particle        = pold
    end    