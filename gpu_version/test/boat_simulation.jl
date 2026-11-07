using SPHExampleGPU, CUDA, Test
include(joinpath(@__DIR__,"..","example","Boat3d.jl"))
@testset "DualSPHysics principal hull geometry" begin
    setup = boat_setup(dx=0.025,duration=0.03,packing_iterations=30,
        save_location=joinpath(@__DIR__,"..","output","boat_validation"),output_interval=0.01)
    @info "Hull setup" setup.packing particles=sum(length(g.Particles.Position) for g in setup.geometry)
    @test setup.packing.minimum_clearance >= 0.0125-1e-8
    @test setup.packing.minimum_spacing >= 0.65*0.025
    if CUDA.functional()
        mkpath(setup.metadata.SaveLocation)
        logger = SimulationLogger(setup.metadata.SaveLocation)
        result = try
            RunSimulation(SimGeometry=setup.geometry,SimMetaData=setup.metadata,
                SimParticles=AllocateDataStructures(setup.geometry,setup.metadata),SimConstants=setup.constants,
                SimKernel=setup.kernel,SimLogger=logger,SimViscosity=ArtificialViscosity(boundary_factor=0),
                SimDensityDiffusion=FourtakasDensityDiffusion(),SimTimeStepping=SymplecticTimeStepping(),
                SimOpenBoundaries=setup.flow)
        finally
            isopen(logger.LoggerIo) && close(logger.LoggerIo)
        end
        particles = result.final_state.particles
        @test all(p -> all(isfinite,p),particles.Position)
        @test all(isfinite,particles.Density)
        @test all(>(0),particles.Density)
        state = result.final_state.open_boundary
        @test state.emitted_particles > 0
        @test state.active_fluid == state.initial_active_fluid+state.emitted_particles-state.retired_particles
        @info "Hull CUDA smoke result" state extrema(particles.Density)
    end
end
