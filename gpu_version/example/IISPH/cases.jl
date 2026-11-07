module IISPHCases

using SPHExampleGPU, StaticArrays

export tank_case

"""Uniform 2D particles with filled fixed walls and an open free surface."""
function tank_case(::Type{T},save; name="IISPHTank",dx=0.025,nx=24,ny=16,
                   fluid_width=nx,fluid_height=ny,duration=0.1,output_interval=duration,
                   viscosity=Laminar(),nu=1e-6,g=9.81,solver=IISPH(max_timestep=0.002)) where T
    mkpath(save)
    constants = SimulationConstants{T}(2;dx=T(dx),g=T(g),ν₀=T(nu))
    kernel = SPHKernelInstance{2,T}(WendlandC2();dx=constants.dx)
    fluid = [SVector{2,T}(T(i)*constants.dx,T(j)*constants.dx)
             for j in 1:fluid_height for i in 1:fluid_width]
    # Include the full support radius plus a spacing of solid depth, including
    # when a soft-wall leading particle intrudes into the first wall layer.
    layers = ceil(Int,kernel.H/constants.dx)+1
    wall = [SVector{2,T}(T(i)*constants.dx,T(j)*constants.dx)
            for j in (1-layers):(ny+layers) for i in (1-layers):(nx+layers)
            if j <= 0 || i <= 0 || i > nx]
    geometry = [SPHGeometry{2,T}(wall;Density=constants.ρ₀,Type=Fixed,GroupMarker=1),
                SPHGeometry{2,T}(fluid;Density=constants.ρ₀,Type=Fluid,GroupMarker=2)]
    metadata = SimulationMetaData{2,T}(SimulationName=name,SaveLocation=save,
        SimulationTime=duration,OutputTimes=output_interval,
        GPUUseGraph=false,GPUAsyncOutput=false,VisualizeInParaview=false,OpenLogFile=false,
        OutputVariables=["Velocity","Density","Pressure","Type","ID"])
    particles = AllocateDataStructures(geometry,metadata)
    logger = SimulationLogger(save)
    return (;SimGeometry=geometry,SimMetaData=metadata,SimConstants=constants,
        SimKernel=kernel,SimParticles=particles,SimLogger=logger,
        SimViscosity=viscosity,SimDensityDiffusion=ZeroDensityDiffusion(),SimSolver=solver)
end

end
