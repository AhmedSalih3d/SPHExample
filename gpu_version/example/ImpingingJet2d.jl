# DualSPHysics 5.4 inletoutlet/08_ImpingingJet: geometry and simulation setup.
# Running or including this file starts the simulation. For inspection/tests,
# define RUN_IMPINGING_JET = false before including it.
using SPHExampleGPU
using StaticArrays: SVector

"""Build the complete impinging-jet setup, including the mDBC plate and open zones."""
function impinging_jet_setup(; spacing=0.0005,duration=0.1,output_interval=0.0002,
        save_location=joinpath(@__DIR__,"..","output","impinging_jet"),use_graph=true,
        float_type=Float32,cache_interaction_fields=false,boundary_mode=SimpleMDBC)
    T = float_type
    boundary_mode in (SimpleMDBC,UpdatedMDBC) || throw(ArgumentError("the impinging jet requires mDBC"))
    dx = Float64(spacing)
    isfinite(dx) && dx > 0 || throw(ArgumentError("jet spacing must be finite and positive"))
    jet_width = 0.03
    initial_jet_bottom = 0.12
    inlet_height = 0.125
    plate_half_width = 0.2
    outlet_height = 0.03
    buffer_layers = 4
    jet_velocity = 20.0
    reference_density = 1000.0
    all(x -> isapprox(x/dx,round(x/dx);atol=1e-8,rtol=0),
        (plate_half_width,jet_width/2,inlet_height-initial_jet_bottom,inlet_height,outlet_height)) ||
        throw(ArgumentError("jet spacing must divide the reference dimensions"))

    flow = ImpingingJet{T}(Spacing=dx,Duration=duration,Velocity=jet_velocity,
        Inlet=(-jet_width/2,jet_width/2,inlet_height),Outlets=(-plate_half_width,plate_half_width),
        OutletHeight=outlet_height,Layers=buffer_layers,
        VerticalLimits=(-buffer_layers*dx,inlet_height+0.1*(inlet_height+(buffer_layers-1)*dx)))
    points(xs,ys) = [SVector{2,Float64}(x,y) for x in xs for y in ys]
    region(p,marker,kind;velocity=zero(SVector{2,T})) = SPHGeometry{2,T}(
        Particles=particle_struct_array(p,T(reference_density);Velocity=fill(velocity,length(p))),
        GroupMarker=Int(marker),Type=kind)

    # GenCase uses a half-open domain: the final plate column is at 0.2+5dx.
    wall_positions = points(-plate_half_width-6dx:dx:plate_half_width+5dx,-(buffer_layers-1)*dx:dx:0.)
    # The upstream normal plane is z=dx/2. Store wall-to-ghost displacement,
    # twice GenCase's boundary-to-interface normal, for reflection in that plane.
    wall_ghosts = [SVector(p[1],dx-p[2]) for p in wall_positions]
    plate = SPHGeometry{2,T}(Particles=particle_struct_array(wall_positions,T(reference_density);
        GhostPoints=wall_ghosts,GhostNormals=wall_ghosts.-wall_positions),GroupMarker=10,Type=Fixed)
    nozzle_x = collect(-jet_width/2:dx:jet_width/2)
    geometry = [plate,
        region(points(nozzle_x,initial_jet_bottom:dx:inlet_height-dx),1,Fluid;velocity=SVector{2,T}(0,-jet_velocity)),
        region(points(nozzle_x,inlet_height:dx:inlet_height+(buffer_layers-1)*dx),2,Fluid;velocity=SVector{2,T}(0,-jet_velocity)),
        region(points(-plate_half_width-(buffer_layers-1)*dx:dx:-plate_half_width,dx:dx:outlet_height),3,Fluid),
        region(points(plate_half_width:dx:plate_half_width+(buffer_layers-1)*dx,dx:dx:outlet_height),4,Fluid)]
    emission_events = ceil(Int,duration*jet_velocity/dx)+1
    reserve = [SVector{2,Float64}(x,inlet_height+(buffer_layers-1)*dx) for _ in 1:emission_events for x in nozzle_x]
    push!(geometry,region(reserve,flow.reserve_marker,Frozen))

    metadata = SimulationMetaData{2,T,NoShifting,NoKernelOutput,boundary_mode,StoreLog}(
        SimulationName="ImpingingJet2D",SaveLocation=save_location,SimulationTime=duration,
        OutputTimes=output_interval,VisualizeInParaview=false,OpenLogFile=false,
        ExportSingleVTKHDF=true,GPUDoublePosition=false,GPUUseGraph=use_graph,
        GPUCacheInteractionFields=cache_interaction_fields,GPUClipTimeStepToOutput=true)
    constants = SimulationConstants{T}(2;dx=dx,ρ₀=reference_density,g=9.81,c₀=230,CFL=0.1,ν₀=1e-6)
    kernel = SPHKernelInstance{2,T}(WendlandC2();h=T(sqrt(2)*dx))
    return (;geometry,flow,metadata,constants,kernel)
end

"""Run the setup above with original mDBC, Laminar+SPS and full Fourtakas diffusion."""
function run_impinging_jet(;kwargs...)
    (;geometry,flow,metadata,constants,kernel) = impinging_jet_setup(;kwargs...)
    mkpath(metadata.SaveLocation)
    logger = SimulationLogger(metadata.SaveLocation)
    try
        return RunSimulation(SimGeometry=geometry,SimMetaData=metadata,
            SimParticles=AllocateDataStructures(geometry,metadata),SimConstants=constants,
            SimKernel=kernel,SimLogger=logger,SimViscosity=LaminarSPS(),
            SimDensityDiffusion=FourtakasDensityDiffusion(full=true),SimTimeStepping=SymplecticTimeStepping(),
            SimOpenBoundaries=flow)
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end

if !@isdefined(RUN_IMPINGING_JET) || RUN_IMPINGING_JET
    run_impinging_jet()
end
