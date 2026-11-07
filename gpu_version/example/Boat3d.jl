# 3D towing-tank flow past a fixed STL hull. See Boat3d.md for units and placement.
using SPHExampleGPU
using StaticArrays

"""
    boat_setup(stl_path; dx=0.01, duration=2.0, scale=1,
               translation=(0.75,0,0.13), ...)

Create a fixed-hull current test with the DualSPHysics 5.4 CurrentHull flow
settings. STL coordinates must describe a closed solid in metres after the
explicit scale/translation. The hull stays fixed; mass and inertia are not
inferred from the surface mesh. No CUDA allocation occurs during setup.
"""
function boat_setup(stl_path=joinpath(@__DIR__,"data","current_hull","principal_hull.stl");T=Float32,dx=0.01,duration=2.0,scale=1,
        translation=(0.75,0,0.13),inlet=-0.2,outlet=2.0,half_width=0.5,
        water_level=0.2,velocity=1.5,packing_iterations=30,
        save_location=joinpath(@__DIR__,"..","output","boat3d"),
        output_interval=0.01,use_graph=true,sph_kernel=WendlandC2())
    all(isfinite,(dx,duration,inlet,outlet,half_width,water_level,velocity,output_interval)) &&
        dx > 0 && duration > 0 && half_width > dx && water_level > dx &&
        velocity > 0 && outlet > inlet && output_interval > 0 ||
        throw(ArgumentError("invalid tank dimensions, spacing, speed or duration"))
    mesh = load_stl(stl_path;scale,translation)
    # Use full-volume particles for a fixed DBC solid; no surface-only gaps.
    hull = sample_solid(mesh,dx;iterations=packing_iterations)
    hull.quality.minimum_spacing >= 0.65dx ||
        throw(ArgumentError("hull packing has particles closer than 0.65dx; refine dx or repair thin features"))
    lo,hi = mesh.nodes[1].lower,mesh.nodes[1].upper
    h = 2dx
    H = 2h
    inlet+H < lo[1] && hi[1] < outlet-H &&
        -half_width+H < lo[2] && hi[2] < half_width-H && lo[3] > H ||
        throw(ArgumentError("hull must clear the inlet/outlet, floor and side walls by a kernel support radius"))
    lo[3] < water_level < hi[3] || throw(ArgumentError("water level must intersect the hull"))
    layers = max(5,ceil(Int,H/dx)+1)
    flow = OpenChannel{T}(Inlet=inlet,Outlet=outlet,WaterLevel=water_level,
        Velocity=velocity,Spacing=dx,Duration=duration,InletLayers=layers,OutletLayers=max(8,layers),HydrostaticCorrection=true)
    ys = collect(-half_width+dx:dx:half_width-dx)
    zs = collect(dx:dx:water_level)
    points = SVector{3,Float64}[SVector(x,y,z) for x in inlet+dx:dx:outlet-dx for y in ys for z in zs
        if signed_distance(mesh,SVector(x,y,z)) >= dx/2]
    seeds(x) = SVector{3,Float64}[SVector(x,y,z) for y in ys for z in zs]
    hydrostatic(p) = T(1000*(1+7*9.81*max(0,water_level-p[3])/60^2)^(1/7))
    function region(p,marker,kind)
        velocities = fill(kind == Fluid ? SVector{3,T}(velocity,0,0) : zero(SVector{3,T}),length(p))
        return SPHGeometry{3,T}(Particles=particle_struct_array(p,T(1000);
            Density=hydrostatic.(p),Velocity=velocities),GroupMarker=marker,Type=kind)
    end
    # Extend floor and side walls under both open buffers. Leave both ends open.
    wall_set = Set{SVector{3,Float64}}()
    xs = inlet-(layers-1)*dx:dx:outlet+(flow.outlet_layers-1)*dx
    for x in xs, y in -half_width-(layers-1)*dx:dx:half_width+(layers-1)*dx, k in 0:layers-1
        push!(wall_set,SVector(x,y,-k*dx))
    end
    wall_height = max(water_level+H,hi[3]+H)
    for x in xs, z in dx:dx:wall_height, k in 0:layers-1, sign in (-1,1)
        push!(wall_set,SVector(x,sign*(half_width+k*dx),z))
    end
    geometry = open_channel_geometry([
        region(points,1,Fluid),region(seeds(inlet),2,Fluid),region(seeds(outlet),3,Fluid),
        region(sort!(collect(wall_set);by=Tuple),10,Fixed),region(hull.positions,20,Fixed)],flow)
    constants = SimulationConstants{T}(3;dx,g=9.81,c₀=60,α=0.02,CFL=0.2)
    kernel = SPHKernelInstance{3,T}(sph_kernel;h=T(h))
    metadata = SimulationMetaData{3,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName="Boat3D",SaveLocation=save_location,SimulationTime=duration,
        OutputTimes=output_interval,VisualizeInParaview=false,OpenLogFile=false,
        ExportSingleVTKHDF=true,GPUDoublePosition=true,GPUUseGraph=use_graph,
        GPUClipTimeStepToOutput=true)
    return (;geometry,flow,constants,kernel,metadata,mesh,packing=hull.quality)
end

"""Run the fixed STL hull case and return the simulation result."""
function run_boat(stl_path=joinpath(@__DIR__,"data","current_hull","principal_hull.stl");kwargs...)
    setup = boat_setup(stl_path;kwargs...)
    (;geometry,flow,constants,kernel,metadata) = setup
    @info "Hull particle discretization" setup.packing
    mkpath(metadata.SaveLocation)
    logger = SimulationLogger(metadata.SaveLocation)
    try
        return RunSimulation(SimGeometry=geometry,SimMetaData=metadata,
            SimParticles=AllocateDataStructures(geometry,metadata),SimConstants=constants,
            SimKernel=kernel,SimLogger=logger,SimViscosity=ArtificialViscosity(boundary_factor=0),
            SimDensityDiffusion=FourtakasDensityDiffusion(),SimTimeStepping=SymplecticTimeStepping(),
            SimOpenBoundaries=flow)
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) ? run_boat() : run_boat(ARGS[1])
end

