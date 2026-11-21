# Definitions only: constructing a case never starts a simulation.
using SPHExampleGPU, StaticArrays, LinearAlgebra, StructArrays

"""Nearest point on the physical rounded float section, including its top cap."""
function float_section_projection(p, release, base)
    radius, corner = release.radius, release.corner
    core, center_z, top = radius-corner,base+corner,base+release.height
    x,z = p
    side = x >= 0 ? 1. : -1.
    candidates = [SVector(x,top),SVector(side*radius,clamp(z,center_z,top))]
    if abs(x) <= core
        push!(candidates,SVector(x,base))
    else
        r = SVector(abs(x)-core,min(z-center_z,0.))
        direction = iszero(r) ? SVector(1.,0.) : r/norm(r)
        push!(candidates,SVector(side*(core+corner*direction[1]),center_z+corner*direction[2]))
    end
    return candidates[argmin(norm.(candidates .- Ref(p)))]
end

"""
    build_planar_case(release; boundary_mode=NoMDBC, ...)

Construct a Cartesian sensitivity case. `SimpleMDBC` reflects boundary ghosts
across the physical tank and rounded float surfaces. Rigid mass and particle
positions are identical between modes; the experiment itself is axisymmetric.
"""
function build_planar_case(release; dx=0.0125,depth=0.8,half_width=8.,wall_top_z=0.25,
        sponge_width=1.,sponge_rate=12.,duration=4.5,output_interval=0.005,
        kernel_family=WendlandC2(),kernel_h_factor=2.,boundary_mode=NoMDBC,
        diffusion=LinearDensityDiffusion(),viscosity=Laminar(),c0=25.,cfl=0.2,pause_time=0.,
        save_location=joinpath(@__DIR__,"runs","planar_mdbc",release.name))
    boundary_mode in (NoMDBC,SimpleMDBC) || throw(ArgumentError("Use NoMDBC or SimpleMDBC; updated/cloned mDBC does not support floating bodies"))
    all(isfinite,(dx,depth,half_width,duration,output_interval,kernel_h_factor)) &&
        dx > 0 && depth > 0 && half_width > release.radius && duration > 0 && output_interval > 0 &&
        kernel_h_factor > 0 || throw(ArgumentError("Invalid planar geometry or time settings"))
    nx,nz = round.(Int,(2half_width/dx,depth/dx))
    isapprox(nx*dx,2half_width) && isapprox(nz*dx,depth) || throw(ArgumentError("Spacing must divide width and depth"))
    0 < sponge_width < half_width || throw(ArgumentError("Damping width must fit inside the tank"))
    T = Float32
    constants = SimulationConstants{T}(2;dx,ρ₀=1000.,g=9.81,c₀=c0,ν₀=1e-6,CFL=cfl,δᵩ=0.1)
    kernel = SPHKernelInstance{2,T}(kernel_family;h=T(kernel_h_factor*dx))
    equilibrium_center_z = release.cog-release.draft
    base = release.offset-release.draft
    layers = ceil(Int,kernel.H/dx)
    top = ceil(Int,(depth+max(wall_top_z,base+release.height))/dx)
    wall_top = ceil(Int,(depth+wall_top_z)/dx)
    wall,water,body = (SVector{2,Float64}[] for _ in 1:3)
    for iz in -layers:top, ix in -layers:nx+layers-1
        x,z = (ix+0.5-nx/2)*dx,(iz+0.5)*dx-depth
        p = SVector(x,z)
        if iz < 0 || !(0 <= ix < nx)
            iz <= wall_top && push!(wall,p)
        elseif in_float(SVector(x,0.,z),release,base)
            push!(body,p)
        elseif iz < nz
            push!(water,p)
        end
    end
    isempty(body) && throw(ArgumentError("No body points at this spacing"))
    core = release.radius-release.corner
    area = 2core*release.corner+π*release.corner^2/2+2release.radius*(release.draft-release.corner)
    line_mass = constants.ρ₀*area
    floating = FloatingDetails{T}(RelativeWeight=line_mass/(constants.m₀*length(body)),PauseTime=pause_time,
        Center=(0.,equilibrium_center_z+release.offset),
        Constraints=FloatingConstraints{2,T}(TranslationLocks=(true,false),RotationLocks=(true,)))
    density(points) = hydrostatic_density(points,constants;water_level=0.)
    ghost(p) = SVector(p[1] < -half_width ? -2half_width-p[1] :
        p[1] > half_width ? 2half_width-p[1] : p[1],p[2] < -depth ? -2depth-p[2] : p[2])
    wall_ghosts = ghost.(wall)
    body_ghosts = [2float_section_projection(p,release,base)-p for p in body]
    boundary_data(points,ghosts) = StructArray((;Position=points,Density=density(points),
        GhostPoints=ghosts,GhostNormals=ghosts.-points))
    tank = SPHGeometry{2,T}(Particles=boundary_data(wall,wall_ghosts),Type=Fixed,GroupMarker=1)
    fluid = SPHGeometry{2,T}(water;Density=density(water),Type=Fluid,GroupMarker=2)
    solid = SPHGeometry{2,T}(Particles=boundary_data(body,body_ghosts),Type=Floating,GroupMarker=3,Floating=floating)
    damping = (DampingZone((half_width-sponge_width,0.),(half_width,0.);rate=sponge_rate),
        DampingZone((-half_width+sponge_width,0.),(-half_width,0.);rate=sponge_rate))
    metadata = SimulationMetaData{2,T,NoShifting,NoKernelOutput,boundary_mode,StoreLog}(
        SimulationName="FreeHeave",SaveLocation=save_location,SimulationTime=duration,OutputTimes=output_interval,
        OutputVariables=["Type","Velocity","Pressure"],ExportSingleVTKHDF=true,
        VisualizeInParaview=false,OpenLogFile=false,GPUDoublePosition=true,GPUClipTimeStepToOutput=true)
    return (;geometry=[tank,fluid,solid],metadata,constants,kernel,damping,equilibrium_center_z,
        dx,half_width,depth,duration,viscosity,diffusion,stepping=SymplecticTimeStepping(),line_mass,
        boundary_mode=string(nameof(boundary_mode)),wall_ghosts,body_ghosts)
end
