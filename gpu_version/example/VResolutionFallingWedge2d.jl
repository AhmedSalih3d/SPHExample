# A free falling wedge enters water at -6.15 m/s. Two body-following refinement
# boxes give three simultaneous resolutions, dp, dp/2 and dp/4.
using SPHExampleGPU
using StaticArrays: SVector
include(joinpath(@__DIR__,"VResolutionFallingWedge2d","Setup.jl"))

"Build the falling-wedge geometry and input structs without launching CUDA."
function falling_wedge_case(;
        dx = 0.002,
        duration = 0.025,
        output_interval = 0.0005,
        initial_velocity = -6.15,
        body_mass = 241.0,
        FloatType = Float32,
        regions = (VariableResolutionBox((-0.49,1.31),(0.49,1.90);ratio=2),
            VariableResolutionBox((-0.39,1.40),(0.40,1.798);ratio=2)),
        save_location = joinpath(@__DIR__,"..","output","vresolution_falling_wedge2d"))
    Dimensions = 2
    T = FloatType
    T in (Float32,Float64) || throw(ArgumentError("FloatType must be Float32 or Float64"))
    isfinite(dx) && 0 < dx <= 0.02 || throw(ArgumentError("dx must lie in (0,0.02]; enlarge regions for coarse checks"))
    isfinite(initial_velocity) && initial_velocity < 0 || throw(ArgumentError("initial_velocity must be negative and finite"))
    isfinite(body_mass) && body_mass > 0 || throw(ArgumentError("body_mass must be positive and finite"))
    isfinite(duration) && duration > 0 && isfinite(output_interval) && output_interval > 0 ||
        throw(ArgumentError("duration and output_interval must be positive and finite"))
    length(regions) == 2 || throw(ArgumentError("this example uses two nested refinement boxes"))
    constants = ntuple(3) do level
        spacing = dx/prod(r.ratio for r in regions[1:level-1];init=1)
        SimulationConstants{T}(Dimensions;
            dx = spacing,
            c₀ = 80,
            γ = 7,
            g = 9.81,
            α = 0.01,
            δᵩ = 0.1,
            CFL = 0.2,
        )
    end
    kernels = map(constants) do c
        h = 2Float64(c.dx)
        SPHKernelInstance{Dimensions,T}(WendlandC2();h=T(h),η²=T((0.1h)^2))
    end
    geometry = ntuple(3) do level
        spacing = dx/prod(r.ratio for r in regions[1:level-1];init=1)
        points = falling_wedge_particles(spacing,kernels[level],level == 1 ? nothing : regions[level-1];finest=level == 3)
        Tank = SPHGeometry{Dimensions,T}(;
            Particles = particle_struct_array(points.fixed,constants[level].ρ₀;
                GhostPoints=points.ghosts,GhostNormals=points.ghosts-points.fixed),
            Type = Fixed,
            GroupMarker = 20,
        )
        Water = SPHGeometry{Dimensions,T}(points.water;
            Density = hydrostatic_density(points.water,constants[level];water_level=1.5),
            Type = Fluid,
            GroupMarker = 2,
        )
        groups = [Tank,Water]
        if level == 3
            details = FloatingDetails{T}(;
                RelativeWeight = body_mass/(length(points.body)*constants[level].m₀),
                InitialVelocity = (0.0,initial_velocity),
            )
            ghosts = [falling_wedge_ghost(p,spacing) for p in points.body]
            Wedge = SPHGeometry{Dimensions,T}(;
                Particles = particle_struct_array(points.body,constants[level].ρ₀;
                    GhostPoints=ghosts,GhostNormals=ghosts-points.body),
                Type = Floating,
                GroupMarker = 51,
                Floating = details,
            )
            push!(groups,Wedge)
        end
        groups
    end
    return (;geometry,constants,kernels,regions,initial_velocity,body_mass,duration,output_interval,save_location=abspath(save_location))
end

"Run three coupled GPU levels and save particles, wedge and both moving boxes in one VTKHDF."
function run_falling_wedge(;particle_reserve=nothing,visualize=true,graphs=true,kwargs...)
    case = falling_wedge_case(;kwargs...)
    T = typeof(first(case.constants).dx)
    # The descending boxes fill initially dry space. Reserve scales as swept
    # area / dp^2; a fixed count sufficient for a smoke run is too small at dp/4.
    reserves = ntuple(2) do interface
        particle_reserve === nothing || return particle_reserve
        region,c,k = case.regions[interface],case.constants[interface+1],case.kernels[interface+1]
        extent = region.upper-region.lower .+ 2region.buffer_h*k.h
        travel = abs(case.initial_velocity)*case.duration+0.5Float64(c.g)*case.duration^2
        max(20000,ceil(Int,1.5extent[1]*min(travel,extent[2])/Float64(c.dx)^2))
    end
    inner = VariableResolutionConfig(case.regions[2];
        fine_geometry = case.geometry[3],
        fine_constants = case.constants[3],
        fine_kernel = case.kernels[3],
        follow = 51,
        particle_reserve = reserves[2],
    )
    refinement = VariableResolutionConfig(case.regions[1];
        fine_geometry = case.geometry[2],
        fine_constants = case.constants[2],
        fine_kernel = case.kernels[2],
        follow = 51,
        child = inner,
        particle_reserve = reserves[1],
    )
    metadata = SimulationMetaData{2,T,NoShifting,NoKernelOutput,SimpleMDBC,StoreLog}(;
        SimulationName = "FallingWedge2D",
        SaveLocation = case.save_location,
        SimulationTime = case.duration,
        OutputTimes = case.output_interval,
        VariableResolution = refinement,
        ExportSingleVTKHDF = true,
        ExportGridCells = false,
        GPUUseGraph = graphs,
        GPUClipTimeStepToOutput = true,
        # Open after adding the animated geometry, below.
        VisualizeInParaview = false,
        OpenLogFile = false,
        OutputVariables = ["Velocity","Density","Pressure","Type","ID","GroupMarker",
            "Resolution","Spacing","Mass","Owned","IsBuffer"],
    )
    viscosity = ArtificialViscosity(;boundary_factor=1)
    diffusion = FourtakasDensityDiffusion(;full=true)
    stepping = SymplecticTimeStepping()
    damping = (DampingZone((1.0,0.0),(1.5,0.0);rate=100),
        DampingZone((-1.0,0.0),(-1.5,0.0);rate=100),DampingZone((0.0,0.4),(0.0,0.0);rate=100))
    mkpath(case.save_location)
    particles = AllocateDataStructures(first(case.geometry),metadata)
    logger = SimulationLogger(case.save_location)
    result = try
        RunSimulation(;
            SimGeometry = first(case.geometry),
            SimConstants = first(case.constants),
            SimKernel = first(case.kernels),
            SimMetaData = metadata,
            SimParticles = particles,
            SimLogger = logger,
            SimViscosity = viscosity,
            SimDensityDiffusion = diffusion,
            SimTimeStepping = stepping,
            SimDamping = damping,
        )
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
    save_falling_wedge_geometry(case,result)
    metadata.VisualizeInParaview = visualize
    AutoOpenParaview(metadata,first(case.constants),metadata.OutputVariables)
    return result
end

# Including this file starts the simulation.
run_falling_wedge()
