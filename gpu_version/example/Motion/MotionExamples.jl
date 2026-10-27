module MotionExamples

using SPHExampleGPU
using StaticArrays
using LinearAlgebra
using HDF5

export motion_example, run_motion_example, motion_frame_times

include("MotionCaseGeometry.jl")
include("MotionCaseTrajectories.jl")

surface_points(shape::PolygonSurface) = shape.points
surface_points(shapes::AbstractVector) = reduce(vcat, surface_points.(shapes))

"""
    motion_example(number; dimensions = 3, FloatType = Float32, dx = nothing)

Build the geometry and commanded motion of DualSPHysics 5.4's distributed
Motion01-09 cases. Shapes, coordinates, axes, tracks, data files and timing
match the reference definitions. These are 3D kinematic cases, without fluid.
Polygon rendering uses the complete surfaces; the optional solver exercise
uses their vertices as a small boundary-particle set, not a GenCase lattice.
"""
function motion_example(number; dimensions = 3, FloatType = Float32, dx = nothing)
    number in 1:9 || throw(ArgumentError("motion example number must be in 1:9"))
    dimensions == 3 || throw(ArgumentError("the distributed DualSPHysics motion cases are 3D; use dimensions = 3"))
    spacing = dx === nothing ? (number == 8 ? 0.02 : number == 9 ? 0.001 : 0.01) : Float64(dx)
    isfinite(spacing) && spacing > 0 || throw(ArgumentError("dx must be finite and positive"))
    parts = case_parts(number)
    motion = case_motion(number)
    names = Tuple(part.name for part in parts)
    regions = NamedTuple{names}(Tuple(part.shape for part in parts))
    moving_ids = Set(node.id for node in motion.motions if node.group_marker !== nothing)
    geometry = [SPHGeometry{3,FloatType}(unique(surface_points(part.shape));
        Density = 1000, GroupMarker = part.mk+1, Type = part.mk in moving_ids ? Moving : Fixed) for part in parts]
    return (; number, parts, regions, geometry, motion, duration = MOTION_DURATIONS[number],
        output_interval = MOTION_INTERVALS[number], dx = spacing,
        constants = SimulationConstants{FloatType}(3;dx=spacing,c₀=10,g=0))
end

function motion_frame_times(duration, interval)
    isfinite(duration) && duration > 0 || throw(ArgumentError("simulation_time must be finite and positive"))
    isfinite(interval) && interval > 0 || throw(ArgumentError("output_interval must be finite and positive"))
    count = floor(Int,duration/interval)
    times = collect(range(0.0;step=Float64(interval),length=count+1))
    if isapprox(last(times),duration;rtol=8eps(Float64),atol=0)
        times[end] = Float64(duration)
    else
        push!(times,Float64(duration))
    end
    return times
end

# Sample a body's pose once per rendered frame, retaining immutable reference
# vertices. This is the same absolute transform used by the particle solver.
function polygon_trajectory(system, id)
    sampled_time = Ref(NaN)
    sampled_pose = Ref(motion_state(system,id,0.0))
    return function(reference,time)
        if sampled_time[] != time
            sampled_pose[] = motion_state(system,id,time)
            sampled_time[] = time
        end
        return motion_position(sampled_pose[],reference)
    end
end

function label_case_regions(path, parts)
    h5open(path,"r+") do file
        root = file["VTKHDF"]
        regions = read(root["CellData/Region"])
        root["CellData/MkBound"] = Int32[parts[region].mk for region in regions]
        root["CellData/Mk"] = Int32[10+parts[region].mk for region in regions]
        if haskey(root,"Steps")
            offsets = read(root["Steps/CellDataOffsets/Region"])
            root["Steps/CellDataOffsets/MkBound"] = offsets
            root["Steps/CellDataOffsets/Mk"] = offsets
        end
    end
end

"""
    run_motion_example(number; simulation_time = nothing, output_interval = nothing,
                       visualize = true, simulate_particles = false, ...)

Write and optionally open a polygon animation, using the distributed case's
full duration and output interval by default. Returns paths, frame times and
the case definition. Set `simulate_particles = true` to additionally exercise
GPU integration on a small boundary-particle set; its SimulationResult is
returned in `simulation`. The primary `MotionXX.vtkhdf` always contains polygon
surfaces, while optional particle output is named `MotionXX_Particles.vtkhdf`.
"""
function run_motion_example(number; dimensions = 3, FloatType = Float32, dx = nothing,
        simulation_time = nothing, output_interval = nothing, visualize = true,
        open_log_file = true, simulate_particles = false, use_graph = true,
        time_stepping = SymplecticTimeStepping(),
        save_location = joinpath(@__DIR__,"..","..","output","Motion","Motion"*lpad(number,2,'0')))
    case = motion_example(number;dimensions,FloatType,dx)
    duration = simulation_time === nothing ? case.duration : Float64(simulation_time)
    interval = output_interval === nothing ? case.output_interval : Float64(output_interval)
    duration <= case.duration || throw(ArgumentError("simulation_time exceeds the reference case's duration $(case.duration)"))
    times = motion_frame_times(duration,interval)
    mkpath(save_location)
    name = "Motion"*lpad(number,2,'0')
    geometry_path = joinpath(save_location,name*"_Geometry.vtkhdf")
    motion_path = joinpath(save_location,name*".vtkhdf")
    moving_parts = [part for part in case.parts if any(node -> node.group_marker == part.mk+1,case.motion.motions)]
    transforms = NamedTuple{Tuple(part.name for part in moving_parts)}(
        Tuple(polygon_trajectory(case.motion,part.mk) for part in moving_parts))
    SavePolygonVTKHDF(geometry_path,case.regions)
    SavePolygonMotionSequence(motion_path,case.regions;motions=transforms,times)
    label_case_regions(geometry_path,case.parts)
    label_case_regions(motion_path,case.parts)
    simulation = nothing
    if simulate_particles
        metadata = SimulationMetaData{3,FloatType,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
            SimulationName=name*"_Particles",SaveLocation=save_location,
            SimulationTime=duration,OutputTimes=interval,GPUUseGraph=use_graph,
            GPUDoublePosition=true,GPUClipTimeStepToOutput=true,GPUCellSubdivision=1,
            ExportGridCells=false,VisualizeInParaview=false,OpenLogFile=open_log_file)
        particles = AllocateDataStructures(case.geometry,metadata)
        logger = SimulationLogger(save_location)
        try
            simulation = RunSimulation(SimGeometry=case.geometry,SimMetaData=metadata,SimParticles=particles,
                SimConstants=case.constants,SimKernel=SPHKernelInstance{3,FloatType}(WendlandC2();dx=FloatType(case.dx)),
                SimLogger=logger,SimTimeStepping=time_stepping,SimViscosity=ZeroViscosity(),
                SimDensityDiffusion=ZeroDensityDiffusion(),SimMotion=case.motion)
        finally
            isopen(logger.LoggerIo) && close(logger.LoggerIo)
        end
    end
    visualize && OpenParaviewFile(motion_path)
    return (;geometry_path,motion_path,times,case,simulation)
end

end
