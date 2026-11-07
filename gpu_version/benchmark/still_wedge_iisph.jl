# Run the existing StillWedge example through its public solver-selection keyword.
# This is an accuracy/output case run, not a paired solver-speed benchmark.
using SPHExampleGPU, CUDA, HDF5, LinearAlgebra, Statistics

let path = joinpath(@__DIR__,"..","example","StillWedgeMDBC.jl")
    source = replace(read(path,String),r"\r?\n# Including this file starts the simulation\.[\s\S]*\z" => "\n")
    include_string(@__MODULE__,source,path)
end

function run_still_wedge_iisph_case(output_dir; simulation_time=4.0,
        solver=IISPH(max_timestep=0.002),generate_pdf_report=true)
    result = run_still_wedge(;solver,simulation_time,save_location=output_dir,
        visualize=false,open_log_file=false,generate_pdf_report)
    p = result.final_state.particles
    fluid = findall(==(Fluid),p.Type)
    floor_clearance(x,y) = y-max(0.0,0.26-abs(x-1.1))
    maximum_intrusion = 0.0
    escaped_ids = Set{Int}()
    frames = 0
    h5open(joinpath(output_dir,"StillWedge.vtkhdf"),"r") do io
        root = io["VTKHDF"]
        times = read(root["Steps/Values"])
        offsets = read(root["Steps/PointOffsets"])
        for frame in eachindex(times)
            columns = Int(offsets[frame])+1:Int(offsets[frame])+length(p)
            position = root["Points"][:,columns]
            types = root["PointData/Type"][columns]
            ids = root["PointData/ID"][columns]
            for i in eachindex(types)
                types[i] == Int8(Fluid) || continue
                x,y = position[1,i],position[2,i]
                maximum_intrusion = max(maximum_intrusion,-floor_clearance(x,y))
                # Beyond the exterior floor/side faces or inside the hollow
                # wedge cavity, allowing half a spacing for volume samples.
                escaped = x < -0.05 || x > 2.25 || y < -0.05 ||
                    (abs(x-1.1) < 0.2 && y < 0.2-abs(x-1.1)-0.01)
                escaped && push!(escaped_ids,Int(ids[i]))
            end
        end
        frames = length(times)
    end
    statistics = (;particle_count=length(p),fluid_count=length(fluid),
        final_time=result.final_state.time,steps=result.performance.steps,frames,
        wall_time_seconds=result.performance.wall_time_seconds,
        density_min=minimum(p.Density[fluid]),density_max=maximum(p.Density[fluid]),
        max_speed=maximum(norm,p.Velocity[fluid]),pressure_min=minimum(p.Pressure[fluid]),
        pressure_max=maximum(p.Pressure[fluid]),maximum_intrusion,
        escaped_count=length(escaped_ids),iisph=result.diagnostics.iisph)
    open(joinpath(output_dir,"StillWedge_IISPH_case_summary.txt"),"w") do io
        println(io,"GPU: ",CUDA.name(CUDA.device()),"; Julia ",VERSION)
        for (key,value) in pairs(statistics)
            println(io,key," = ",value)
        end
        println(io,"escaped_ids = ",sort!(collect(escaped_ids)))
    end
    println(statistics)
    return (;result,statistics)
end

if abspath(PROGRAM_FILE) == @__FILE__
    positional = filter(arg -> !startswith(arg,"--"),ARGS)
    destination = isempty(positional) ? joinpath(@__DIR__,"..","output","StillWedgeIISPH") : positional[1]
    duration = length(positional) >= 2 ? parse(Float64,positional[2]) : 4.0
    timestep = length(positional) >= 3 ? parse(Float64,positional[3]) : 0.002
    run_still_wedge_iisph_case(destination;simulation_time=duration,
        solver=IISPH(max_timestep=timestep),generate_pdf_report=!("--no-report" in ARGS))
end
