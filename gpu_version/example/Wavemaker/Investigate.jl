# Run from gpu_version/ with Julia --project=. -t 2,0.
include(joinpath(@__DIR__, "Wavemaker.jl"))
using CUDA, CSV, Statistics, LinearAlgebra

const CL = SPHExampleGPU.SPHCellList
const INVESTIGATION_OUTPUT = joinpath(@__DIR__, "..", "..", "output", "wavemaker_investigation")

function leakage(result, case)
    particles = result.final_state.particles
    fluid = particles.GroupMarker .== 3
    active = fluid .& (particles.Type .!= Frozen)
    points = particles.Position[active]
    clearance = [WavemakerGeometry.bed_clearance(p[1],p[3]) for p in points]
    return (; time=result.final_state.time, fluid_total=count(fluid), fluid_active=count(active),
        excluded=count(fluid .& (particles.Type .== Frozen)),
        active_below_half_dx=count(<(-case.dx/2),clearance), active_below_006=count(<(-0.06),clearance),
        min_active_clearance=minimum(clearance), min_active_z=minimum(last,points),
        steps=result.performance.steps, rebuilds=result.diagnostics.cell_list_rebuilds,
        readbacks=result.diagnostics.step_state_readbacks, seconds=result.performance.wall_time_seconds)
end

function check_reference(path)
    case = wavemaker_geometry()
    table = CSV.File(path; header=4, normalizenames=false)
    for (points,code) in zip(values(case.positions),(0,1,3))
        expected = Set(SVector{3,Float32}(row[1],row[2],row[3]) for row in table if row.Type==code)
        actual = Set(SVector{3,Float32}.(points))
        @assert actual==expected "Generated positions differ from the exported reference"
        println("Matched type ",code,": ",length(actual)," positions")
    end
end

function containment(duration=10.0)
    case = wavemaker_geometry()
    directory = joinpath(INVESTIGATION_OUTPUT,"matched_dbc_$(duration)s")
    result = run_wavemaker(; simulation_time=duration, visualize=false,open_log_file=false,save_location=directory)
    report = leakage(result,case)
    println("MATCHED_DBC ",report)
    open(joinpath(directory,"containment.txt"),"w") do io
        println(io,report)
        println(io,result.configuration)
    end
    return result
end

# These methods are changed only by this standalone timing process.
function search_variant!(bounded)
    if bounded
        @eval CL periodic_bounds(sup) = isempty(sup.PeriodicBounds) ? nothing : sup.PeriodicBounds
    else
        @eval CL periodic_bounds(sup) = nothing
    end
end

function benchmark_run(duration,directory,cell_coordinates,fast_math)
    return run_wavemaker(; simulation_time=duration,output_interval=duration,
        periodic_cell_coordinates=cell_coordinates,fast_math,visualize=false,open_log_file=false,
        save_location=directory)
end

function timing(experiment=:search)
    records = NamedTuple[]
    reference = nothing
    reference_work = nothing
    for (round,variant) in enumerate((false,true,true,false))
        search_variant!(experiment===:search ? variant : true)
        cell_coordinates = experiment===:coordinates ? variant : true
        fast_math = experiment===:fastmath && variant
        Base.invokelatest(benchmark_run,0.01,joinpath(INVESTIGATION_OUTPUT,"$(experiment)_warm_$(round)"),cell_coordinates,fast_math)
        result = Base.invokelatest(benchmark_run,0.1,joinpath(INVESTIGATION_OUTPUT,"$(experiment)_timing_$(round)"),cell_coordinates,fast_math)
        particles = result.final_state.particles
        order = sortperm(particles.ID)
        snapshot = (; positions=particles.Position[order],velocity=particles.Velocity[order],density=particles.Density[order])
        work = (result.performance.steps,result.diagnostics.cell_list_rebuilds,result.diagnostics.step_state_readbacks)
        if reference===nothing
            reference,reference_work = snapshot,work
        elseif experiment===:search
            @assert snapshot==reference "Search bounds changed the final particle state"
            @assert work==reference_work "Search bounds changed step/rebuild/readback counts"
        end
        differences = (; position=maximum(norm.(snapshot.positions.-reference.positions)),
            velocity=maximum(norm.(snapshot.velocity.-reference.velocity)),
            density=maximum(abs.(snapshot.density.-reference.density)))
        record = (; round,variant,seconds=result.performance.wall_time_seconds,
            steps=work[1],rebuilds=work[2],readbacks=work[3],differences...)
        push!(records,record)
        println("TIMING ",record)
    end
    search_variant!(true)
    CSV.write(joinpath(INVESTIGATION_OUTPUT,"paired_matched_$(experiment).csv"),records)
    println("Median baseline/variant: ",median(r.seconds for r in records if !r.variant)," / ",median(r.seconds for r in records if r.variant))
end

function profile_run()
    benchmark_run(0.01, joinpath(INVESTIGATION_OUTPUT, "profile_warm"), true, true)
    result = benchmark_run(0.1, joinpath(INVESTIGATION_OUTPUT, "profile_unprofiled"), true, true)
    println("UNPROFILED ", (; seconds = result.performance.wall_time_seconds,
        steps = result.performance.steps, rebuilds = result.diagnostics.cell_list_rebuilds,
        readbacks = result.diagnostics.step_state_readbacks))
    # Profiling overhead is excluded from the elapsed-time comparison above.
    profile = CUDA.@profile external=false trace=false begin
        benchmark_run(0.1, joinpath(INVESTIGATION_OUTPUT, "profile_traced"), true, true)
    end
    display(profile)
    return nothing
end

if abspath(PROGRAM_FILE)==@__FILE__
    mode = isempty(ARGS) ? "containment" : ARGS[1]
    if mode=="containment"
        containment(length(ARGS)>=2 ? parse(Float64,ARGS[2]) : 10.0)
    elseif mode=="check"
        check_reference(ARGS[2])
    elseif mode in ("timing","timing_coordinates","timing_fastmath")
        timing(mode=="timing" ? :search : mode=="timing_coordinates" ? :coordinates : :fastmath)
    elseif mode=="profile"
        profile_run()
    else
        error("Use containment [duration], check initial.csv, timing, timing_coordinates, timing_fastmath, or profile")
    end
end
