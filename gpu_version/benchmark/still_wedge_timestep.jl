using SPHExampleGPU, CUDA, CSV, TOML, Statistics, LinearAlgebra

const WORK = mkpath(joinpath(@__DIR__,"..","validation cases","still_wedge_timestep","runs"))
const RESULTS = mkpath(joinpath(@__DIR__,"..","validation cases","still_wedge_timestep","results"))
const EXAMPLE = abspath(joinpath(@__DIR__,"..","example","StillWedgeMDBC.jl"))
source = read(EXAMPLE,String)
safe_source = replace(source,r"\nrun_still_wedge\([^\r\n]*\)\s*\z"=>"\n")
source != safe_source || error("Example startup guard did not match; refusing to execute it")
include_string(Main,safe_source,EXAMPLE)

function run_variant(legacy,label)
    directory = mktempdir(WORK; prefix=label*"_")
    normpath(dirname(directory)) == normpath(abspath(WORK)) || error("Benchmark output escaped workspace")
    elapsed = @elapsed result = open(joinpath(directory,"console.log"),"w") do io
        redirect_stdout(io) do
            run_still_wedge(; legacy_timestep=legacy,save_location=directory,visualize=false,
                open_log_file=false,generate_pdf_report=false)
        end
    end
    result.final_state.complete && result.diagnostics.frozen_particles == 0 || error("Incomplete StillWedge run")
    particles = result.final_state.particles
    order = sortperm(particles.ID)
    state = hcat([Float64[x[1],x[2]] for x in particles.Position[order]]...)'
    velocity = hcat([Float64[x[1],x[2]] for x in particles.Velocity[order]]...)'
    state = hcat(state,velocity,Float64.(particles.Density[order]))
    all(isfinite,state) || error("Nonfinite particle state")
    fluid = particles.Type .== Fluid
    metrics = (; velocity_rms=sqrt(mean(sum(abs2,v) for v in particles.Velocity[fluid])),
        velocity_max=maximum(norm,particles.Velocity[fluid]),
        density_rms=sqrt(mean((Float64(r)/1000-1)^2 for r in particles.Density[fluid])))
    return result,elapsed,state,metrics,directory
end

records = NamedTuple[]
states = Dict{Bool,Matrix{Float64}}()
for legacy in (true,false)
    println("Full-duration warmup: ",legacy ? "legacy" : "pairwise"); flush(stdout)
    run_variant(legacy,"warm")
    GC.gc(true); CUDA.reclaim()
end
for repetition in 1:5
    variants = isodd(repetition) ? (true,false) : (false,true)
    for legacy in variants
        println("Pair ",repetition,": ",legacy ? "legacy" : "pairwise"); flush(stdout)
        GC.gc(true); CUDA.reclaim()
        result,elapsed,state,metrics,directory = run_variant(legacy,"pair$(repetition)")
        old = get!(states,legacy,copy(state))
        difference = maximum(abs,state-old)
        difference == 0 || error("Repeated particle state changed within a variant")
        p,d = result.performance,result.diagnostics
        push!(records,(; repetition,criterion=legacy ? "legacy" : "pairwise",wall_seconds=elapsed,
            solver_seconds=p.wall_time_seconds,steps=p.steps,frames=d.output_frames,particles=d.particle_count,
            rebuilds=d.cell_list_rebuilds,readbacks=d.step_state_readbacks,repeat_state_max_difference=difference,metrics...))
        CSV.write(joinpath(RESULTS,"timings.csv"),records)
    end
end
delta = abs.(states[false]-states[true])
open(joinpath(RESULTS,"state_difference.toml"),"w") do io
    TOML.print(io,Dict("maximum_position_difference"=>maximum(delta[:,1:2]),
        "maximum_velocity_difference"=>maximum(delta[:,3:4]),"maximum_density_difference"=>maximum(delta[:,5]));sorted=true)
end
open(joinpath(RESULTS,"settings.toml"),"w") do io
    TOML.print(io,Dict("example"=>"example/StillWedgeMDBC.jl","boundary"=>"NoMDBC (actual example default)",
        "duration"=>4.,"output_interval"=>0.01,"dx"=>0.02,"cfl"=>0.5,"x_cell_subdivision"=>4,
        "radial_interactions"=>true,"cache_interaction_fields"=>true,"grid_output"=>true,
        "pdf_report"=>false,"gpu"=>CUDA.name(CUDA.device()),"julia"=>string(VERSION),
        "cuda_jl"=>string(pkgversion(CUDA)),"cuda_driver"=>string(CUDA.driver_version()),
        "timing"=>"Full warmup each, five alternating unprofiled pairs; end-to-end includes particle generation and output",
        "precision"=>"Float32 physics, Float64 positions");sorted=true)
end
println("Completed five matched pairs; all runs complete, finite, unfrozen and repeatable.")
