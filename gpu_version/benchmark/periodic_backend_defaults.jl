# Standard 2D periodic case, preserving its symplectic integrator and diffusion.
# julia --project=. -t 2,0 benchmark/periodic_backend_defaults.jl [duration=0.1] [dx=0.002] [repeats=4]
using SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra
const ROOT=normpath(joinpath(@__DIR__,".."))
const DIRECTORY=mktempdir(joinpath(ROOT,"output");prefix="periodic_default_benchmark_",cleanup=false)
const DURATION=isempty(ARGS) ? 0.1 : parse(Float64,ARGS[1])
const SPACING=length(ARGS)<2 ? 0.002 : parse(Float64,ARGS[2])
const REPEATS=length(ARGS)<3 ? 4 : parse(Int,ARGS[3])
scope=Module(:PeriodicBackendBenchmark)
Core.eval(scope,:(RunSimulation(;kwargs...)=(;kwargs...)))
case_path=joinpath(ROOT,"example","Periodicity2d.jl")
Base.include(expr->expr isa Expr && expr.head===:call && first(expr.args)===:run_periodicity_2d ? nothing : expr,scope,case_path)
setup=Base.invokelatest(getproperty(scope,:run_periodicity_2d);dx=SPACING,simulation_time=DURATION,
    output_interval=min(0.01,DURATION),visualize=false,open_log_file=false,save_location=joinpath(DIRECTORY,"setup"))
function run_backend(images,duration,label)
    meta=deepcopy(setup.SimMetaData)
    meta.GPUPeriodicImages=images
    meta.SimulationTime=duration
    meta.OutputTimes=min(0.01,duration)
    meta.SaveLocation=joinpath(DIRECTORY,label)
    mkpath(meta.SaveLocation)
    particles=AllocateDataStructures(setup.SimGeometry,meta)
    logger=SimulationLogger(meta.SaveLocation)
    try
        result=redirect_stdout(devnull) do
            RunSimulation(;setup...,SimMetaData=meta,SimParticles=particles,SimLogger=logger)
        end
        @assert result.final_state.complete
        @assert result.configuration.effective.periodic_images==images
        @assert result.configuration.effective.time_stepping===SymplecticTimeStepping
        @assert result.configuration.effective.density_diffusion_model===ComplexDensityDiffusion
        @assert result.configuration.execution.graph_capture_failures==0
        @assert result.diagnostics.frozen_particles==0
        @assert all(isfinite,particles.Density)
        @assert all(x->all(isfinite,x),particles.Position)
        return result
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end
println("ARTIFACT_DIRECTORY=$DIRECTORY")
run_backend(false,DURATION,"warm_queries")
run_backend(true,DURATION,"warm_images")
snapshots=Dict{Bool,Any}()
open(joinpath(DIRECTORY,"results.txt"),"w") do io
    for iteration in 1:REPEATS
        images=mod(iteration,4) in (2,3)
        result=run_backend(images,DURATION,"run_$(iteration)_$(images ? "images" : "queries")")
        particles=result.final_state.particles
        order=sortperm(particles.ID)
        snapshots[images]=(;position=particles.Position[order],velocity=particles.Velocity[order],density=particles.Density[order])
        summary="WHOLE_SOLVER images=$images performance=$(result.performance) diagnostics=$(result.diagnostics)"
        println(summary)
        println(io,summary)
        if length(snapshots)==2
            a,b=snapshots[false],snapshots[true]
            periodic=setup.SimMetaData.Periodicity
            period=periodic.upper-periodic.lower
            displacement=[delta+round(delta[periodic.axis]/period)*periodic.translation for delta in b.position-a.position]
            summary="STATE_COMPARISON position_maxdiff=$(maximum(norm.(displacement))) position_rms=$(norm(displacement)/sqrt(length(displacement))) velocity_relative_l2=$(norm(b.velocity-a.velocity)/norm(a.velocity)) density_relative_l2=$(norm(b.density-a.density)/norm(a.density))"
            println(summary)
            println(io,summary)
        end
        flush(stdout)
        flush(io)
        CUDA.reclaim()
    end
end
println("PERIODIC_DEFAULT_BENCHMARK_COMPLETE")
