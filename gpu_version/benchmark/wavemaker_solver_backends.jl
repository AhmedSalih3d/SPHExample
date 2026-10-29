# Whole-solver comparison, including device copy generation and source sorting.
# julia --project=. -t 2,0 benchmark/wavemaker_solver_backends.jl [duration=0.1] [repeats=4]
using SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra, Statistics, HDF5
const CL = SPHExampleGPU.SPHCellList
const ROOT = normpath(joinpath(@__DIR__,".."))
const DIRECTORY = mktempdir(joinpath(ROOT,"output");prefix="wavemaker_solver_backends_",cleanup=false)
const DURATION = isempty(ARGS) ? 0.1 : parse(Float64,ARGS[1])
const REPEATS = length(ARGS) < 2 ? 4 : parse(Int,ARGS[2])
scope = Module(:WavemakerBackendAudit)
Core.eval(scope,quote
    include(path) = Base.include($scope,path)
    RunSimulation(;kwargs...) = (;kwargs...)
end)
case_path = joinpath(ROOT,"example","Wavemaker","WavemakerCase.jl")
skip_launch(expr) = expr isa Expr && expr.head===:call && first(expr.args)===:run_wavemaker ? nothing : expr
Base.include_string(skip_launch,scope,read(case_path,String),case_path)
setup = Base.invokelatest(getproperty(scope,:run_wavemaker);simulation_time=DURATION,
    visualize=false,open_log_file=false,save_location=joinpath(DIRECTORY,"setup"))
function scheme(images)
    original = setup.SimTimeStepping
    return DualSPHysicsSymplectic(; (name=>getproperty(original,name) for name in fieldnames(typeof(original)) if name != :periodic_images)...,
        periodic_images=images)
end
function run_backend(images,duration,label)
    meta = deepcopy(setup.SimMetaData)
    meta.SimulationTime = duration
    meta.OutputTimes = min(0.1,duration)
    meta.SaveLocation = joinpath(DIRECTORY,label)
    mkpath(meta.SaveLocation)
    particles = AllocateDataStructures(setup.SimGeometry,meta)
    logger = SimulationLogger(meta.SaveLocation)
    try
        return RunSimulation(;setup...,SimMetaData=meta,SimParticles=particles,SimLogger=logger,SimTimeStepping=scheme(images))
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end
function force_audit(frame)
    input = joinpath(ROOT,"output","wavemaker","Wavemaker.vtkhdf")
    isfile(input) || return
    meta = deepcopy(setup.SimMetaData)
    host = AllocateDataStructures(setup.SimGeometry,meta)
    loaded = h5open(input,"r") do file
        root=file["VTKHDF"]
        frame <= length(root["Steps/Values"]) || return false
        offset=Int(root["Steps/PointOffsets"][frame])
        columns=offset+1:offset+length(host)
        host.Position .= SVector{3,Float64}.(eachcol(root["Points"][:,columns]))
        host.Velocity .= SVector{3,Float32}.(eachcol(root["PointData/Velocity"][:,columns]))
        host.Density .= root["PointData/Density"][columns]
        host.ID .= root["PointData/ID"][columns]
        host.GroupMarker .= root["PointData/GroupMarker"][columns]
        host.Type .= ParticleType.(root["PointData/Type"][columns])
        return true
    end
    loaded || return
    states = map((false,true)) do images
        metadata=deepcopy(meta)
        metadata.TimeSteppingMode=scheme(images)
        gpu=CL.upload_particles(host,metadata)
        sup=CL.GPUSupportArrays{3,Float32}(gpu,metadata)
        cl=SPHExampleGPU.GPUCellGrid.CellListWorkspace{3,Float64}(length(gpu);
            reach=metadata.GPUCellSubdivision,periodicity=metadata.Periodicity)
        CL.rebuild_cell_list!(gpu,cl,setup.SimKernel.H⁻¹;support=sup,constants=setup.SimConstants,kernel=setup.SimKernel)
        dt=CL.dual_force_stage!(gpu.Position,gpu.Velocity,gpu.Density,sup.PosCells,gpu,cl,sup,
            setup.SimKernel,setup.SimConstants,setup.SimDensityDiffusion,setup.SimViscosity,metadata,SPHExampleGPU.GPUStepState.HostStep(0.0002,0.0))
        order=sortperm(Array(gpu.ID))
        count=images ? only(Array(sup.dual_state.images.count)) : length(gpu)
        (;dt,count,acceleration=Array(gpu.Acceleration)[order],rate=Array(sup.dρdtI)[order],viscous=Array(sup.dual_state.viscous)[order])
    end
    a,b=states
    relative=norm(b.acceleration-a.acceleration)/max(norm(a.acceleration),eps())
    @assert relative < 1e-5
    @assert abs(b.dt/a.dt-1) < 1e-5
    @assert maximum(abs.(b.rate-a.rate)) < 0.02
    println("FORCE_AUDIT frame=$frame source_count=$(b.count) acceleration_relative_l2=$relative dt_query=$(a.dt) dt_images=$(b.dt) rate_maxdiff=$(maximum(abs.(b.rate-a.rate)))")
    CUDA.reclaim()
end
println("ARTIFACT_DIRECTORY=$DIRECTORY")
force_audit(1)
force_audit(51)
run_backend(false,0.003,"warm_queries")
run_backend(true,0.003,"warm_images")
results=Any[]
snapshots=Dict{Bool,Any}()
for iteration in 1:REPEATS
    images = mod(iteration,4) in (2,3)
    result=run_backend(images,DURATION,"run_$(iteration)_$(images ? "images" : "queries")")
    @assert result.final_state.complete
    @assert length(result.final_state.particles)==463200
    @assert result.diagnostics.frozen_particles==0
    @assert all(isfinite,result.final_state.particles.Density)
    @assert all(x->all(isfinite,x),result.final_state.particles.Position)
    push!(results,(;images,result.performance...))
    particles=result.final_state.particles
    order=sortperm(particles.ID)
    snapshots[images]=(;position=particles.Position[order],velocity=particles.Velocity[order],density=particles.Density[order])
    if length(snapshots)==2
        a,b=snapshots[false],snapshots[true]
        periodic=setup.SimMetaData.Periodicity
        period=periodic.upper-periodic.lower
        displacement=map(b.position-a.position) do delta
            delta+round(delta[periodic.axis]/period)*periodic.translation
        end
        println("STATE_COMPARISON position_minimum_image_maxdiff=$(maximum(norm.(displacement))) position_minimum_image_rms=$(norm(displacement)/sqrt(length(displacement))) velocity_relative_l2=$(norm(b.velocity-a.velocity)/norm(a.velocity)) density_relative_l2=$(norm(b.density-a.density)/norm(a.density))")
    end
    println("WHOLE_SOLVER images=$images performance=$(result.performance) diagnostics=$(result.diagnostics)")
    flush(stdout)
    CUDA.reclaim()
end
open(joinpath(DIRECTORY,"results.txt"),"w") do io
    foreach(row->println(io,row),results)
end
println("BACKEND_AUDIT_COMPLETE")
