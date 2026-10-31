# Paired, unprofiled whole-driver comparison. Initialization/JIT and output
# are reported separately; solver timings synchronize at frame boundaries.
# julia --project=gpu_version -t 2,0 gpu_version/benchmark/variable_resolution.jl [dx] [duration] [pairs]
using SPHExampleGPU, CUDA, Statistics, Printf
include(joinpath(@__DIR__,"..","example","VResolutionDamBreak2d","Setup.jl"))

dx = length(ARGS) >= 1 ? parse(Float64,ARGS[1]) : 0.01
duration = length(ARGS) >= 2 ? parse(Float64,ARGS[2]) : 0.02
pairs = length(ARGS) >= 3 ? parse(Int,ARGS[3]) : 3
pairs > 0 || error("pairs must be positive")

function run_variant(execution)
    directory = mktempdir()
    result = redirect_stdout(devnull) do
        run_vresolution_dambreak(; dx,duration,output_interval=0.0025,save_location=directory,execution)
    end
    snapshots = map((result.coarse,result.fine)) do d
        p = d.particles
        ids,types = Array(p.ID),Array(p.Type)
        keep = findall(!=(Frozen),types)
        keep = keep[sortperm(ids[keep])]
        (; ids=ids[keep],position=Array(p.Position)[keep],density=Array(p.Density)[keep],
            velocity=Array(p.Velocity)[keep],events=Array(d.events))
    end
    return (; solver=result.timings.solver_seconds,total=result.timings.total_wall_seconds,
        initialization=result.timings.initialization_seconds,output=result.timings.output_seconds,
        steps=result.iterations,frames=length(result.times),snapshots,
        rebuilds=map(d -> (d.cells.nrebuilds,d.predictor.cells.nrebuilds),(result.coarse,result.fine)),
        readbacks=result.configuration.control_readbacks,path=result.output)
end

println("GPU: ",CUDA.name(CUDA.device()),"; Julia ",VERSION,"; CUDA.jl ",pkgversion(CUDA))
println("dx=$dx; duration=$duration; output_interval=0.0025; Float32 physics / Float64 positions and MLS")
println("Warming both variants (excluded from paired results)...")
run_variant(:host)
GC.gc(true); CUDA.reclaim()
run_variant(:gpu)
GC.gc(true); CUDA.reclaim()
ratios = Float64[]
for pair in 1:pairs
    order = isodd(pair) ? (:host,:gpu) : (:gpu,:host)
    results = Dict{Symbol,Any}()
    for variant in order
        results[variant] = run_variant(variant)
        GC.gc(true); CUDA.reclaim()
    end
    a,b = results[:host],results[:gpu]
    @assert a.steps == b.steps
    @assert a.frames == b.frames
    @assert a.rebuilds == b.rebuilds
    for (x,y) in zip(a.snapshots,b.snapshots)
        @assert x.events == y.events
        @assert x.ids == y.ids
        @assert isapprox(x.position,y.position;rtol=1e-8,atol=1e-9)
        @assert isapprox(x.velocity,y.velocity;rtol=1e-5,atol=1e-6)
        @assert isapprox(x.density,y.density;rtol=1e-6)
    end
    push!(ratios,a.solver/b.solver)
    @printf("pair %d: host %.6f s, GPU %.6f s; %.3fx; steps=%d, frames=%d, GPU control readbacks=%d\n",
        pair,a.solver,b.solver,last(ratios),a.steps,a.frames,b.readbacks)
    println("  rebuilds accepted/predictor: ",a.rebuilds)
    @printf("  total host/GPU %.6f/%.6f s; init %.6f/%.6f s; output %.6f/%.6f s\n",
        a.total,b.total,a.initialization,b.initialization,a.output,b.output)
    println("  logs: ",a.path," ; ",b.path)
end
@printf("Median paired solver speedup: %.3fx (range %.3f–%.3f)\n",median(ratios),minimum(ratios),maximum(ratios))
