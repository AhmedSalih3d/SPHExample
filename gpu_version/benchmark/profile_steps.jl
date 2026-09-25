# Kernel level profile of a few time steps. Usage (from the repository root):
#
#     julia --project=gpu_version gpu_version/benchmark/profile_steps.jl [--float32] <case substring>
#
using SPHExampleGPU
using CUDA
using StaticArrays
using Printf

include(joinpath(@__DIR__, "cases.jl"))

function setup(case::BenchCase, ::Type{T}) where {T}
    save = mktempdir()
    kw   = case.build(T, save)
    particles = AllocateDataStructures(kw.SimGeometry)
    if kw.SimMetaData.FlagMDBCSimple
        _, gp, gn = LoadBoundaryNormals(Val(case.dims), T, kw.ParticleNormalsPath)
        for gi in eachindex(gp)
            particles.GhostPoints[gi]  = gp[gi]
            particles.GhostNormals[gi] = gn[gi]
        end
    end
    Pressure!(particles.Pressure, particles.Density, kw.SimConstants)
    n   = length(particles)
    gpu = upload_particles(particles)
    sup = GPUSupportArrays{case.dims, T}(n)
    red = ReductionWorkspace{SVector{3, T}}(n)
    cl  = CellListWorkspace{case.dims, T}(n)
    mot = MotionArrays(kw.SimGeometry, particles)
    kw.SimMetaData.OutputIterationCounter = 1
    return kw, gpu, sup, red, cl, mot
end

function main(args)
    T = "--float32" in args ? Float32 : Float64
    names = filter(a -> !startswith(a, "--"), args)
    case  = first(select_cases(names))
    kw, gpu, sup, red, cl, mot = setup(case, T)
    meta = kw.SimMetaData
    loop() = SimulationLoop(kw.SimDensityDiffusion, kw.SimViscosity, kw.SimKernel, meta, kw.SimConstants,
                            gpu, cl, sup, red, mot)
    # warm up / compile: one output interval
    loop(); CUDA.synchronize()
    it0 = meta.Iteration
    meta.OutputIterationCounter += 1
    t = @elapsed (loop(); CUDA.synchronize())
    steps = meta.Iteration - it0
    @printf("%s %s: %d particles, %d steps, %.3f ms/step, grid %s, rebuilds %d\n", case.name, T, length(gpu),
            steps, 1e3 * t / steps, cl.grid.dims, cl.nrebuilds)
    k = SPHExampleGPU.GPUKernels
    meta.OutputIterationCounter += 1
    prof = CUDA.@profile loop()
    display(prof)
    println()
end

main(ARGS)
