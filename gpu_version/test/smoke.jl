# Quick development smoke test: run a short 2D and 3D case on the GPU.
using SPHExampleGPU
using CUDA
using StaticArrays
using Printf
using LinearAlgebra

include(joinpath(@__DIR__, "..", "benchmark", "cases.jl"))

function smoke(case::BenchCase, ::Type{T}; simtime = nothing, sync = false) where {T}
    save = mktempdir()
    kw   = case.build(T, save)
    if simtime !== nothing
        kw.SimMetaData.SimulationTime = T(simtime)
        kw.SimMetaData.OutputTimes    = T(simtime / 2)
    end
    kw.SimMetaData.GPUSyncTimers = sync
    particles = AllocateDataStructures(kw.SimGeometry)
    logger    = SimulationLogger(save; to_console = false)
    RunSimulation(; kw..., SimLogger = logger, SimParticles = particles)
    @printf("%s %s: %d particles, %d steps, TotalTime=%.4f, ρ range = [%.3f, %.3f], max|v| = %.4f\n",
            case.name, T, length(particles), kw.SimMetaData.Iteration, kw.SimMetaData.TotalTime,
            minimum(particles.Density), maximum(particles.Density),
            maximum(norm.(particles.Velocity)))
    return particles, kw
end

T = "--float32" in ARGS ? Float32 : Float64
names = filter(a -> !startswith(a, "--"), ARGS)
isempty(names) && (names = ["StillWedge2D_MDBC"])
for c in select_cases(names)
    smoke(c, T; simtime = 0.02, sync = true)
end
