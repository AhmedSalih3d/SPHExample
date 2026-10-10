# Benchmark of the cell list reach: cells of edge H (3^D stencil, the
# default), H/2 (5^D stencil) and optionally H/3 (7^D stencil), following the
# interleaved per kernel protocol of gpu_version/README.md.
#
#     julia --project=gpu_version gpu_version/benchmark/bench_cell_subdivision.jl [--float64] [--rounds N] [--subdiv 1,2,3] <case substrings...>
#
# For every case all variants are set up once (same initial particles), warmed
# up, then measured in `rounds` interleaved rounds (variant 1, 2, ..., 1, 2,
# ...) under `CUDA.@profile`. Reported per variant: device time per step of
# the interaction kernel, of the mDBC kernel, of the cell list rebuild kernels
# and in total, each as the median over rounds of the ratio to variant 1, plus
# the candidate pairs scanned per particle and the fraction that pass the
# distance check (counted on the host from the device cell list).
using SPHExampleGPU
using CUDA
using StaticArrays
using Printf
using Statistics
using LinearAlgebra

include(joinpath(@__DIR__, "cases.jl"))

const G = SPHExampleGPU.GPUCellGrid

function setup(case::BenchCase, ::Type{T}, subdiv::Int; lanes::Int = 0) where {T}
    save = mktempdir()
    kw   = case.build(T, save)
    particles = AllocateDataStructures(kw.SimGeometry, kw.SimMetaData)
    if kw.ParticleNormalsPath !== nothing   # SimpleMDBC cases
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
    cl  = CellListWorkspace{case.dims, T}(n; reach = subdiv)
    mot = MotionArrays(kw.SimGeometry, particles)
    st  = StepState{T}(; dx = one(T) + kw.SimKernel.h)
    kw.SimMetaData.TimeSteppingMode = kw.SimTimeStepping
    kw.SimMetaData.OutputIterationCounter = 1
    kw.SimMetaData.GPUCellSubdivision = subdiv
    kw.SimMetaData.GPULanesPerParticle = lanes
    # many output intervals so that repeated SimulationLoop calls keep stepping
    kw.SimMetaData.SimulationTime = T(1e6)
    return (; kw, gpu, sup, red, cl, mot, st, subdiv)
end

function loop!(v)
    kw = v.kw
    SimulationLoop(kw.SimDensityDiffusion, kw.SimViscosity, kw.SimKernel, kw.SimMetaData, kw.SimConstants,
                   v.gpu, v.cl, v.sup, v.red, v.mot, v.st)
    kw.SimMetaData.OutputIterationCounter += 1
    return nothing
end

# Candidate pairs scanned by the gather kernel (per particle) and the fraction
# inside the support radius, evaluated on the host from the current cell list.
function candidate_stats(v)
    grid      = v.cl.grid
    CellStart = Array(view(v.cl.CellStart, 1:(grid.ncells + 1)))
    CellID    = Array(v.gpu.CellID)
    Position  = Array(v.gpu.Position)
    H²        = v.kw.SimKernel.H²
    n = length(Position)
    scanned = 0; accepted = 0
    Threads.@threads for i in 1:n
        s = 0; a = 0
        xi = Position[i]
        c  = CellID[i]
        for off in G.row_offsets(grid)
            jlo, jhi = G.row_range(grid, CellStart, c, off)
            s += jhi - jlo + 1
            for j in jlo:jhi
                j == i && continue
                d = xi - Position[j]
                a += dot(d, d) <= H²
            end
        end
        Threads.atomic_add!(SCANNED, s - 1)   # own particle excluded
        Threads.atomic_add!(ACCEPTED, a)
    end
    scanned  = SCANNED[];  SCANNED[]  = 0
    accepted = ACCEPTED[]; ACCEPTED[] = 0
    return (scanned = scanned / n, accepted = accepted / n, grid = grid.dims, ncells = grid.ncells)
end
const SCANNED  = Threads.Atomic{Int}(0)
const ACCEPTED = Threads.Atomic{Int}(0)

function kernel_times(prof)
    df = prof.device   # NamedTuple of columns
    t = Dict{String, Float64}()
    for r in eachindex(df.name)
        name = String(df.name[r])
        key = occursin("interaction_kernel", name) ? "interaction" :
              occursin("mdbc_kernel", name)        ? "mdbc" :
              (occursin("cellid_hist", name) || occursin("scatter_kernel", name) ||
               occursin("cell_sort", name) || occursin("gather_kernel", name) ||
               occursin("accumulate", name) || occursin("scan", name) || occursin("reduce_kernel", name)) ? "celllist" :
              "other"
        t[key] = get(t, key, 0.0) + (df.stop[r] - df.start[r])
    end
    t["total"] = sum(values(t))
    return t
end

function getopt(args, name, default, parsefn = x -> parse(Int, x))
    i = findfirst(==(name), args)
    i === nothing && return default, args
    return parsefn(args[i + 1]), [args[1:i-1]; args[i+2:end]]
end

function main(args)
    T = "--float64" in args ? Float64 : Float32
    rounds, args  = getopt(args, "--rounds", 9)
    subdivs, args = getopt(args, "--subdiv", [1, 2], s -> parse.(Int, split(s, ",")))
    lanes, args   = getopt(args, "--lanes", 0)
    names = filter(a -> !startswith(a, "--"), args)
    cases = select_cases(names)
    keys_ = ("interaction", "mdbc", "celllist", "total")

    println("GPU $(CUDA.name(CUDA.device())), $T, $rounds interleaved rounds, subdivisions $(subdivs)")
    for case in cases
        variants = [setup(case, T, s; lanes = lanes) for s in subdivs]
        # warm up / compile
        for _ in 1:3, v in variants
            loop!(v)
        end
        CUDA.synchronize()
        stats = [candidate_stats(v) for v in variants]

        # measured rounds; every SimulationLoop call runs one output interval
        times = [Dict{String, Vector{Float64}}(k => Float64[] for k in keys_) for _ in variants]
        steps = zeros(Int, length(variants))
        for r in 1:rounds, (k, v) in enumerate(variants)
            it0  = v.kw.SimMetaData.Iteration
            prof = CUDA.@profile loop!(v)
            ns   = v.kw.SimMetaData.Iteration - it0
            steps[k] += ns
            kt = kernel_times(prof)
            for key in keys_
                push!(times[k][key], 1e3 * get(kt, key, 0.0) / ns)   # ms per step
            end
        end

        n = length(variants[1].gpu)
        @printf("\n%s: %d particles, steps per round ≈ %d, lanes %s\n", case.name, n, steps[1] ÷ rounds,
                lanes == 0 ? "auto=$(choose_lanes(n, Val(case.dims)))" : string(lanes))
        @printf("%-8s %-14s %9s %8s %8s %9s %9s %9s %9s %9s %9s %9s\n", "subdiv", "grid", "cells", "cand/p", "acc/p",
                "hit %", "inter", "mdbc", "cellist", "total", "ratio_i", "ratio_t")
        for (k, v) in enumerate(variants)
            s = stats[k]
            med(key) = median(times[k][key])
            ratio(key) = median(times[k][key] ./ times[1][key])
            @printf("%-8d %-14s %9d %8.1f %8.1f %8.1f%% %9.4f %9.4f %9.4f %9.4f %9.3f %9.3f\n",
                    v.subdiv, string(s.grid), s.ncells, s.scanned, s.accepted, 100 * s.accepted / s.scanned,
                    med("interaction"), med("mdbc"), med("celllist"), med("total"),
                    ratio("interaction"), ratio("total"))
        end
        println("  (ms of device time per step; ratio_* = median over rounds of the per round ratio to subdiv $(subdivs[1]))")
        flush(stdout)
        variants = nothing
        GC.gc(); CUDA.reclaim()
    end
end

main(ARGS)
