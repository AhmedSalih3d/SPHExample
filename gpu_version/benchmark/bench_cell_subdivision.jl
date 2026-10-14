# Benchmark of the cell list reach: cells of edge H (3^D stencil, the
# default), H/2 (5^D stencil) and optionally H/3 (7^D stencil), following the
# interleaved per kernel protocol of gpu_version/README.md.
#
#     julia --project=gpu_version gpu_version/benchmark/bench_cell_subdivision.jl [--float64] [--double-position] [--radial] [--scalar-sps] [--rounds N] [--subdiv 1,2,4] [--lanes K] [--output results.csv] <case substrings...>
#
# For every case all variants are set up once (same initial particles), warmed
# up, then measured in `rounds` interleaved rounds with a rotating start.
# Each round measures an unprofiled output interval and a separate interval
# under `CUDA.@profile`. Reported per variant: wall time per step and device time of
# the interaction kernel, of the mDBC kernel, of the cell list rebuild kernels
# and in total, each as the median over rounds of the ratio to variant 1, plus
# the candidate pairs scanned per particle and the fraction that pass the
# distance check (a host snapshot after warmup, not a hardware bandwidth metric).
module CellSubdivisionBenchmark

using SPHExampleGPU
using CUDA
using StaticArrays
using Printf
using Statistics
using LinearAlgebra
using CSV

include(joinpath(@__DIR__, "cases.jl"))

const G = SPHExampleGPU.GPUCellGrid

function setup(case::BenchCase, ::Type{T}, subdiv::Int; lanes::Int = 0,
               double_position::Bool = false, radial::Bool = false, scalar_sps::Bool = false) where {T}
    save = mktempdir()
    kw   = case.build(T, save)
    meta = kw.SimMetaData
    meta.GPUCellSubdivision = subdiv
    meta.GPULanesPerParticle = lanes
    meta.GPUDoublePosition = double_position
    meta.GPURadialInteractions = radial
    meta.GPUScalarSPSViscosity = scalar_sps
    TP = position_float_type(meta)
    particles = AllocateDataStructures(kw.SimGeometry, kw.SimMetaData)
    if kw.ParticleNormalsPath !== nothing   # SimpleMDBC cases
        _, gp, gn = LoadBoundaryNormals(Val(case.dims), TP, kw.ParticleNormalsPath)
        for gi in eachindex(gp)
            particles.GhostPoints[gi]  = gp[gi]
            particles.GhostNormals[gi] = gn[gi]
        end
    end
    Pressure!(particles.Pressure, particles.Density, kw.SimConstants)
    n   = length(particles)
    gpu = upload_particles(particles, meta)
    sup = GPUSupportArrays{case.dims, T}(n, meta)
    red = ReductionWorkspace{SVector{3, T}}(n)
    cl  = CellListWorkspace{case.dims, TP}(n; reach = subdiv)
    mot = MotionArrays(kw.SimGeometry, particles)
    st  = StepState{T}(; dx = one(T) + kw.SimKernel.h)
    kw.SimMetaData.TimeSteppingMode = kw.SimTimeStepping
    kw.SimMetaData.OutputIterationCounter = 1
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

const ParticleCandidateCounts = @NamedTuple{scanned::Int, accepted::Int, rows::Int, empty_rows::Int, self_skips::Int}

function particle_candidate_counts(i, pairs, CellStart, CellID, grid, H, H²,
                                   onaccepted::F = Returns(nothing)) where {F}
    scanned = 0
    accepted = 0
    empty_rows = 0
    self_skips = 0
    c = CellID[i]
    s = cell_size(grid, H)
    for row in G.cell_rows(grid)
        jlo, jhi = G.row_range(grid, CellStart, c, row)
        empty_rows += jlo > jhi
        for j in jlo:jhi
            if j == i
                self_skips += 1
                continue
            end
            scanned += 1
            d = pair_vector(pairs, pairs[i], j, c + row.off, row, s)
            if dot(d, d) <= H²
                accepted += 1
                onaccepted(i, j)
            end
        end
    end
    return (; scanned, accepted, rows = length(G.cell_rows(grid)), empty_rows, self_skips)
end

function candidate_counts(pairs, CellStart, CellID, grid, H, H²)
    n = length(pairs)
    n > 0 || throw(ArgumentError("candidate statistics require at least one particle"))
    length(CellID) == n || throw(DimensionMismatch("pair sources and cell IDs must have the same length"))
    counts = Vector{ParticleCandidateCounts}(undef, n)
    Threads.@threads for i in eachindex(pairs)
        counts[i] = particle_candidate_counts(i, pairs, CellStart, CellID, grid, H, H²)
    end
    return counts
end

function summarize_candidates(counts)
    isempty(counts) && throw(ArgumentError("candidate statistics require at least one particle"))
    n = length(counts)
    scanned_total = sum(c -> c.scanned, counts)
    accepted_total = sum(c -> c.accepted, counts)
    hit_fraction = iszero(scanned_total) ? missing : accepted_total / scanned_total
    return (; scanned = scanned_total / n, accepted = accepted_total / n,
        scanned_total, accepted_total, rejected_total = scanned_total - accepted_total,
        hit_fraction, p95_candidates = quantile([c.scanned for c in counts], 0.95),
        maximum_candidates = maximum(c -> c.scanned, counts),
        empty_row_fraction = sum(c -> c.empty_rows, counts) / sum(c -> c.rows, counts),
        self_skips = sum(c -> c.self_skips, counts))
end

function candidate_stats(v)
    grid = v.cl.grid
    CellStart = Array(view(v.cl.CellStart, 1:(grid.ncells + 1)))
    CellID = Array(v.gpu.CellID)
    Position = Array(v.gpu.Position)
    kernel = v.kw.SimKernel
    # Start-of-step PosCells are stale after the final integration. Reconstruct
    # this snapshot from the current positions and the retained cell assignments.
    pairs = uses_pos_cells(v.sup) ?
        [pos_cell(x, c, grid, cell_size(grid, kernel.H)) for (x, c) in zip(Position, CellID)] : Position
    counts = candidate_counts(pairs, CellStart, CellID, grid, kernel.H, kernel.H²)
    return merge(summarize_candidates(counts), (; grid = grid.dims, ncells = grid.ncells))
end

const CELL_LIST_KERNELS = ("bbox_reduce_scalar_kernel", "finish_grid_metadata_kernel",
    "cellid_hist_kernel", "scatter_kernel", "cell_sort_kernel", "gather_kernel",
    "scan_block_kernel", "scan_add_offsets_kernel", "nonzero_flag_kernel", "compact_kernel")

function kernel_times(prof)
    df = prof.device   # NamedTuple of columns
    t = Dict{String, Float64}()
    for r in eachindex(df.name)
        name = String(df.name[r])
        key = occursin("interaction_kernel", name) ? "interaction" :
              occursin("mdbc_kernel", name)        ? "mdbc" :
              any(k -> occursin(k, name), CELL_LIST_KERNELS) ? "celllist" :
              "other"
        t[key] = get(t, key, 0.0) + (df.stop[r] - df.start[r])
    end
    t["total"] = sum(values(t))
    return t
end

function configuration(args)
    T = Float32
    rounds = 9
    subdivs = [1, 2]
    lanes = 0
    double_position = false
    radial = false
    scalar_sps = false
    output = nothing
    names = String[]
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--float64"
            T = Float64
        elseif arg == "--float32"
            T = Float32
        elseif arg == "--double-position"
            double_position = true
        elseif arg == "--radial"
            radial = true
        elseif arg == "--scalar-sps"
            scalar_sps = true
        elseif arg in ("--rounds", "--subdiv", "--lanes", "--output")
            i < length(args) || throw(ArgumentError("missing value after $arg"))
            i += 1
            value = args[i]
            if arg == "--rounds"
                rounds = parse(Int, value)
            elseif arg == "--subdiv"
                subdivs = parse.(Int, split(value, ","))
            elseif arg == "--lanes"
                lanes = parse(Int, value)
            else
                output = abspath(value)
            end
        elseif startswith(arg, "--")
            throw(ArgumentError("unknown option $arg"))
        else
            push!(names, arg)
        end
        i += 1
    end
    rounds > 0 || throw(ArgumentError("--rounds must be positive"))
    !isempty(subdivs) && all(>=(1), subdivs) || throw(ArgumentError("--subdiv must contain positive integers"))
    lanes == 0 || (ispow2(lanes) && 1 <= lanes <= 32) ||
        throw(ArgumentError("--lanes must be 0 (automatic) or a power of two from 1 to 32"))
    return (; T, rounds, subdivs, lanes, double_position, radial, scalar_sps, output, names)
end

function main(args)
    cfg = configuration(args)
    (; T, rounds, subdivs, lanes, names) = cfg
    cases = select_cases(names)
    isempty(cases) && throw(ArgumentError("no benchmark cases match $(join(names, ", "))"))
    keys_ = ("interaction", "mdbc", "celllist", "other", "total", "wall")
    records = NamedTuple[]

    println("GPU $(CUDA.name(CUDA.device())), $T, $rounds interleaved rounds, subdivisions $(subdivs)")
    println("Double positions: $(cfg.double_position); radial: $(cfg.radial); scalar SPS: $(cfg.scalar_sps)")
    println("Counts exclude self and test distance only; they do not establish a bandwidth bottleneck.")
    println("Different grids can change reference diffusion orientation and floating-point accumulation.")
    for case in cases
        variants = [setup(case, T, s; lanes, double_position = cfg.double_position,
            radial = cfg.radial, scalar_sps = cfg.scalar_sps) for s in subdivs]
        # warm up / compile
        for _ in 1:3, v in variants
            loop!(v)
        end
        CUDA.synchronize()
        stats = [candidate_stats(v) for v in variants]

        # measured rounds; every SimulationLoop call runs one output interval
        times = [Dict{String, Vector{Float64}}(k => Float64[] for k in keys_) for _ in variants]
        steps = zeros(Int, length(variants))
        for r in 1:rounds, offset in 0:(length(variants) - 1)
            k = mod1(r + offset, length(variants))
            v = variants[k]
            wall_it0 = v.kw.SimMetaData.Iteration
            wall_seconds = @elapsed begin
                loop!(v)
                CUDA.synchronize()
            end
            wall_steps = v.kw.SimMetaData.Iteration - wall_it0
            wall_steps > 0 || error("the wall-time interval executed no steps for $(case.name)")
            wall_ms = 1e3 * wall_seconds / wall_steps
            push!(times[k]["wall"], wall_ms)
            it0  = v.kw.SimMetaData.Iteration
            prof = CUDA.@profile loop!(v)
            ns   = v.kw.SimMetaData.Iteration - it0
            ns > 0 || error("the profiled interval executed no steps for $(case.name)")
            steps[k] += ns
            kt = kernel_times(prof)
            for key in keys_
                key == "wall" && continue
                push!(times[k][key], 1e3 * get(kt, key, 0.0) / ns)   # ms per step
            end
            s = stats[k]
            push!(records, (; case = case.name, precision = string(T),
                position_precision = string(position_float_type(v.kw.SimMetaData)),
                radial = cfg.radial, scalar_sps = cfg.scalar_sps, subdivision = v.subdiv,
                lanes = lanes == 0 ? choose_lanes(length(v.gpu), Val(case.dims)) : lanes,
                round = r, wall_steps, profiled_steps = ns, wall_ms,
                interaction_ms = times[k]["interaction"][end], mdbc_ms = times[k]["mdbc"][end],
                celllist_ms = times[k]["celllist"][end], other_ms = times[k]["other"][end],
                device_ms = times[k]["total"][end], candidates = s.scanned, accepted = s.accepted,
                p95_candidates = s.p95_candidates, hit_fraction = s.hit_fraction,
                empty_row_fraction = s.empty_row_fraction, cells = s.ncells))
        end

        n = length(variants[1].gpu)
        @printf("\n%s: %d particles, steps per round ≈ %d, lanes %s\n", case.name, n, steps[1] ÷ rounds,
                lanes == 0 ? "auto=$(choose_lanes(n, Val(case.dims)))" : string(lanes))
        @printf("%-8s %-14s %9s %8s %8s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n", "subdiv", "grid", "cells", "cand/p", "acc/p",
                "hit %", "inter", "mdbc", "cellist", "total", "ratio_i", "ratio_t", "wall", "ratio_w")
        for (k, v) in enumerate(variants)
            s = stats[k]
            med(key) = median(times[k][key])
            ratio(key) = median(times[k][key] ./ times[1][key])
            hit = ismissing(s.hit_fraction) ? "n/a" : @sprintf("%.1f%%", 100 * s.hit_fraction)
            @printf("%-8d %-14s %9d %8.1f %8.1f %9s %9.4f %9.4f %9.4f %9.4f %9.3f %9.3f %9.4f %9.3f\n",
                    v.subdiv, string(s.grid), s.ncells, s.scanned, s.accepted, hit,
                    med("interaction"), med("mdbc"), med("celllist"), med("total"),
                    ratio("interaction"), ratio("total"), med("wall"), ratio("wall"))
            wall_ratios = times[k]["wall"] ./ times[1]["wall"]
            @printf("  p95 candidates %.1f; max %d; empty rows %.1f%%; wall ratio IQR [%.3f, %.3f]\n",
                s.p95_candidates, s.maximum_candidates, 100 * s.empty_row_fraction,
                quantile(wall_ratios, 0.25), quantile(wall_ratios, 0.75))
        end
        println("  (ms of device time per step; ratio_* = median over rounds of the per round ratio to subdiv $(subdivs[1]))")
        println("  (wall = unprofiled loop ms/step including rebuilds and host orchestration, excluding output; counts = post-warmup snapshot)")
        if cfg.output !== nothing
            mkpath(dirname(cfg.output))
            CSV.write(cfg.output, records)
        end
        flush(stdout)
        variants = nothing
        GC.gc(); CUDA.reclaim()
    end
    return records
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    CellSubdivisionBenchmark.main(ARGS)
end
