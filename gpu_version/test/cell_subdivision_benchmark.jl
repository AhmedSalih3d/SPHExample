module CellSubdivisionBenchmarkTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra

include(joinpath(@__DIR__, "..", "benchmark", "bench_cell_subdivision.jl"))
const B = CellSubdivisionBenchmark
const G = SPHExampleGPU.GPUCellGrid
const GK = SPHExampleGPU.GPUKernels

function host_fixture(points::Vector{SVector{D, TP}}, ::Type{T}, H, R; moved = points) where {D, T, TP}
    coordinates = [G.cell_coords(p, TP(R) * TP(inv(H))) for p in points]
    origin = ntuple(d -> minimum(c[d] for c in coordinates) - Int32(R), D)
    upper = ntuple(d -> maximum(c[d] for c in coordinates) + Int32(R), D)
    dims = upper .- origin .+ Int32(1)
    grid = CellGrid{D, R}(origin, dims, Int32(prod(dims)))
    cells = [G.linear_cell(grid, c) for c in coordinates]
    order = sortperm(cells)
    histogram = zeros(Int32, grid.ncells)
    for c in cells
        histogram[c] += Int32(1)
    end
    starts = Int32[0; cumsum(histogram)]
    positions = moved[order]
    cell_ids = cells[order]
    pairs = TP === T ? positions : [pos_cell(p, c, grid, cell_size(grid, H)) for (p, c) in zip(positions, cell_ids)]
    return (; pairs, positions, cell_ids, starts, grid, ids = order, coordinates = coordinates[order])
end

function lattice(::Val{D}, ::Type{TP}, H, offset) where {D, TP}
    points = vec([SVector{D, TP}(ntuple(d -> TP((-1)^d) * TP(offset) +
        TP(H) / TP(3.2) * (TP(index[d]) + TP(0.09) * sin(TP(sum(Tuple(index)) + d))), D))
        for index in CartesianIndices(ntuple(_ -> -4:4, D))])
    push!(points, points[1])
    return points
end

function accepted_ids(f, H)
    neighbors = [Int[] for _ in f.ids]
    for i in eachindex(f.pairs)
        B.particle_candidate_counts(i, f.pairs, f.starts, f.cell_ids, f.grid, H, H^2,
            (i, j) -> push!(neighbors[f.ids[i]], f.ids[j]))
    end
    return sort!.(neighbors)
end

@testset "cell subdivision benchmark input and kernel accounting" begin
    cfg = B.configuration(String[])
    @test cfg.T === Float32
    @test cfg.subdivs == [1, 2]
    @test cfg.rounds == 9
    @test cfg.lanes == 0
    cfg = B.configuration(["--float64", "--double-position", "--radial", "--scalar-sps",
        "--rounds", "3", "--subdiv", "1,2,3,4", "--lanes", "1", "DamBreak3D"])
    @test cfg.T === Float64
    @test cfg.double_position && cfg.radial && cfg.scalar_sps
    @test cfg.subdivs == [1, 2, 3, 4]
    @test cfg.names == ["DamBreak3D"]
    for args in (["--unknown"], ["--rounds"], ["--rounds", "0"], ["--subdiv", "0"],
                 ["--subdiv", ""], ["--lanes", "3"], ["--lanes", "64"])
        @test_throws ArgumentError B.configuration(args)
    end
    @test_throws ArgumentError B.main(["no-such-case"])
    profile = (; device = (; name = ["interaction_kernel!", "mdbc_kernel!",
        "bbox_reduce_scalar_kernel!", "finish_grid_metadata_kernel!", "scan_block_kernel!",
        "reduce_kernel!", "final_step_kernel!"], start = zeros(7), stop = collect(1.0:7.0)))
    times = B.kernel_times(profile)
    @test times["interaction"] == 1
    @test times["mdbc"] == 2
    @test times["celllist"] == 3 + 4 + 5
    @test times["other"] == 6 + 7
    @test times["total"] == 28
end

@testset "candidate accounting excludes self, not coincident particles" begin
    for T in (Float32, Float64), R in (1, 2, 3, 4)
        H = T(0.125)
        point = zero(SVector{2, T})
        singleton = host_fixture([point], T, H, R)
        counts = B.candidate_counts(singleton.pairs, singleton.starts, singleton.cell_ids, singleton.grid, H, H^2)
        stats = B.summarize_candidates(counts)
        @test stats.scanned_total == stats.accepted_total == 0
        @test stats.self_skips == 1
        @test ismissing(stats.hit_fraction)
        @test stats.empty_row_fraction == (2R) / (2R + 1)
        coincident = host_fixture([point, point], T, H, R)
        stats = B.summarize_candidates(B.candidate_counts(coincident.pairs, coincident.starts,
            coincident.cell_ids, coincident.grid, H, H^2))
        @test stats.scanned_total == stats.accepted_total == 2
        @test stats.self_skips == 2
        @test stats.hit_fraction == 1
        @test stats.p95_candidates == stats.maximum_candidates == 1
        @test accepted_ids(coincident, H) == [[2], [1]]
        @test_throws ArgumentError B.candidate_counts(SVector{2, T}[], Int32[0], Int32[], singleton.grid, H, H^2)
        @test_throws DimensionMismatch B.candidate_counts(singleton.pairs, singleton.starts, Int32[], singleton.grid, H, H^2)
    end
    @test_throws ArgumentError B.summarize_candidates(B.ParticleCandidateCounts[])
end

@testset "support cutoff is inclusive" begin
    for T in (Float32, Float64), R in (1, 2, 3, 4)
        H = T(0.125)
        points = [SVector{2, T}(x, 0) for x in (zero(T), prevfloat(H), H, nextfloat(H))]
        f = host_fixture(points, T, H, R)
        @test accepted_ids(f, H)[1] == [2, 3]
    end
end

@testset "fine grids preserve exact frozen-state neighbor IDs" begin
    for D in (2, 3), (T, TP) in ((Float32, Float32), (Float64, Float64), (Float32, Float64)), offset in (0, 10_000)
        H = T(0.125)
        points = lattice(Val(D), TP, H, offset)
        expected = [[j for j in eachindex(points) if j != i && sum(abs2, points[i] - points[j]) <= H^2]
            for i in eachindex(points)]
        summaries = []
        for R in (1, 2, 3, 4)
            f = host_fixture(points, T, H, R)
            @test accepted_ids(f, H) == expected
            counts = B.candidate_counts(f.pairs, f.starts, f.cell_ids, f.grid, H, H^2)
            @test counts == B.candidate_counts(f.pairs, f.starts, f.cell_ids, f.grid, H, H^2)
            stats = B.summarize_candidates(counts)
            @test stats.accepted_total == sum(length, expected)
            @test stats.self_skips == length(points)
            @test stats.rejected_total == stats.scanned_total - stats.accepted_total >= 0
            push!(summaries, stats)
        end
        if D == 3 && T === Float64 && iszero(offset)
            @test summaries[2].scanned <= 0.8 * summaries[1].scanned
            @test summaries[4].scanned <= 0.75 * summaries[1].scanned
        end
    end
end

@testset "retained cell assignments remain the candidate-search authority" begin
    for D in (2, 3), (T, TP) in ((Float32, Float32), (Float64, Float64), (Float32, Float64)), R in (1, 2, 4)
        H = T(0.125)
        points = lattice(Val(D), TP, H, 10_000)
        moved = [p + SVector{D, TP}(ntuple(d -> d == 1 ? TP(isodd(i) ? H : -H) : zero(TP), D))
            for (i, p) in enumerate(points)]
        f = host_fixture(points, T, H, R; moved)
        expected = [sort([f.ids[j] for j in eachindex(f.pairs) if j != i &&
            all(abs.(f.coordinates[i] .- f.coordinates[j]) .<= R) &&
            sum(abs2, f.positions[i] - f.positions[j]) <= H^2]) for i in eachindex(f.pairs)]
        @test accepted_ids(f, H)[f.ids] == expected
    end
end

function device_counts_kernel!(out, pairs, starts, cells, grid, H, H²)
    i = G.thread_index()
    if i <= length(pairs)
        @inbounds out[i] = B.particle_candidate_counts(i, pairs, starts, cells, grid, H, H²)
    end
    return nothing
end

function tiny_case(D)
    B.BenchCase("Traversal$(D)D", D, (T, save) -> begin
        H = T(0.125)
        points = lattice(Val(D), T, H, 0)[1:end-1]
        constants = SimulationConstants{T}(dx = H / T(3.2), c₀ = T(30), m₀ = T(1000) * (H / T(3.2))^D)
        geometry = [SPHGeometry{D, T}(points; Density = T[1000 + 0.1 * sin(i) for i in eachindex(points)],
            Type = Fluid, GroupMarker = 1)]
        meta = SimulationMetaData{D, T}(SimulationName = "Traversal", SaveLocation = save,
            SimulationTime = T(0.01), OutputTimes = T(0.001), VisualizeInParaview = false, OpenLogFile = false)
        (; SimGeometry = geometry, SimMetaData = meta, SimConstants = constants,
            SimKernel = SPHKernelInstance{D, T}(WendlandC2(); h = H / 2),
            SimViscosity = LaminarSPS(), SimDensityDiffusion = ZeroDensityDiffusion(),
            SimTimeStepping = SymplecticTimeStepping(), ParticleNormalsPath = nothing)
    end)
end

if CUDA.functional()
    @testset "candidate accounting agrees on host and device" begin
        for D in (2, 3), (T, TP) in ((Float32, Float32), (Float64, Float64), (Float32, Float64)), R in (1, 2, 4)
            H = T(0.125)
            f = host_fixture(lattice(Val(D), TP, H, 10_000), T, H, R)
            expected = B.candidate_counts(f.pairs, f.starts, f.cell_ids, f.grid, H, H^2)
            out = CuVector{B.ParticleCandidateCounts}(undef, length(f.pairs))
            @cuda threads=128 blocks=cld(length(f.pairs), 128) device_counts_kernel!(
                out, CuArray(f.pairs), CuArray(f.starts), CuArray(f.cell_ids), f.grid, H, H^2)
            @test Array(out) == expected
        end
    end

    @testset "benchmark precision, arithmetic options, and current snapshot" begin
        for D in (2, 3)
            v = B.setup(tiny_case(D), Float32, 4; lanes = 1, double_position = true, radial = true, scalar_sps = true)
            @test eltype(v.gpu.Position) == SVector{D, Float64}
            @test uses_pos_cells(v.sup)
            @test v.kw.SimMetaData.GPURadialInteractions
            @test v.kw.SimMetaData.GPUScalarSPSViscosity
            B.loop!(v)
            stale = Array(v.sup.PosCells)
            positions = Array(v.gpu.Position)
            for i in eachindex(positions)
                isodd(i) && (positions[i] += SVector{D, Float64}(ntuple(d -> d == 1 ? 0.1 : 0.0, D)))
            end
            copyto!(v.gpu.Position, positions)
            snapshot = B.candidate_stats(v)
            GK.launch_pos_cells!(v.sup.PosCells, v.gpu.Position, v.gpu.CellID, v.cl.grid_dev,
                HostStep(0f0, 0f0), v.kw.SimKernel)
            pairs = Array(v.sup.PosCells)
            starts = Array(view(v.cl.CellStart, 1:(v.cl.grid.ncells + 1)))
            cells = Array(v.gpu.CellID)
            H, H² = v.kw.SimKernel.H, v.kw.SimKernel.H²
            fresh_counts = B.candidate_counts(pairs, starts, cells, v.cl.grid, H, H²)
            stale_counts = B.candidate_counts(stale, starts, cells, v.cl.grid, H, H²)
            @test snapshot.accepted_total == B.summarize_candidates(fresh_counts).accepted_total
            @test snapshot.accepted_total != B.summarize_candidates(stale_counts).accepted_total
        end
    end

    @testset "fine-grid short trajectories with orientation-independent diffusion" begin
        for D in (2, 3)
            states = map((1, 2, 4)) do R
                v = B.setup(tiny_case(D), Float64, R; lanes = 1)
                B.loop!(v)
                order = sortperm(Array(v.gpu.ID))
                fields = map(f -> Array(getproperty(v.gpu, f))[order], (:Position, :Velocity, :Density, :Pressure, :Acceleration))
                (; fields, steps = v.kw.SimMetaData.Iteration, time = v.kw.SimMetaData.TotalTime)
            end
            for state in states[2:end]
                @test state.steps == states[1].steps > 1
                @test state.time == states[1].time
                for (actual, expected) in zip(state.fields, states[1].fields)
                    @test isapprox(actual, expected; rtol = 1e-10, atol = 1e-10)
                end
            end
        end
    end
else
    @testset "CUDA traversal checks require a functional GPU" begin
        @test_skip CUDA.functional()
    end
end

end
