module CellSubdivisionBenchmarkTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra

include(joinpath(@__DIR__, "..", "benchmark", "bench_cell_subdivision.jl"))
const B = CellSubdivisionBenchmark
const G = SPHExampleGPU.GPUCellGrid
const GK = SPHExampleGPU.GPUKernels

function host_fixture(points::Vector{SVector{D, TP}}, ::Type{T}, H, R; Rx = R, moved = points) where {D, T, TP}
    grid0 = CellGrid{D, R, Rx}()
    inv_cell = G.bin_scale(grid0, TP(inv(H)))
    coordinates = [G.cell_coords(p, inv_cell) for p in points]
    origin = ntuple(d -> minimum(c[d] for c in coordinates) - G.axis_reach(grid0, d), D)
    upper = ntuple(d -> maximum(c[d] for c in coordinates) + G.axis_reach(grid0, d), D)
    dims = upper .- origin .+ Int32(1)
    grid = CellGrid{D, R, Rx}(origin, dims, Int32(prod(dims)))
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
        # Anisotropic x-row subdivision (R = 1, Rx in (1, 2, 4)) keeps 3^(D-1) rows
        # and retains 100% of accepted pairs while monotonically reducing candidates.
        aniso_summaries = []
        for Rx in (1, 2, 4)
            f = host_fixture(points, T, H, 1; Rx)
            @test f.grid isa CellGrid{D, 1, Rx}
            @test f.grid isa CellGrid{D, 1}
            @test G.reach(f.grid) == 1 && G.x_reach(f.grid) == Rx
            @test length(G.cell_rows(f.grid)) == 3^(D - 1)
            @test accepted_ids(f, H) == expected
            counts = B.candidate_counts(f.pairs, f.starts, f.cell_ids, f.grid, H, H^2)
            stats = B.summarize_candidates(counts)
            @test stats.accepted_total == sum(length, expected)
            @test stats.self_skips == length(points)
            @test stats.rejected_total == stats.scanned_total - stats.accepted_total >= 0
            push!(aniso_summaries, stats)
        end
        @test aniso_summaries[3].scanned_total < aniso_summaries[2].scanned_total < aniso_summaries[1].scanned_total
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
        for D in (2, 3), (T, TP) in ((Float32, Float32), (Float64, Float64), (Float32, Float64)), (R, Rx) in ((1, 1), (2, 2), (4, 4), (1, 2), (1, 4))
            H = T(0.125)
            f = host_fixture(lattice(Val(D), TP, H, 10_000), T, H, R; Rx)
            expected = B.candidate_counts(f.pairs, f.starts, f.cell_ids, f.grid, H, H^2)
            out = CuVector{B.ParticleCandidateCounts}(undef, length(f.pairs))
            @cuda threads=128 blocks=cld(length(f.pairs), 128) device_counts_kernel!(
                out, CuArray(f.pairs), CuArray(f.starts), CuArray(f.cell_ids), f.grid, H, H^2)
            @test Array(out) == expected
        end
    end

    @testset "anisotropic PosCell accuracy at 10,000 m offset and 128-bit load" begin
        H = 0.04f0
        for (D, R, Rx) in ((2, 1, 2), (2, 1, 4), (3, 1, 2), (3, 1, 4), (3, 2, 4))
            offset = SVector{D, Float64}(ntuple(d -> (-1.0)^d * 1e4 + 3.0 * d, D))
            grid0 = CellGrid{D, R, Rx}()
            inv_cell = G.bin_scale(grid0, 1.0 / Float64(H))
            c0 = G.cell_coords(offset, inv_cell)
            grid = CellGrid{D, R, Rx}(c0 .- Int32(40), ntuple(_ -> Int32(80), D), Int32(80^D))
            s = cell_size(grid, H)
            @test s == SVector{D, Float32}(ntuple(d -> H / (d == 1 ? Float32(Rx) : Float32(R)), D))
            n1, n2 = grid.dims[1], grid.dims[2]
            worst_pc = 0.0
            worst_naive = 0.0
            tested = 0
            for _ in 1:1000
                xi = offset + 0.08 .* randn(SVector{D, Float64})
                xj = xi + 0.02 .* randn(SVector{D, Float64})
                gi = G.cell_coords(xi, inv_cell)
                gj = G.cell_coords(xj, inv_cell)
                margin = ntuple(d -> G.axis_reach(grid, d), D)
                (G.in_grid(grid, gi .- margin) && G.in_grid(grid, gi .+ margin) &&
                 G.in_grid(grid, gj .- margin) && G.in_grid(grid, gj .+ margin)) || continue
                ci = G.linear_cell(grid, gi)
                cj = G.linear_cell(grid, gj)
                d = G.local_coords(grid, cj) .- G.local_coords(grid, ci)
                (abs(d[1]) <= Rx && all(k -> abs(d[k]) <= R, 2:D)) || continue
                dy = Int32(d[2])
                dz = D == 3 ? Int32(d[3]) : Int32(0)
                row = CellRow(dy * n1 + dz * n1 * n2, dy, dz)
                pcs = [pos_cell(xi, ci, grid, s), pos_cell(xj, cj, grid, s)]
                @test all(k -> abs(pcs[1].rel[k]) <= s[k], 1:D)
                v = pair_vector(pcs, pcs[1], Int32(2), ci + row.off, row, s)
                exact = xi - xj
                naive = SVector{D, Float32}(xi) - SVector{D, Float32}(xj)
                worst_pc = max(worst_pc, norm(v - exact))
                worst_naive = max(worst_naive, norm(naive - exact))
                tested += 1
            end
            @test tested > 200
            @test worst_pc < 1e-5 * H
            @test worst_naive > 100 * worst_pc
            cg = G.cell_coords(offset, inv_cell)
            pg = pos_cell(offset, cg, s)
            @test pg.cell == 0 && all(k -> abs(pg.rel[k]) <= s[k], 1:D)
        end

        row3 = CellRow(Int32(-80), Int32(-1), Int32(0))
        s3 = cell_size(CellGrid{3, 1, 4}(), H)
        host_pcs = [
            PosCell{3, Float32}(SVector{3, Float32}(0.0f0, 0.0f0, 0.0f0), Int32(1)),
            PosCell{3, Float32}(SVector{3, Float32}(0.0025f0, -0.0075f0, 0.019f0), Int32(105)),
            PosCell{3, Float32}(SVector{3, Float32}(-0.004f0, 0.011f0, -0.004f0), Int32(28)),
        ]
        dev_pcs = CuVector(host_pcs)
        out_vec = CUDA.zeros(SVector{3, Float32}, 1)
        function aniso_pos_cell_vec_probe!(out, pcs, row, s)
            p_i = G.load_pos_cell(pcs, Int32(2))
            @inbounds out[1] = pair_vector(pcs, p_i, Int32(3), p_i.cell + row.off, row, s)
            return nothing
        end
        ptx_io = IOBuffer()
        CUDA.@device_code_ptx io=ptx_io @cuda threads=1 aniso_pos_cell_vec_probe!(out_vec, dev_pcs, row3, s3)
        @test occursin("ld.global.v4.b32", String(take!(ptx_io)))
        @cuda threads=1 aniso_pos_cell_vec_probe!(out_vec, dev_pcs, row3, s3)
        @test isequal(only(Array(out_vec)), pair_vector(host_pcs, host_pcs[2], Int32(3), host_pcs[2].cell + row3.off, row3, s3))
    end

    @testset "SimpleMDBC and UpdatedMDBC ghost interpolation across Rx = 1, 2, 4" begin
        for D in (2, 3), double_position in (false, true), boundary_mode in (SimpleMDBC, UpdatedMDBC)
            T = Float64
            TP = Float64
            dx = 0.02
            offset = SVector{D, TP}(ntuple(d -> d == 1 ? 10_000.0 : -5_000.0 * d, D))
            constants = SimulationConstants{T}(dx = dx, c₀ = 30.0, g = 9.81, ρ₀ = 1000.0)
            kernel = SPHKernelInstance{D, T}(WendlandC2(); dx = dx)
            wall_pts = [offset + SVector{D, TP}(ntuple(d -> d == D ? -l * dx : k * dx, D))
                        for l in 0:2 for k in -3:3]
            fluid_pts = [offset + SVector{D, TP}(ntuple(d -> d == D ? l * dx : k * dx, D))
                         for l in 1:4 for k in -3:3]
            normals = [SVector{D, T}(ntuple(d -> d == D ? (0.5 * dx - (p[D] - offset[D])) : 0.0, D)) for p in wall_pts]
            ghosts = [wall_pts[i] + SVector{D, TP}(normals[i]) for i in eachindex(wall_pts)]
            fluid_rho = [1000.0 + 5.0 * (4 - (p[D] - offset[D]) / dx) + 0.3 * ((p[1] - offset[1]) / dx) for p in fluid_pts]
            fluid_vel = [SVector{D, T}(ntuple(d -> d == 1 ? 0.4 * (p[D] - offset[D]) : -0.1 * d, D)) for p in fluid_pts]
            geom = [
                SPHGeometry{D, T}(Particles = particle_struct_array(wall_pts, constants.ρ₀;
                    GhostPoints = ghosts, GhostNormals = normals), GroupMarker = 1, Type = Fixed),
                SPHGeometry{D, T}(Particles = particle_struct_array(fluid_pts, constants.ρ₀;
                    Density = fluid_rho, Velocity = fluid_vel), GroupMarker = 2, Type = Fluid),
            ]
            results = map((1, 2, 4)) do Rx
                meta = SimulationMetaData{D, T, NoShifting, NoKernelOutput, boundary_mode, NoLog}(
                    SimulationName = "mdbc_rx", SaveLocation = tempdir(),
                    GPUCellSubdivision = 1, GPUXCellSubdivision = Rx,
                    GPUDoublePosition = double_position, VisualizeInParaview = false, OpenLogFile = false)
                host = AllocateDataStructures(geom, meta)
                gpu = upload_particles(host, meta)
                sup = GPUSupportArrays{D, T}(gpu, meta; kernel)
                cl = CellListWorkspace{D, TP}(length(gpu); reach = 1, x_reach = Rx)
                SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu, cl, kernel.H⁻¹; support = sup, kernel)
                step = HostStep(0.001, 0.0)
                if boundary_mode === UpdatedMDBC
                    GK.launch_updated_mdbc!(gpu.Density, gpu.Position, gpu.Velocity,
                        gpu.GhostPoints, gpu.GhostNormals, gpu.GhostIndex, gpu.Type,
                        cl.CellStart, cl.grid_dev, step, kernel, constants, sup.boundary_data;
                        pos_cells = sup.PosCells, compact_ghosts = Val(gpu.BoundarySlot !== nothing))
                else
                    GK.launch_mdbc!(gpu.Density, gpu.Position, gpu.GhostPoints, gpu.GhostIndex,
                        gpu.Type, cl.CellStart, cl.grid_dev, step, kernel, constants;
                        pos_cells = sup.PosCells, ghost_normals = gpu.GhostNormals,
                        compact_ghosts = Val(gpu.BoundarySlot !== nothing))
                end
                order = sortperm(Array(gpu.ID))
                rho = Array(gpu.Density)[order]
                if boundary_mode === UpdatedMDBC
                    owners = Array(gpu.GhostIndex)
                    owner_ids = Array(gpu.ID)[owners]
                    owner_perm = sortperm(owner_ids)
                    active = Array(sup.boundary_data.active)[owner_perm]
                    vel = Array(sup.boundary_data.velocity)[owner_perm]
                    tvel = Array(sup.boundary_data.tangent_velocity)[owner_perm]
                    (; rho, active, vel, tvel)
                else
                    (; rho)
                end
            end
            for r in results[2:end]
                @test isapprox(r.rho, results[1].rho; rtol = 1e-12, atol = 1e-12)
                if boundary_mode === UpdatedMDBC
                    @test r.active == results[1].active
                    @test isapprox(r.vel, results[1].vel; rtol = 1e-12, atol = 1e-12)
                    @test isapprox(r.tvel, results[1].tvel; rtol = 1e-12, atol = 1e-12)
                end
            end
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

    @testset "periodic queries and periodic images support Rx = 1, 2, 4" begin
        for D in (2, 3), periodic_images in (false, true)
            T = Float64
            dx = 0.02
            p = PeriodicBoundary{D}(1; min = 0.0, max = 0.4, shift = ntuple(d -> d == 2 ? 0.02 : 0.0, D))
            constants = SimulationConstants{T}(dx = dx, c₀ = 20.0, g = 0.0, ρ₀ = 1000.0)
            kernel = SPHKernelInstance{D, T}(WendlandC2(); h = 0.04)
            pts = [SVector{D, T}(ntuple(d -> d == 1 ? x : d == 2 ? y : 0.02, D))
                   for x in (0.01, 0.03, 0.37, 0.39) for y in (0.02, 0.04)]
            dens = [1000.0 + 0.5 * i for i in eachindex(pts)]
            vels = [SVector{D, T}(ntuple(d -> 0.1 * (i - d), D)) for i in eachindex(pts)]
            geom = [SPHGeometry{D, T}(Particles = particle_struct_array(pts, constants.ρ₀;
                Density = dens, Velocity = vels), GroupMarker = 1, Type = Fluid)]
            results = map((1, 2, 4)) do Rx
                meta = SimulationMetaData{D, T}(SimulationName = "per_rx", SaveLocation = tempdir(),
                    Periodicity = p, GPUPeriodicImages = periodic_images,
                    GPUCellSubdivision = 1, GPUXCellSubdivision = Rx,
                    GPUDoublePosition = true, VisualizeInParaview = false, OpenLogFile = false)
                host = AllocateDataStructures(geom, meta)
                gpu = upload_particles(host, meta)
                sup = GPUSupportArrays{D, T}(gpu, meta; kernel)
                cl = CellListWorkspace{D, Float64}(length(gpu); reach = 1, x_reach = Rx, periodicity = p)
                SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu, cl, kernel.H⁻¹; support = sup, constants, kernel)
                periodic_images && SPHExampleGPU.SPHCellList.configure_periodic_images!(sup.PeriodicImages, p, cl.grid, kernel)
                ctx = (; gpu, sup, cl, SimKernel = kernel, SimConstants = constants,
                    SimDensityDiffusion = ZeroDensityDiffusion(), SimViscosity = ArtificialViscosity(),
                    FlagKernel = Val(false), FlagShift = Val(false), threads = 128, lanes = Val(1), bforces = Val(true))
                step = HostStep(0.001, 0.0)
                SPHExampleGPU.SPHCellList.prepare_periodic_bounds!(ctx, gpu.Position, step)
                SPHExampleGPU.SPHCellList.launch_solver_interactions!(ctx, gpu.Position, gpu.Density, gpu.Velocity, sup.PosCells, step)
                order = sortperm(Array(gpu.ID))
                (; rate = Array(sup.dρdtI)[order], acc = Array(gpu.Acceleration)[order])
            end
            for r in results[2:end]
                @test isapprox(r.rate, results[1].rate; rtol = 1e-12, atol = 1e-12)
                @test isapprox(r.acc, results[1].acc; rtol = 1e-12, atol = 1e-12)
            end
        end
    end

    @testset "fine-grid short trajectories with orientation-independent diffusion" begin
        for D in (2, 3)
            states = map(((1, 1), (2, 2), (4, 4), (1, 2), (1, 4))) do (R, Rx)
                v = B.setup(tiny_case(D), Float64, R; x_subdiv = Rx, lanes = 1)
                B.loop!(v)
                order = sortperm(Array(v.gpu.ID))
                fields = map(f -> Array(getproperty(v.gpu, f))[order], (:Position, :Velocity, :Density, :Acceleration))
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
