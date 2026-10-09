using Test, SPHExampleGPU, CUDA, StaticArrays, StructArrays

const CELL_GRID = SPHExampleGPU.GPUCellGrid
const CELL_DRIVER = SPHExampleGPU.SPHCellList

# The original host construction independently checks rounding, stencil
# margins, prefix boundaries, and deterministic counting-sort order.
function cell_rebuild_reference(points::Vector{SVector{D, TP}}, inverse_cutoff, reach) where {D, TP}
    inv_cell = typeof(inverse_cutoff)(reach) * inverse_cutoff
    old_map(x) = Int32(sign(x)) * unsafe_trunc(Int32, muladd(abs(x), inv_cell, TP(0.5)))
    coordinates = [ntuple(d -> old_map(p[d]), D) for p in points]
    origin = ntuple(d -> minimum(c[d] for c in coordinates) - Int32(reach), D)
    upper = ntuple(d -> maximum(c[d] for c in coordinates) + Int32(reach), D)
    dims = ntuple(d -> upper[d] - origin[d] + Int32(1), D)
    ncells = Int32(prod(Int64.(dims)))
    grid = CellGrid{D, reach}(origin, dims, ncells)
    cell_ids = Int32[1 + sum((c[d] - origin[d]) * prod(Int64.(dims[1:d-1]); init = 1)
                             for d in 1:D) for c in coordinates]
    counts = zeros(Int32, ncells)
    for c in cell_ids
        counts[c] += Int32(1)
    end
    starts = vcat(Int32(0), cumsum(counts))
    permutation = sortperm(eachindex(points); by = i -> (cell_ids[i], i))
    return (; grid, coordinates, cell_ids, counts, starts, permutation)
end

function cell_rebuild_fixture(::Val{D}, ::Type{T}, ::Type{TP}; reach = 1) where {D, T, TP}
    points = vec([SVector{D, TP}(ntuple(d -> TP(Tuple(c)[d]) * TP(0.08), D))
                  for c in CartesianIndices(ntuple(_ -> -2:2, D))])
    # Coincident and adjacent-cell particles expose overlapping scatter slots.
    append!(points, points[1:3])
    points = points[reverse(eachindex(points))]
    n = length(points)
    vec_t(i) = SVector{D, T}(ntuple(d -> T(i * d) / T(16), D))
    vec_p(i) = SVector{D, TP}(ntuple(d -> TP(i + d) / TP(32), D))
    host = StructArray((
        Position = points,
        Velocity = [vec_t(i) for i in 1:n],
        Density = T.(1000 .+ (1:n)),
        ID = collect(reverse(1:n)),
        Type = [iszero(i % 4) ? Floating : (iszero(i % 3) ? Moving : (iszero(i % 2) ? Fixed : Fluid)) for i in 1:n],
        GroupMarker = UInt.(mod.(1:n, 5) .+ 1),
        GhostPoints = [iszero(i % 3) ? vec_p(i) : zero(SVector{D, TP}) for i in 1:n],
        GhostNormals = [vec_t(-i) for i in 1:n],
        Acceleration = [vec_t(i + n) for i in 1:n],
        Pressure = T.(1:n) ./ T(8),
        Kernel = zeros(T, n),
        KernelGradient = zeros(SVector{D, T}, n),
        Cells = fill(CartesianIndex(ntuple(_ -> 0, D)), n),
    ))
    gpu = upload_particles(host; position_type = TP)
    ws = CellListWorkspace{D, TP}(n; reach)
    floating = (; active = true, indices = CuArray(Int32.(findall(==(Floating), host.Type))))
    return (; host, gpu, ws, floating)
end

function check_cell_rebuild!(fixture, inverse_cutoff; neighbors = true)
    gpu, ws = fixture.gpu, fixture.ws
    before = NamedTuple{CELL_DRIVER.PERSISTENT_FIELDS}(Tuple(Array(getproperty(gpu, f))
                                                            for f in CELL_DRIVER.PERSISTENT_FIELDS))
    reference = cell_rebuild_reference(before.Position, inverse_cutoff, Int(CELL_GRID.reach(ws.grid)))
    grid = CELL_DRIVER.rebuild_cell_list!(gpu, ws, inverse_cutoff)
    SPHExampleGPU.GPUFloating.update_floating_indices!(fixture.floating, gpu.Type, ws)
    @test grid == reference.grid
    @test Array(ws.grid_dev)[1] == grid
    @test Array(view(ws.CellStart, 1:(Int(grid.ncells) + 1))) == reference.starts
    @test reference.starts[end] == length(gpu)
    for f in CELL_DRIVER.PERSISTENT_FIELDS
        @test Array(getproperty(gpu, f)) == getproperty(before, f)[reference.permutation]
    end
    sorted_ids = reference.cell_ids[reference.permutation]
    @test Array(gpu.CellID) == sorted_ids
    @test Array(gpu.GhostIndex) == Int32.(findall(!iszero, Array(gpu.GhostPoints)))
    @test Array(fixture.floating.indices) == Int32.(findall(==(Floating), Array(gpu.Type)))
    coordinates = reference.coordinates[reference.permutation]
    expected_cells = unique(CartesianIndex.(coordinates))
    @test CELL_GRID.unique_cells_host(grid, sorted_ids) == expected_cells

    if neighbors
        for i in eachindex(sorted_ids)
            candidates = Int[]
            for row in CELL_GRID.cell_rows(grid)
                lo, hi = CELL_GRID.row_range(grid, reference.starts, sorted_ids[i], row)
                append!(candidates, Int(lo):Int(hi))
            end
            expected = findall(c -> all(abs(c[d] - coordinates[i][d]) <= CELL_GRID.reach(grid)
                                       for d in eachindex(c)), coordinates)
            @test sort(candidates) == expected
        end
    end
    return (; grid, reference)
end

function simple_cell_rebuild(points::Vector{SVector{D, T}}; inverse_cutoff = one(T),
                             max_cells = 50_000_000, reach = 1, workspace = nothing) where {D, T}
    n = length(points)
    ws = workspace === nothing ? CellListWorkspace{D, T}(n; max_cells, reach) : workspace
    position = CuArray(points)
    ids = CuArray(collect(1:n))
    sorted_position = similar(position)
    sorted_ids = similar(ids)
    cell_ids = CuVector{Int32}(undef, n)
    sources = (position, ids, ws.CellIDScratch)
    destinations = (sorted_position, sorted_ids, cell_ids)
    grid = CELL_GRID.update_cell_list!(ws, position, inverse_cutoff, sources, destinations)
    CUDA.synchronize()
    return (; ws, grid, position, ids, sorted_position, sorted_ids, cell_ids, sources, destinations)
end

function cell_rebuild_driver_growth(use_graph; scheme = SymplecticTimeStepping())
    T = Float64
    points = [SVector{2, T}(T(i) * T(0.02), T(j) * T(0.02)) for j in 0:7 for i in 0:7]
    geometry = [SPHGeometry{2, T}(points; Density = T(1000), GroupMarker = 1, Type = Fluid)]
    metadata = SimulationMetaData{2, T}(SimulationName = "CellRebuildGrowth", SaveLocation = mktempdir(),
        SimulationTime = T(0.004), OutputTimes = T(0.002), VisualizeInParaview = false,
        OpenLogFile = false, GPUUseGraph = use_graph, GPUMaxStepsPerSync = 4)
    metadata.TimeSteppingMode = scheme
    metadata.OutputIterationCounter = 1
    constants = SimulationConstants{T}(dx = T(0.02), g = zero(T), c₀ = T(20))
    kernel = SPHKernelInstance{2, T}(WendlandC2(); dx = constants.dx)
    particles = AllocateDataStructures(geometry, metadata)
    gpu = upload_particles(particles)
    support = GPUSupportArrays{2, T}(length(gpu))
    reduction = ReductionWorkspace{SVector{3, T}}(length(gpu))
    cells = CellListWorkspace{2, T}(length(gpu))
    motion = MotionArrays(geometry, particles)
    state = StepState{T}(; dx = one(T) + kernel.h)
    advance!() = SimulationLoop(ZeroDensityDiffusion(), ZeroViscosity(), kernel, metadata,
                               constants, gpu, cells, support, reduction, motion, state)
    advance!()
    graph_count = length(state.graphs)
    @test use_graph ? graph_count > 0 : graph_count == 0
    generation = cells.generation
    old_pointer = UInt(pointer(cells.CellStart))
    keys_before = collect(keys(state.graphs))

    # The first interval populates the step graph cache. Force a subsequent
    # dynamic-grid growth while those graph executables still refer to it.
    moved = Array(gpu.Position)
    moved[1] += SVector{2, T}(20, 0)
    copyto!(gpu.Position, moved)
    S = SPHExampleGPU.GPUStepState
    S.set_slot!(state.f, state.fh, S.F_DX, one(T) + kernel.h)
    metadata.OutputIterationCounter += 1
    advance!()
    @test cells.generation > generation
    @test UInt(pointer(cells.CellStart)) != old_pointer
    @test cells.capacity_growths > 0
    @test cells.grid_status_readbacks == 0
    @test cells.bbox_host_readbacks == 0
    @test cells.nrebuilds >= 2
    if use_graph
        @test all(key -> key[2] == cells.generation, keys(state.graphs))
        @test all(key -> !haskey(state.graphs, key), keys_before)
        @test !isempty(state.graphs)
    end
    fields = NamedTuple{CELL_DRIVER.PERSISTENT_FIELDS}(Tuple(Array(getproperty(gpu, f))
                                                            for f in CELL_DRIVER.PERSISTENT_FIELDS))
    return (; fields, cell_ids = Array(gpu.CellID), grid = cells.grid, steps = metadata.Iteration,
              time = metadata.TotalTime, dt = metadata.CurrentTimeStep, rebuilds = cells.nrebuilds)
end

if CUDA.functional()
    @testset "device-resident cell-list rebuild" begin
        @testset "host reference: D=$D T=$T TP=$TP reach=$reach" for D in (2, 3),
                (T, TP) in ((Float32, Float32), (Float64, Float64), (Float32, Float64)), reach in (1, 2, 3)
            fixture = cell_rebuild_fixture(Val(D), T, TP; reach)
            inverse_cutoff = T(10)
            first_grid = check_cell_rebuild!(fixture, inverse_cutoff).grid
            pointer_pair = (UInt(pointer(fixture.gpu.Position)), UInt(pointer(fixture.gpu.scratch.Position)))
            ghost_pointer = UInt(pointer(fixture.gpu.GhostIndex))
            floating_pointer = UInt(pointer(fixture.floating.indices))
            growths = fixture.ws.capacity_growths
            generation = fixture.ws.generation
            check_cell_rebuild!(fixture, inverse_cutoff)
            @test UInt(pointer(fixture.gpu.Position)) == pointer_pair[2]
            @test fixture.ws.capacity_growths == growths
            @test fixture.ws.generation == generation

            translation = SVector{D, TP}(ntuple(d -> TP(isodd(d) ? 0.4 : -0.7), D))
            translated = Array(fixture.gpu.Position) .+ Ref(translation)
            copyto!(fixture.gpu.Position, translated)
            moved = check_cell_rebuild!(fixture, inverse_cutoff)
            @test moved.grid.origin != first_grid.origin
            @test UInt(pointer(fixture.gpu.Position)) == pointer_pair[1]
            @test UInt(pointer(fixture.gpu.GhostIndex)) == ghost_pointer
            @test UInt(pointer(fixture.floating.indices)) == floating_pointer
            @test fixture.ws.bbox_host_readbacks == 0
            @test fixture.ws.grid_status_readbacks == fixture.ws.nrebuilds == 3
        end

        @testset "cell rounding at positive and negative half cells" begin
            for T in (Float32, Float64)
                points = [SVector{2, T}(x, zero(T)) for x in
                          T[-1.5, -0.5, -0.0, 0.0, 0.5, 1.5]]
                result = simple_cell_rebuild(points)
                ref = cell_rebuild_reference(points, one(T), 1)
                @test result.grid == ref.grid
                @test Array(result.sorted_ids) == ref.permutation
                @test Array(result.cell_ids) == ref.cell_ids[ref.permutation]
                @test Array(view(result.ws.CellStart, 1:(Int(result.grid.ncells) + 1))) == ref.starts
            end
        end

        @testset "preallocated hierarchical scans" begin
            scan = CELL_GRID.IntScanWorkspace(65_537)
            pointers = UInt.(pointer.(vcat(scan.sums, scan.offsets)))
            for n in (0, 1, 255, 256, 257, 65_537), inclusive in (false, true)
                values = Int32[mod(i, 7) for i in 1:n]
                expected = inclusive ? cumsum(values) : vcat(Int32(0), cumsum(values))[1:n]
                input = CuArray(values)
                output = similar(input)
                CELL_GRID.scan_int32!(scan, output, input; inclusive)
                @test Array(output) == expected
                CELL_GRID.scan_int32!(scan, input, input; inclusive)
                @test Array(input) == expected
                @test UInt.(pointer.(vcat(scan.sums, scan.offsets))) == pointers
            end
        end

        @testset "capacity growth preserves the grid and invalidates pointers" begin
            points = [SVector{2, Float64}(0, 0), SVector{2, Float64}(1, 0)]
            result = simple_cell_rebuild(points)
            initial_generation = result.ws.generation
            initial_pointer = UInt(pointer(result.ws.CellStart))
            grown_points = [SVector{2, Float64}(0, 0), SVector{2, Float64}(2000, 0)]
            grown = simple_cell_rebuild(grown_points; workspace = result.ws)
            ref = cell_rebuild_reference(grown_points, 1.0, 1)
            @test grown.grid == ref.grid
            @test Array(grown.sorted_ids) == ref.permutation
            @test length(grown.ws.Counts) >= grown.grid.ncells
            @test grown.ws.generation > initial_generation
            @test grown.ws.capacity_growths > 0
            @test UInt(pointer(grown.ws.CellStart)) != initial_pointer
            generation = grown.ws.generation
            pointer_after_growth = UInt(pointer(grown.ws.CellStart))
            simple_cell_rebuild(grown_points; workspace = grown.ws)
            @test grown.ws.generation == generation
            @test UInt(pointer(grown.ws.CellStart)) == pointer_after_growth
        end

        @testset "invalid grids stop before histogram or scatter" begin
            for T in (Float32, Float64)
                valid = [SVector{2, T}(0, 0), SVector{2, T}(1, 0), SVector{2, T}(2, 0)]
                ws = CellListWorkspace{2, T}(length(valid))
                fill!(ws.Perm, Int32(-17))
                fill!(ws.CellIDScratch, Int32(-19))
                for bad in (T(NaN), T(Inf), T(-Inf))
                    # Put NaN between finite samples: min/max alone can hide it.
                    points = copy(valid)
                    points[2] = SVector{2, T}(bad, zero(T))
                    @test_throws ErrorException simple_cell_rebuild(points; workspace = ws)
                    @test Array(ws.Perm) == fill(Int32(-17), length(valid))
                    @test Array(ws.CellIDScratch) == fill(Int32(-19), length(valid))
                    @test ws.nrebuilds == 0
                end
                huge = [SVector{2, T}(0, 0), SVector{2, T}(T(2)^32, 0)]
                @test_throws ErrorException simple_cell_rebuild(huge)
                product_overflow = [SVector{2, T}(0, 0), SVector{2, T}(50_000, 50_000)]
                @test_throws ErrorException simple_cell_rebuild(product_overflow;
                                                               max_cells = typemax(Int32))
                for inverse_cutoff in (zero(T), T(-1), T(NaN), T(Inf))
                    @test_throws ErrorException simple_cell_rebuild(valid; inverse_cutoff)
                end
                @test_throws ErrorException simple_cell_rebuild(valid; max_cells = 1)
            end
            # Mapped coordinates fit, but adding the stencil margin must fail.
            margin_overflow = [SVector{2, Float64}(Float64(typemax(Int32)) - 0.25, 0)]
            @test_throws ErrorException simple_cell_rebuild(margin_overflow)
        end

        @testset "nonfinite detection in later blocks and grid-stride passes" begin
            for n in (257, 131_073), T in (Float32, Float64)
                points = [SVector{2, T}(T(mod(i - 1, 512)), T((i - 1) ÷ 512)) for i in 1:n]
                ws = CellListWorkspace{2, T}(n)
                @test ws.bbox_ws.nblocks > 1
                n > 131_072 && @test ws.bbox_ws.nblocks == 512
                invalid = copy(points)
                invalid[end] = SVector{2, T}(T(NaN), zero(T))
                fill!(ws.Perm, Int32(-17))
                @test_throws ErrorException simple_cell_rebuild(invalid; workspace = ws)
                @test Array(ws.grid_state_dev)[1].status == CELL_GRID.GRID_NONFINITE
                @test Array(ws.Perm) == fill(Int32(-17), n)
                # Per-block flags must be freshly overwritten after an error.
                recovered = simple_cell_rebuild(points; workspace = ws)
                @test Array(recovered.sorted_ids) == collect(1:n)
                @test Array(ws.grid_state_dev)[1].status in (CELL_GRID.GRID_OK, CELL_GRID.GRID_CAPACITY)
                @test ws.nrebuilds == 1
            end
        end

        @testset "conditional preparation uses the existing timestep readback" begin
            S = SPHExampleGPU.GPUStepState
            for D in (2, 3), T in (Float32, Float64)
                points = [zero(SVector{D, T}), SVector{D, T}(ntuple(_ -> T(0.2), D))]
                position = CuArray(points)
                ws = CellListWorkspace{D, T}(length(points))
                state = StepState{T}()
                CELL_GRID.prepare_cell_grid!(ws, position, one(T); step = state)
                S.readback!(state)
                @test state.ih[S.I_GRID_STATUS] == CELL_GRID.GRID_NOT_NEEDED
                @test !CELL_GRID.consume_cell_grid!(ws, state.ih)
                @test ws.grid_status_readbacks == 0

                state.ih[S.I_STOP] = S.STOP_REBUILD
                copyto!(state.i, state.ih)
                CELL_GRID.prepare_cell_grid!(ws, position, one(T); step = state)
                S.readback!(state)
                @test CELL_GRID.consume_cell_grid!(ws, state.ih)
                @test ws.grid == cell_rebuild_reference(points, one(T), 1).grid
                @test ws.grid_status_readbacks == 0
                @test state.readbacks == 2

                points[1] = SVector{D, T}(ntuple(_ -> T(NaN), D))
                copyto!(position, points)
                CELL_GRID.prepare_cell_grid!(ws, position, one(T); step = state)
                S.readback!(state)
                @test state.ih[S.I_GRID_STATUS] == CELL_GRID.GRID_NONFINITE
                @test_throws ErrorException CELL_GRID.consume_cell_grid!(ws, state.ih)
                @test !ws.grid_prepared
                @test ws.grid_status_readbacks == 0
            end
        end

        @testset "prevalidated rebuild graph replay" begin
            # Capture only the prevalidated enqueue phase. The scan lengths,
            # dimensions, and particle-buffer pointers are fixed by capture.
            for D in (2, 3), T in (Float32, Float64)
                points = [SVector{D, T}(ntuple(d -> T(mod(i + d, 3)) / T(4), D)) for i in 1:33]
                result = simple_cell_rebuild(points; inverse_cutoff = T(4))
                ws = result.ws
                reference = cell_rebuild_reference(points, T(4), 1)
                CELL_GRID.prepare_cell_grid!(ws, result.position, T(4))
                @test CELL_GRID.sync_cell_grid!(ws)
                graph = CUDA.capture(; throw_error = true) do
                    CELL_GRID.update_cell_list!(ws, result.position, T(4), result.sources,
                                                result.destinations; prepared = true)
                end
                executable = CUDA.instantiate(graph)
                generation = ws.generation
                pointers = UInt.(pointer.((ws.CellStart, ws.Counts, ws.Perm, ws.CellIDScratch)))
                for repetition in 1:3
                    translated = points .+ Ref(SVector{D, T}(ntuple(d -> T(repetition * d), D)))
                    copyto!(result.position, translated)
                    CELL_GRID.prepare_cell_grid!(ws, result.position, T(4))
                    @test CELL_GRID.sync_cell_grid!(ws)
                    ref = cell_rebuild_reference(translated, T(4), 1)
                    @test ws.grid.dims == reference.grid.dims
                    reads = ws.grid_status_readbacks
                    CUDA.launch(executable)
                    @test Array(result.sorted_position) == translated[ref.permutation]
                    @test Array(result.sorted_ids) == ref.permutation
                    @test Array(result.cell_ids) == ref.cell_ids[ref.permutation]
                    @test Array(view(ws.CellStart, 1:(Int(ws.grid.ncells) + 1))) == ref.starts
                    @test ws.grid_status_readbacks == reads
                    @test ws.generation == generation
                    @test UInt.(pointer.((ws.CellStart, ws.Counts, ws.Perm, ws.CellIDScratch))) == pointers
                end
                # Failed preparation cannot authorize the normal prepared API.
                invalid = copy(points)
                invalid[1] = SVector{D, T}(ntuple(_ -> T(NaN), D))
                copyto!(result.position, invalid)
                CELL_GRID.prepare_cell_grid!(ws, result.position, T(4))
                @test_throws ErrorException CELL_GRID.sync_cell_grid!(ws)
                @test_throws ErrorException CELL_GRID.update_cell_list!(ws, result.position, T(4),
                    result.sources, result.destinations; prepared = true)
            end
        end

        @testset "simulation graphs survive capacity growth" begin
            for scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
                captured = cell_rebuild_driver_growth(true; scheme)
                direct = cell_rebuild_driver_growth(false; scheme)
                @test captured.fields == direct.fields
                @test captured.cell_ids == direct.cell_ids
                @test captured.grid == direct.grid
                @test captured.steps == direct.steps > 1
                @test captured.time == direct.time
                @test captured.dt == direct.dt
                @test captured.rebuilds == direct.rebuilds
            end
        end

        @testset "asynchronous output retains its particle grid snapshot" begin
            fixture = cell_rebuild_fixture(Val(2), Float32, Float64)
            result = check_cell_rebuild!(fixture, 10.0f0)
            expected_position = Array(fixture.gpu.Position)
            expected_cell_ids = Array(fixture.gpu.CellID)
            download = CELL_DRIVER.OutputDownload(fixture.gpu, (:ID,); cells = true)
            try
                CELL_DRIVER.enqueue_download!(download, fixture.gpu)
                copyto!(fixture.gpu.Position, expected_position .+ Ref(SVector{2, Float64}(8, -4)))
                check_cell_rebuild!(fixture, 10.0f0)
                unique_cells = CELL_DRIVER.finish_download!(download, fixture.host, result.grid)
                @test fixture.host.Position == expected_position
                @test unique_cells == CELL_GRID.unique_cells_host(result.grid, expected_cell_ids)
                @test fixture.host.Cells == CartesianIndex.([CELL_GRID.global_coords(result.grid, c)
                                                             for c in expected_cell_ids])
            finally
                CELL_DRIVER.free!(download)
            end
        end
    end
end
