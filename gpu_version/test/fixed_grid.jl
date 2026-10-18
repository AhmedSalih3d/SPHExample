module FixedGridTests

using Test, SPHExampleGPU, CUDA, StaticArrays

include(joinpath(@__DIR__, "..", "benchmark", "cases.jl"))

const GRID = SPHExampleGPU.GPUCellGrid
const STEP = SPHExampleGPU.GPUStepState

function rebuild(points::Vector{SVector{D, T}}, workspace; inverse_cutoff = one(T)) where {D, T}
    position = CuArray(points)
    sorted_position = similar(position)
    cell_ids = CuVector{Int32}(undef, length(points))
    grid = update_cell_list!(workspace, position, inverse_cutoff,
        (position, workspace.CellIDScratch), (sorted_position, cell_ids))
    return (; grid, position = Array(sorted_position), cell_ids = Array(cell_ids))
end

function check_neighbors(result, workspace, inverse_cutoff)
    grid, points, ids = result.grid, result.position, result.cell_ids
    scale = GRID.bin_scale(grid, inverse_cutoff)
    coordinates = [GRID.cell_coords(p, scale) for p in points]
    starts = Array(view(workspace.CellStart, 1:(Int(grid.ncells) + 1)))
    @test starts[end] == length(points)
    @test all(i -> GRID.global_coords(grid, ids[i]) == coordinates[i], eachindex(ids))
    for i in eachindex(points)
        candidates = Int[]
        for row in GRID.cell_rows(grid)
            lo, hi = GRID.row_range(grid, starts, ids[i], row)
            append!(candidates, Int(lo):Int(hi))
        end
        expected = findall(c -> all(abs(c[d] - coordinates[i][d]) <= GRID.reach(grid)
                                   for d in eachindex(c)), coordinates)
        @test sort(candidates) == expected
    end
end

function simulate(case, ::Type{T}; fixed_grid, scheme, double_position = false) where {T}
    return mktempdir() do save
        kwargs = case.build(T, save)
        metadata = kwargs.SimMetaData
        metadata.GPUFixedGrid = fixed_grid
        metadata.GPUDoublePosition = double_position
        metadata.GPUUseGraph = fixed_grid
        metadata.GPUMaxStepsPerSync = fixed_grid ? 32 : 1
        metadata.SimulationTime = T(0.01)
        metadata.OutputTimes = T(0.005)
        particles = AllocateDataStructures(kwargs.SimGeometry, metadata)
        logger = SimulationLogger(save)
        try
            redirect_stdout(devnull) do
                RunSimulation(; merge(kwargs, (; SimTimeStepping = scheme))...,
                    SimParticles = particles, SimLogger = logger)
            end
        finally
            isopen(logger.LoggerIo) && close(logger.LoggerIo)
        end
        order = sortperm(particles.ID)
        return (; steps = metadata.Iteration, time = metadata.TotalTime,
            position = particles.Position[order], velocity = particles.Velocity[order],
            density = particles.Density[order], pressure = particles.Pressure[order])
    end
end

@testset "fixed grid configuration" begin
    for D in (2, 3)
        metadata = SimulationMetaData{D, Float32}(SimulationName = "FixedGrid", SaveLocation = tempdir())
        @test !metadata.GPUFixedGrid
        @test metadata.GPUGridExpansion == ntuple(_ -> (false, false), D)
        expansion = ntuple(d -> (false, d == D), D)
        metadata = SimulationMetaData{D, Float32}(SimulationName = "FixedGrid", SaveLocation = tempdir(),
            GPUFixedGrid = true, GPUGridExpansion = expansion)
        @test metadata.GPUGridExpansion == expansion
        @test_throws ArgumentError SimulationMetaData{D, Float32}(
            SimulationName = "Invalid", SaveLocation = tempdir(), GPUGridExpansion = expansion)
        @test_throws ArgumentError CellListWorkspace{D, Float32}(2; expansion)
    end
end

if CUDA.functional()
    @testset "fixed initial bounds: D=$D T=$T R=$R" for D in (2, 3), T in (Float32, Float64), R in (1, 2, 4)
        initial = [SVector{D, T}(ntuple(_ -> T(sign * 2), D)) for sign in (-1, 1)]
        workspace = CellListWorkspace{D, T}(2; reach = R, fixed_grid = true)
        first = rebuild(initial, workspace)
        dynamic = rebuild(initial, CellListWorkspace{D, T}(2; reach = R))
        @test first == dynamic
        generation = workspace.generation
        pointers = UInt.(pointer.((workspace.CellStart, workspace.Counts)))
        interior = initial ./ T(2)
        result = rebuild(interior, workspace)
        @test result.grid == first.grid
        @test workspace.generation == generation
        @test UInt.(pointer.((workspace.CellStart, workspace.Counts))) == pointers
        check_neighbors(result, workspace, one(T))

        for d in 1:D, sign in (-1, 1)
            escaped = copy(interior)
            escaped[1] = SVector{D, T}(ntuple(axis -> axis == d ? T(sign * 4) : zero(T), D))
            permutation = Array(workspace.Perm)
            cells = Array(workspace.CellIDScratch)
            @test_throws r"locked side" rebuild(escaped, workspace)
            @test Array(workspace.grid_state_dev)[1].status == GRID.GRID_FIXED_BOUNDS
            @test Array(workspace.grid_dev)[1] == first.grid
            @test workspace.grid == first.grid
            @test Array(workspace.Perm) == permutation
            @test Array(workspace.CellIDScratch) == cells
            @test !workspace.grid_prepared
        end
        # Validation errors must not poison subsequent valid preparations.
        @test rebuild(initial, workspace).grid == first.grid
        check_neighbors(rebuild(initial, workspace), workspace, one(T))
    end

    @testset "each expansion direction: D=$D T=$T R=$R" for D in (2, 3), T in (Float32, Float64), R in (1, 2)
        initial = [SVector{D, T}(ntuple(_ -> T(sign), D)) for sign in (-1, 1)]
        for axis in 1:D, side in 1:2
            expansion = ntuple(d -> (d == axis && side == 1, d == axis && side == 2), D)
            workspace = CellListWorkspace{D, T}(2; reach = R, fixed_grid = true, expansion)
            first = rebuild(initial, workspace).grid
            points = copy(initial)
            points[side] = SVector{D, T}(ntuple(d -> d == axis ? T(side == 1 ? -4 : 4) : zero(T), D))
            result = rebuild(points, workspace)
            check_neighbors(result, workspace, one(T))
            for d in 1:D
                @test result.grid.origin[d] == first.origin[d] - (d == axis && side == 1 ? 3R : 0)
                @test result.grid.dims[d] == first.dims[d] + (d == axis ? 3R : 0)
            end
            # Both the original domain and previously expanded cells are retained.
            @test rebuild(initial, workspace).grid == result.grid
            opposite = copy(initial)
            opposite[1] = -points[side]
            @test_throws r"locked side" rebuild(opposite, workspace)
        end
    end

    @testset "growth validation and timestep header" begin
        for D in (2, 3), T in (Float32, Float64)
            expansion = ntuple(d -> (false, d == 1), D)
            workspace = CellListWorkspace{D, T}(2; fixed_grid = true, expansion, max_cells = 100_000)
            initial = [zero(SVector{D, T}), SVector{D, T}(ntuple(_ -> one(T), D))]
            position = CuArray(initial)
            state = StepState{T}()
            GRID.prepare_cell_grid!(workspace, position, one(T); step = state)
            readback!(state)
            @test !GRID.consume_cell_grid!(workspace, state.ih)
            @test !workspace.grid_initialized
            state.ih[STEP.I_STOP] = STEP.STOP_REBUILD
            copyto!(state.i, state.ih)
            GRID.prepare_cell_grid!(workspace, position, one(T); step = state)
            readback!(state)
            @test GRID.consume_cell_grid!(workspace, state.ih)
            first = workspace.grid
            generation = workspace.generation
            points = copy(initial)
            points[2] = SVector{D, T}(ntuple(d -> d == 1 ? T(2000) : one(T), D))
            copyto!(position, points)
            GRID.prepare_cell_grid!(workspace, position, one(T); step = state)
            readback!(state)
            @test state.ih[STEP.I_GRID_STATUS] == GRID.GRID_CAPACITY
            @test GRID.consume_cell_grid!(workspace, state.ih)
            @test workspace.grid.origin == first.origin
            @test workspace.generation > generation
            @test workspace.grid_status_readbacks == 0
            expanded = workspace.grid
            rebuild(initial, workspace)
            @test workspace.grid == expanded
            for invalid in (T(NaN), T(Inf), T(2)^40)
                points[2] = SVector{D, T}(ntuple(d -> d == 1 ? invalid : zero(T), D))
                @test_throws ErrorException rebuild(points, workspace)
                @test workspace.grid == expanded
            end
            points[2] = SVector{D, T}(ntuple(d -> d == 1 ? T(100_000) : zero(T), D))
            @test_throws r"allowed .* cells" rebuild(points, workspace)
            @test workspace.grid == expanded
        end
    end

    @testset "simultaneous growth retains the union of visited bounds" begin
        for D in (2, 3), R in (1, 2)
            initial = [SVector{D, Float64}(ntuple(_ -> sign, D)) for sign in (-1, 1)]
            workspace = CellListWorkspace{D, Float64}(2; reach = R, fixed_grid = true,
                expansion = ntuple(_ -> (true, true), D))
            first = rebuild(initial, workspace).grid
            shift = SVector{D, Float64}(ntuple(d -> isodd(d) ? 4 : -4, D))
            rebuild(initial .+ Ref(shift), workspace)
            result = rebuild(initial .- Ref(shift), workspace)
            @test result.grid.origin == first.origin .- 4R
            @test result.grid.dims == first.dims .+ 8R
            check_neighbors(result, workspace, 1.0)
            @test rebuild(initial, workspace).grid == result.grid
        end
    end

    @testset "fixed simulation matches dynamic grid and direct launches" begin
        for name in ("StillWedge2D_MDBC_dp0.02", "DamBreak3D_dp0.02"),
            (T, double_position) in ((Float32, true), (Float64, false)),
            scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            case = only(filter(c -> c.name == name, BENCH_CASES))
            dynamic = simulate(case, T; fixed_grid = false, scheme, double_position)
            fixed = simulate(case, T; fixed_grid = true, scheme, double_position)
            @test fixed.steps > 1
            @test fixed == dynamic
        end
    end
end

end
