using Random
using StaticArrays

struct GridDescription{D}
    domain_min::NTuple{D,Int}
    cells_per_axis::NTuple{D,Int}
end

mutable struct BenchmarkParticles{D}
    Cells::Vector{CartesianIndex{D}}
    Position::Vector{SVector{D,Float64}}
end

@inline function map_floor(x, inv_cutoff)
    Int(sign(x)) * unsafe_trunc(Int, muladd(abs(x), inv_cutoff, 0.5))
end

@inline function linear_cell_id(cell::CartesianIndex{D},
                                grid::GridDescription{D}) where D
    linear = one(Int)
    stride = one(Int)
    @inbounds for dim in 1:D
        offset = cell[dim] - grid.domain_min[dim]
        if offset < 0 || offset >= grid.cells_per_axis[dim]
            return zero(Int)
        end
        linear += offset * stride
        stride *= grid.cells_per_axis[dim]
    end
    return linear
end

function update_grid_description(cells::Vector{CartesianIndex{D}}) where D
    mins = ntuple(i -> minimum(ci -> ci[i], cells), D)
    maxs = ntuple(i -> maximum(ci -> ci[i], cells), D)
    counts = ntuple(i -> maxs[i] - mins[i] + 1, D)
    return GridDescription{D}(mins, counts)
end

function extract_cells!(particles::BenchmarkParticles{D}, inv_cutoff) where D
    @inbounds for i in eachindex(particles.Cells)
        coords = Tuple(particles.Position[i])
        particles.Cells[i] = CartesianIndex(map(x -> map_floor(x, inv_cutoff), coords))
    end
    return particles
end

function sort_particles!(particles::BenchmarkParticles)
    order = sortperm(particles.Cells)
    particles.Cells = particles.Cells[order]
    particles.Position = particles.Position[order]
    return particles
end

function update_neighbors_hash!(particles::BenchmarkParticles{D}, inv_cutoff,
                                ranges, unique_cells, dict) where D
    extract_cells!(particles, inv_cutoff)
    sort_particles!(particles)
    cells = particles.Cells

    fill!(ranges, 0)
    ranges[1] = 1
    idx = 2
    ranges[idx] = 1
    unique_cells[idx] = cells[1]
    empty!(dict)
    dict[cells[1]] = idx

    @inbounds for i in eachindex(cells)[2:end]
        if cells[i] != cells[i - 1]
            idx += 1
            ranges[idx] = i
            unique_cells[idx] = cells[i]
            dict[cells[i]] = idx
        end
    end

    ranges[idx + 1] = length(cells) + 1
    return idx
end

function update_neighbors_grid!(particles::BenchmarkParticles{D}, inv_cutoff,
                                ranges, unique_cells, cell_id_map,
                                cell_counts, cell_prefix) where D
    extract_cells!(particles, inv_cutoff)
    sort_particles!(particles)
    cells = particles.Cells

    grid = update_grid_description(cells)
    total_cells = prod(grid.cells_per_axis)
    resize!(cell_id_map, total_cells)
    fill!(cell_id_map, 1)
    resize!(cell_counts, total_cells)
    fill!(cell_counts, 0)
    resize!(cell_prefix, total_cells)

    @inbounds @simd for cell in cells
        lid = linear_cell_id(cell, grid)
        cell_counts[lid] += 1
    end

    running = 0
    @inbounds for i in eachindex(cell_counts)
        running += cell_counts[i]
        cell_prefix[i] = running
    end

    fill!(ranges, 0)
    ranges[1] = 1
    idx = 1
    @inbounds for lid in eachindex(cell_counts)
        count = cell_counts[lid]
        if count != 0
            idx += 1
            start_index = cell_prefix[lid] - count + 1
            ranges[idx] = start_index
            unique_cells[idx] = cells[start_index]
            cell_id_map[lid] = idx
        end
    end
    ranges[idx + 1] = length(cells) + 1

    return idx
end

function synthetic_particles(dimensions, n; spread = 25.0)
    positions = [SVector{dimensions}(randn(dimensions) .* spread)
                 for _ in 1:n]
    cells = fill(zero(CartesianIndex{dimensions}), n)
    return BenchmarkParticles{dimensions}(cells, positions)
end

function benchmark_neighbor_update(dimensions, n; iterations = 5)
    Random.seed!(0xCE11)
    base_particles = synthetic_particles(dimensions, n)
    inv_cutoff = inv(4.0)

    ranges_hash = zeros(Int, n + 2)
    ranges_grid = similar(ranges_hash)
    unique_hash = zeros(CartesianIndex{dimensions}, n)
    unique_grid = similar(unique_hash)
    dict = Dict{CartesianIndex{dimensions}, Int}()
    cell_id_map = Int[]
    cell_counts = Int[]
    cell_prefix = Int[]

    hash_particles = BenchmarkParticles(copy(base_particles.Cells),
                                        copy(base_particles.Position))
    grid_particles = BenchmarkParticles(copy(base_particles.Cells),
                                        copy(base_particles.Position))

    hash_idx = update_neighbors_hash!(hash_particles, inv_cutoff,
                                      ranges_hash, unique_hash, dict)
    grid_idx = update_neighbors_grid!(grid_particles, inv_cutoff,
                                      ranges_grid, unique_grid, cell_id_map,
                                      cell_counts, cell_prefix)

    @assert hash_idx == grid_idx
    @assert ranges_hash[1:(hash_idx + 1)] == ranges_grid[1:(grid_idx + 1)]
    @assert unique_hash[2:grid_idx] == unique_grid[2:grid_idx]

    function run_hash!()
        total = 0.0
        for _ in 1:iterations
            run_particles = BenchmarkParticles(copy(base_particles.Cells),
                                               copy(base_particles.Position))
            total += @elapsed update_neighbors_hash!(run_particles, inv_cutoff,
                                                     ranges_hash, unique_hash,
                                                     dict)
        end
        return total / iterations
    end

    function run_grid!()
        total = 0.0
        for _ in 1:iterations
            run_particles = BenchmarkParticles(copy(base_particles.Cells),
                                               copy(base_particles.Position))
            total += @elapsed update_neighbors_grid!(run_particles, inv_cutoff,
                                                     ranges_grid, unique_grid,
                                                     cell_id_map, cell_counts,
                                                     cell_prefix)
        end
        return total / iterations
    end

    hash_time = run_hash!()
    grid_time = run_grid!()

    println("Particle count: $(n)")
    println("  Hash-based update: $(hash_time * 1e3) ms/update")
    println("  Linear-id update:  $(grid_time * 1e3) ms/update")
end

for n in (10_000, 50_000)
    benchmark_neighbor_update(3, n; iterations = 6)
end
