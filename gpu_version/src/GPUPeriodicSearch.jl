# Included in GPUKernels. Periodic queries translate the source cell's anchor,
# retaining the same stale-list displacement allowance as the ordinary stencil.
# The broad iterator remains available for low-level calls. The simulation
# driver supplies measured cell-offset bounds for the tighter iterator below.
struct PeriodicPairSource{D, T, P}
    positions::P
end
PeriodicPairSource(::Type{T}, positions::AbstractVector{SVector{D, TP}}) where {T, D, TP} =
    PeriodicPairSource{D, T, typeof(positions)}(positions)
Adapt.adapt_structure(to, p::PeriodicPairSource{D, T}) where {D, T} =
    PeriodicPairSource(T, Adapt.adapt(to, p.positions))
@inline Base.getindex(p::PeriodicPairSource, i) = @inbounds p.positions[i]

# Keep Float64 positions for integration, but use local working-precision
# coordinates for pair arithmetic, as in DualSPHysics' Pos-Cell path.
struct PeriodicCellPairSource{T,P,C,G}
    positions::P
    cells::C
    grid::G
end
PeriodicCellPairSource(::Type{T}, positions, cells, grid) where {T} =
    PeriodicCellPairSource{T,typeof(positions),typeof(cells),typeof(grid)}(positions, cells, grid)
Adapt.adapt_structure(to, p::PeriodicCellPairSource{T}) where {T} =
    PeriodicCellPairSource(T, Adapt.adapt(to,p.positions), Adapt.adapt(to,p.cells), Adapt.adapt(to,p.grid))

struct PeriodicCellPoint{P,C,G,D}
    position::P
    cell::C
    grid::G
    coordinates::NTuple{D,Int32}
end
@inline function Base.getindex(p::PeriodicCellPairSource, i)
    grid = load_grid(p.grid)
    @inbounds cell = p.cells[i]
    @inbounds return PeriodicCellPoint(p.positions[i], cell, grid, global_coords(grid, cell.cell))
end
@inline pair_position(point) = point
@inline pair_position(point::PeriodicCellPoint) = point.position

struct PeriodicRow{D, T}
    lower::NTuple{D, Int32}
    upper_x::Int32
    shift::SVector{D, T}
    off::Int32
end

struct PeriodicRows{D, T, G, P}
    grid::G
    center::NTuple{D, Int32}
    size::T
    periodicity::P
end

@inline function Base.iterate(rows::PeriodicRows{D}, state::Int32 = Int32(0)) where {D}
    grid, s, p = rows.grid, rows.size, rows.periodicity
    R = reach(grid) + Int32(1)
    width = Int32(2) * R + Int32(1)
    count = D == 2 ? width : width * width
    state >= 3count && return nothing
    image, offset = divrem(state, count)
    shift = (image == 0 ? 0 : image == 1 ? 1 : -1) * p.translation
    anchor = GPUCellGrid.cell_anchor(rows.center, s, eltype(shift)) + shift
    center = cell_coords(anchor, inv(s))
    dy = offset % width - R
    dz = D == 2 ? Int32(0) : offset ÷ width - R
    lower = ntuple(d -> d == 1 ? max(center[1] - R - grid.origin[1], Int32(0)) :
        center[d] - grid.origin[d] + (d == 2 ? dy : dz), Val(D))
    upper_x = min(center[1] + R - grid.origin[1], grid.dims[1] - Int32(1))
    return PeriodicRow(lower, upper_x, shift, Int32(0)), state + Int32(1)
end

# Exact global bounds of positions relative to their assigned cell anchors.
# These are refreshed for each evaluated state, including predictors. They
# account for stale cell assignments without assuming a displacement margin.
# Components 1:3 are coordinate minima; 4:6 are maxima (unused axes are zero).
@inline halo_identity() = SVector(Inf, Inf, Inf, -Inf, -Inf, -Inf)
@inline halo_merge(a, b) = SVector{6,Float64}(ntuple(k -> k <= 3 ? min(a[k], b[k]) : max(a[k], b[k]), 6))

function periodic_halo_kernel!(bounds, positions, cells, gridarg, H, step, n)
    step_active(step) || return nothing
    grid = load_grid(gridarg)
    s = Float64(cell_size(grid, H))
    acc = halo_identity()
    i = thread_index()
    while i <= n
        @inbounds begin
            if cells[i] <= Int32(1)
                # Cell one is reserved for frozen/inactive entries.
                i += blockDim().x * gridDim().x
                continue
            end
            anchor = cell_anchor(global_coords(grid, cells[i]), s, Float64)
            relative = SVector{length(anchor),Float64}(positions[i]) - anchor
            value = SVector{6,Float64}(ntuple(k -> mod1(k, 3) <= length(relative) ? relative[mod1(k, 3)] : 0.0, 6))
            acc = halo_merge(acc, value)
        end
        i += blockDim().x * gridDim().x
    end
    acc = block_reduce(acc, halo_merge)
    threadIdx().x == 1 && (@inbounds bounds[blockIdx().x + 1] = acc)
    return nothing
end

function periodic_halo_finish_kernel!(bounds, step)
    step_active(step) || return nothing
    acc = halo_identity()
    i = threadIdx().x + 1
    while i <= length(bounds)
        @inbounds acc = halo_merge(acc, bounds[i])
        i += blockDim().x
    end
    acc = block_reduce(acc, halo_merge)
    threadIdx().x == 1 && (@inbounds bounds[1] = acc)
    return nothing
end

function launch_periodic_halo!(bounds, positions, cells, grid, H, step)
    isempty(bounds) && return nothing
    @cuda threads=256 blocks=length(bounds)-1 periodic_halo_kernel!(bounds, positions, cells, grid, H, step, Int32(length(positions)))
    @cuda threads=256 blocks=1 periodic_halo_finish_kernel!(bounds, step)
    return nothing
end

struct BoundedPeriodicRows{D,G,P,B}
    grid::G
    reference::SVector{D,Float64}
    size::Float64
    radius::Float64
    periodicity::P
    bounds::B
end

@inline function periodic_image_limits(rows::BoundedPeriodicRows{D}, image::Int32) where {D}
    grid,s,H,p = rows.grid,rows.size,rows.radius,rows.periodicity
    @inbounds halo = rows.bounds[1]
    shift = (image == 0 ? 0 : image == 1 ? 1 : -1) * p.translation
    reference = rows.reference + shift
    lo = ntuple(d -> max(Int32(0),ceil(Int32,(reference[d]-H-halo[d+3])/s)-grid.origin[d]),Val(D))
    hi = ntuple(d -> min(grid.dims[d]-Int32(1),floor(Int32,(reference[d]+H-halo[d])/s)-grid.origin[d]),Val(D))
    return lo,hi
end
@inline function Base.iterate(rows::BoundedPeriodicRows)
    lo,hi = periodic_image_limits(rows,Int32(0))
    return iterate(rows,(Int32(0),Int32(0),lo,hi))
end
@inline function Base.iterate(rows::BoundedPeriodicRows{D}, state::Tuple{Int32,Int32,NTuple{D,Int32},NTuple{D,Int32}}) where {D}
    image,offset,lo,hi = state
    while image < Int32(3)
        wy = max(Int32(0),hi[2]-lo[2]+Int32(1))
        wz = D == 2 ? Int32(1) : max(Int32(0),hi[3]-lo[3]+Int32(1))
        if lo[1] <= hi[1] && offset < wy*wz
            lower = D == 2 ? (lo[1],lo[2]+offset) : (lo[1],lo[2]+offset%wy,lo[3]+offset÷wy)
            shift = (image == 0 ? 0 : image == 1 ? 1 : -1) * rows.periodicity.translation
            return PeriodicRow(lower,hi[1],shift,Int32(0)),(image,offset+Int32(1),lo,hi)
        end
        image += Int32(1)
        image >= Int32(3) && return nothing
        offset = Int32(0)
        lo,hi = periodic_image_limits(rows,image)
    end
    return nothing
end

@inline neighbor_rows(p, grid::CellGrid{D}, c, s, reference, ::Nothing) where {D} = neighbor_rows(p, grid, c, s)
@inline neighbor_rows(::Nothing, grid::CellGrid{D}, c, s, reference, bounds) where {D} = interaction_rows(grid)
@inline neighbor_rows(::Nothing, grid::CellGrid{D}, c, s, reference, ::Nothing) where {D} = interaction_rows(grid)
@inline function neighbor_rows(p, grid::CellGrid{D}, c, s, reference, bounds) where {D}
    # The distance check uses working-precision subtraction and a rounded pair
    # vector. Widen geometric bounds to include its rounding at large offsets.
    padding = 16max(eps(eltype(reference)), eps(typeof(s))) * max(1.0, maximum(abs, reference), maximum(abs, p.translation))
    return BoundedPeriodicRows(grid, SVector{D,Float64}(reference), Float64(s),
        Float64(s) * reach(grid) + padding, p, bounds)
end

@inline neighbor_rows(::Nothing, grid, c, s) = interaction_rows(grid)
@inline neighbor_rows(p, grid, c, s) = PeriodicRows(grid, global_coords(grid, c), typeof(p.lower)(s), p)

@inline GPUCellGrid.row_range(grid::CellGrid{D}, starts, c, row::PeriodicRow{D}) where {D} = periodic_row_range(grid, starts, row)
@inline function periodic_row_range(grid::CellGrid{D}, starts, row::PeriodicRow{D}) where {D}
    lower = row.lower
    valid = lower[1] <= row.upper_x
    for d in 2:D
        valid &= 0 <= lower[d] < grid.dims[d]
    end
    valid || return Int32(1), Int32(0)
    first = GPUCellGrid.linear_local(grid, lower)
    @inbounds return starts[first] + Int32(1), starts[first + row.upper_x - lower[1] + Int32(1)]
end

@inline function GPUCellGrid.pair_vector(p::PeriodicPairSource{D, T}, ref, j::Int32, rowcell, row::PeriodicRow, s) where {D, T}
    @inbounds return SVector{D, T}((ref - p.positions[j]) + row.shift)
end

@inline function GPUCellGrid.pair_vector(p::PeriodicCellPairSource{T}, ref::PeriodicCellPoint{P,C,G,D},
        j::Int32, rowcell, row::PeriodicRow, s) where {T,P,C,G,D}
    @inbounds other = p.cells[j]
    grid = ref.grid
    first_cell = GPUCellGrid.linear_local(grid, row.lower)
    delta = ntuple(d -> ref.coordinates[d] - grid.origin[d] - row.lower[d] -
        (d == 1 ? other.cell - first_cell : Int32(0)), Val(D))
    local_difference = ref.cell.rel - other.rel
    if iszero(row.shift)
        return local_difference + s * SVector{D,T}(delta)
    end
    # Combine the image translation with the anchor offset before rounding;
    # this avoids cancellation between metre-sized Float32 quantities.
    anchor_difference = SVector{D,T}(Float64(s) * SVector{D,Float64}(delta) + row.shift)
    return local_difference + anchor_difference
end

@inline function mdbc_neighborhood(b, A, gp, Pairs, Position, grid, CellStart, s,
                                   Density, ParticleType, SimKernel, m₀, lane, lanes,
                                   Velocity, InvDensity, ::Nothing, periodic_bounds = nothing)
    cg = cell_coords(gp, bin_scale(grid, SimKernel.H⁻¹))
    lg = ntuple(d -> cg[d] - grid.origin[d], length(cg))
    ref = mdbc_reference(Pairs, gp, cg, s)
    return mdbc_rows(b, A, ref, lg, grid, CellStart, Pairs, s, Density, ParticleType,
        SimKernel, m₀, lane, lanes, Velocity, InvDensity)
end

@inline function mdbc_neighborhood(b, A, gp, Pairs, Position, grid::CellGrid{D}, CellStart, s,
                                   Density, ParticleType, SimKernel, m₀, lane, lanes,
                                   Velocity, InvDensity, periodicity::PeriodicBoundary, periodic_bounds = nothing) where {D}
    cg = cell_coords(gp, bin_scale(grid, SimKernel.H⁻¹))
    pairs = PeriodicPairSource(eltype(Density), Position)
    rows = periodic_bounds === nothing ? PeriodicRows(grid, cg, typeof(periodicity.lower)(s), periodicity) :
        neighbor_rows(periodicity, grid, Int32(0), s, gp, periodic_bounds)
    for row in rows
        jlo, jhi = periodic_row_range(grid, CellStart, row)
        b, A = mdbc_range(b, A, jlo, jhi, lane, lanes, gp, pairs, Int32(0), row,
            s, Density, ParticleType, SimKernel, m₀, Velocity, InvDensity)
    end
    return b, A
end
