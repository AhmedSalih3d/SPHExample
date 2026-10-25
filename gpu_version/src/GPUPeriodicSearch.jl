# Included in GPUKernels. Periodic queries translate the source cell's anchor,
# retaining the same stale-list displacement allowance as the ordinary stencil.
# One extra cell handles arbitrary (non-cell-aligned) periodic translations.
struct PeriodicPairSource{D, T, P}
    positions::P
end
PeriodicPairSource(::Type{T}, positions::AbstractVector{SVector{D, TP}}) where {T, D, TP} =
    PeriodicPairSource{D, T, typeof(positions)}(positions)
Adapt.adapt_structure(to, p::PeriodicPairSource{D, T}) where {D, T} =
    PeriodicPairSource(T, Adapt.adapt(to, p.positions))
@inline Base.getindex(p::PeriodicPairSource, i) = @inbounds p.positions[i]

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

@inline function mdbc_neighborhood(b, A, gp, Pairs, Position, grid, CellStart, s,
                                   Density, ParticleType, SimKernel, m₀, lane, lanes,
                                   Velocity, InvDensity, ::Nothing)
    cg = cell_coords(gp, bin_scale(grid, SimKernel.H⁻¹))
    lg = ntuple(d -> cg[d] - grid.origin[d], length(cg))
    ref = mdbc_reference(Pairs, gp, cg, s)
    return mdbc_rows(b, A, ref, lg, grid, CellStart, Pairs, s, Density, ParticleType,
        SimKernel, m₀, lane, lanes, Velocity, InvDensity)
end

@inline function mdbc_neighborhood(b, A, gp, Pairs, Position, grid::CellGrid{D}, CellStart, s,
                                   Density, ParticleType, SimKernel, m₀, lane, lanes,
                                   Velocity, InvDensity, periodicity::PeriodicBoundary) where {D}
    cg = cell_coords(gp, bin_scale(grid, SimKernel.H⁻¹))
    pairs = PeriodicPairSource(eltype(Density), Position)
    rows = PeriodicRows(grid, cg, typeof(periodicity.lower)(s), periodicity)
    for row in rows
        jlo, jhi = periodic_row_range(grid, CellStart, row)
        b, A = mdbc_range(b, A, jlo, jhi, lane, lanes, gp, pairs, Int32(0), row,
            s, Density, ParticleType, SimKernel, m₀, Velocity, InvDensity)
    end
    return b, A
end
