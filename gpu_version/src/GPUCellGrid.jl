"""
GPU cell list (uniform grid neighbour search).

Particles are binned into a dense grid of cubic cells with edge length
`H / R` (`H` the kernel support radius, `R` the *reach* of the grid, a type
parameter). A particle's neighbours then lie in the `(2R+1)^D` cells around
its own: `R = 1` is the classic grid of edge `H` with a 3x3(x3) stencil,
`R = 2` bins at `H/2` with a 5x5(x5) stencil, which scans about 42 % less
volume in 3D (2.5^3 = 15.6 cells of edge `H` instead of 27) at the price of
more, shorter cell ranges. The grid covers the bounding box of all particles
plus an `R` cell margin on every side, so that the rows of neighbouring cells
around any occupied cell are always valid indices. Particles are physically
reordered by cell (counting sort) so that a particle's neighbours are
contiguous in memory.

Cell indices are linear with the first coordinate varying fastest. The
`2R+1` cells `(cx-R .. cx+R, cy, cz)` of one row therefore form one contiguous
particle range (`row_range`), which the interaction kernels exploit: a 3D
particle scans `(2R+1)^2` ranges (9 for `R = 1`, 25 for `R = 2`).

`CellStart` stores exclusive prefix sums with a leading zero: particles of
cell `c` occupy the (1-based) index range `CellStart[c]+1 : CellStart[c+1]`.

Positions may be stored in a higher precision than the rest of the state
(`GPUDoublePosition`: `Float64` positions with `Float32` arithmetic). The pair
loops then never touch the double positions: every particle carries a
`PosCell`, its position relative to the anchor of its cell list cell in the
working precision plus the linear cell index (the `poscell` of DualSPHysics),
and the pair vector is `(relᵢ - relⱼ) + cellsize * (cellᵢ - cellⱼ)`
(`pair_vector`). The cell difference is a small integer known from the row
being scanned, so the pair distance has the precision of the cell edge rather
than of the distance to the origin.
"""
module GPUCellGrid

using CUDA
using StaticArrays
using ..GPUStepState: DeviceStep, StepState, I_STOP, STOP_REBUILD,
                      I_GRID_STATUS, I_GRID_ORIGIN, I_GRID_DIMS, I_GRID_NCELLS

export CellGrid, CellListWorkspace, update_cell_list!, compact_nonzero!, unique_cells_host,
       map_floor, cell_coords, linear_cell, local_coords, row_offsets, row_range, in_grid,
       reach, bin_scale, gather_kernel!, thread_index, load_grid,
       PosCell, CellRow, cell_rows, cell_size, cell_anchor, global_coords, pos_cell, pair_vector,
       prepare_cell_grid!, sync_cell_grid!, consume_cell_grid!, GRID_OK, GRID_NOT_NEEDED, GRID_CAPACITY,
       GRID_NONFINITE, GRID_COORD_OVERFLOW, GRID_CELL_OVERFLOW, GRID_MAX_CELLS

const SORT_THREADS = 256
# Padded launch indices and the final, one-based cell boundary must fit Int32.
const CELL_INDEX_LIMIT = Int(typemax(Int32)) - SORT_THREADS

"""
Return the global 1D thread index as an `Int32`.
"""
@inline thread_index() = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

"""
    map_floor(x, InverseCutOff)

Cell coordinate of position `x`. Identical formula to the CPU version so that
both codes bin particles into the same cells: round to nearest, ties away
from zero.
"""
@inline function map_floor(x::T, InverseCutOff) where {T}
    Int32(sign(x)) * unsafe_trunc(Int32, muladd(abs(x), InverseCutOff, T(0.5)))
end

"""
Uniform grid description: `origin` is the global cell coordinate of the first
grid cell (including the margin) and `dims` the number of cells per axis. The
type parameter `R` is the reach of the neighbour stencil in cells (the cell
edge is `H / R`, see the module documentation); `CellGrid{D}` is `R = 1`.
"""
struct CellGrid{D, R}
    origin::NTuple{D, Int32}
    dims::NTuple{D, Int32}
    ncells::Int32
end

# The grid is built on the device. Simulations return this header through the
# timestep state; standalone rebuilds copy it separately. Bounding-box partials
# stay on the device. GRID_NOT_NEEDED marks skipped conditional preparation.
const GRID_OK             = Int32(0)
const GRID_NOT_NEEDED     = Int32(1)
const GRID_CAPACITY       = Int32(2)
const GRID_NONFINITE      = Int32(3)
const GRID_COORD_OVERFLOW = Int32(4)
const GRID_CELL_OVERFLOW  = Int32(5)
const GRID_MAX_CELLS      = Int32(6)

struct GridBuildState{D, R}
    grid::CellGrid{D, R}
    status::Int32
end

struct BoundingBoxWorkspace{D, T}
    partial::CuVector{T}
    invalid::CuVector{Int32}
    nblocks::Int
end

function BoundingBoxWorkspace{D, T}(n::Integer) where {D, T}
    nblocks = max(1, min(512, cld(n, 256)))
    return BoundingBoxWorkspace{D, T}(CuVector{T}(undef, 2D * nblocks),
                                      CuVector{Int32}(undef, nblocks), nblocks)
end

(::Type{CellGrid{D}})(origin, dims, ncells) where {D} = CellGrid{D, 1}(origin, dims, ncells)
CellGrid{D, R}() where {D, R} = CellGrid{D, R}(ntuple(_ -> Int32(0), Val(D)), ntuple(_ -> Int32(1), Val(D)), Int32(1))
CellGrid{D}() where {D} = CellGrid{D, 1}()

"""
    reach(grid) -> Int32

Number of cells the neighbour stencil extends from a particle's own cell in
every direction (`R`).
"""
@inline reach(::CellGrid{D, R}) where {D, R} = Int32(R)

"""
    bin_scale(grid, InverseCutOff)

Factor that converts a position into a cell coordinate of `grid`: `R / H`.
`R` is a small integer, so the scaled value is exact for `R = 1, 2, 4`.
"""
@inline bin_scale(::CellGrid{D, R}, InverseCutOff::T) where {D, R, T} = T(R) * InverseCutOff

"""
    load_grid(g) -> CellGrid

Grid description for a kernel: either a `CellGrid` passed by value or the
single element of a device vector (`CellListWorkspace.grid_dev`), which lets
a kernel pick up the grid of the latest cell list rebuild without a change
of its launch arguments.
"""
@inline load_grid(g::CellGrid) = g
@inline load_grid(v::AbstractVector{<:CellGrid}) = @inbounds v[1]

#---------------------------------------------------------------
# Preallocated Int32 prefix scan
#---------------------------------------------------------------

const SCAN_THREADS = 256

"""
    IntScanWorkspace(n)

Scratch storage for the block-hierarchical exclusive/inclusive `Int32` scan
used by the cell histogram and stream compaction. CUDA.jl's generic scan
allocates an aggregate vector for long inputs; keeping those vectors here
avoids temporary device allocations during steady-state rebuilds and leaves
all scan pointers stable between graph replays. Host launches still allocate.
"""
mutable struct IntScanWorkspace
    capacity::Int
    sums::Vector{CuVector{Int32}}
    offsets::Vector{CuVector{Int32}}
end

function IntScanWorkspace(n::Integer)
    capacity = max(1, Int(n))
    sums = CuVector{Int32}[]
    offsets = CuVector{Int32}[]
    level_n = capacity
    while true
        nb = cld(level_n, SCAN_THREADS)
        push!(sums, CuVector{Int32}(undef, max(1, nb)))
        push!(offsets, CuVector{Int32}(undef, max(1, nb)))
        nb <= 1 && break
        level_n = nb
    end
    return IntScanWorkspace(capacity, sums, offsets)
end

function ensure_scan_capacity!(ws::IntScanWorkspace, n::Integer)
    n <= ws.capacity && return nothing
    newcap = max(Int(n), max(ws.capacity + 1, (3 * ws.capacity) ÷ 2))
    old_sums = ws.sums
    old_offsets = ws.offsets
    fresh = IntScanWorkspace(newcap)
    ws.capacity = fresh.capacity
    ws.sums = fresh.sums
    ws.offsets = fresh.offsets
    foreach(CUDA.unsafe_free!, old_sums)
    foreach(CUDA.unsafe_free!, old_offsets)
    return nothing
end

@inline function cell_coords(pos::SVector{D, T}, InverseCutOff) where {D, T}
    return ntuple(d -> map_floor(pos[d], InverseCutOff), Val(D))
end

@inline function in_grid(grid::CellGrid{D}, c::NTuple{D, Int32}) where {D}
    ok = true
    for d in 1:D
        l = c[d] - grid.origin[d]
        ok &= (l >= Int32(0)) & (l < grid.dims[d])
    end
    return ok
end

"""
Linear (1-based) cell index of the global cell coordinates `c`. The result is
clamped to the interior of the grid (`R` cells away from the edge, i.e. inside
the margin) so that a corrupt position can never produce an out of bounds
neighbour range.
"""
@inline function linear_cell(grid::CellGrid{2, R}, c::NTuple{2, Int32}) where {R}
    l1 = clamp(c[1] - grid.origin[1], Int32(R), grid.dims[1] - Int32(R + 1))
    l2 = clamp(c[2] - grid.origin[2], Int32(R), grid.dims[2] - Int32(R + 1))
    return Int32(1) + l1 + grid.dims[1] * l2
end

@inline function linear_cell(grid::CellGrid{3, R}, c::NTuple{3, Int32}) where {R}
    l1 = clamp(c[1] - grid.origin[1], Int32(R), grid.dims[1] - Int32(R + 1))
    l2 = clamp(c[2] - grid.origin[2], Int32(R), grid.dims[2] - Int32(R + 1))
    l3 = clamp(c[3] - grid.origin[3], Int32(R), grid.dims[3] - Int32(R + 1))
    return Int32(1) + l1 + grid.dims[1] * (l2 + grid.dims[2] * l3)
end

"""
Linear index from local (0-based, unclamped) coordinates.
"""
@inline linear_local(grid::CellGrid{2}, l::NTuple{2, Int32}) = Int32(1) + l[1] + grid.dims[1] * l[2]
@inline linear_local(grid::CellGrid{3}, l::NTuple{3, Int32}) = Int32(1) + l[1] + grid.dims[1] * (l[2] + grid.dims[2] * l[3])

"""
Local (0-based) coordinates of linear cell index `c`.
"""
@inline function local_coords(grid::CellGrid{2}, c::Int32)
    c0 = c - Int32(1)
    return (c0 % grid.dims[1], c0 ÷ grid.dims[1])
end

@inline function local_coords(grid::CellGrid{3}, c::Int32)
    c0 = c - Int32(1)
    l1 = c0 % grid.dims[1]
    t  = c0 ÷ grid.dims[1]
    return (l1, t % grid.dims[2], t ÷ grid.dims[2])
end

"""
Global cell coordinates of linear cell index `c`.
"""
@inline function global_coords(grid::CellGrid{D}, c::Int32) where {D}
    l = local_coords(grid, c)
    return ntuple(d -> l[d] + grid.origin[d], Val(D))
end

"""
One row of the neighbour stencil: `off` is the linear index offset of the
row's centre cell from the particle's own cell, `dy` and `dz` the shifts of
the row along the second and third axis in cells (`dz = 0` in 2D).
"""
struct CellRow
    off::Int32
    dy::Int32
    dz::Int32
end

"""
    cell_rows(grid)

The neighbouring rows of an interior cell (`2R+1` rows in 2D, `(2R+1)^2` in
3D). Each row holds the `2R+1` consecutive cells `(cx-R .. cx+R)`; `off`
points at the centre cell of the row. For `R = 1` a tuple (the kernels' row
loops unroll), for `R >= 2` a lazy iterator.
"""
@inline function cell_rows(grid::CellGrid{2, 1})
    n1 = grid.dims[1]
    return (CellRow(-n1, Int32(-1), Int32(0)), CellRow(Int32(0), Int32(0), Int32(0)),
            CellRow(n1, Int32(1), Int32(0)))
end

@inline function cell_rows(grid::CellGrid{3, 1})
    n1  = grid.dims[1]
    n12 = grid.dims[1] * grid.dims[2]
    return (CellRow(-n12 - n1, Int32(-1), Int32(-1)), CellRow(-n12, Int32(0), Int32(-1)),
            CellRow(-n12 + n1, Int32(1), Int32(-1)),
            CellRow(-n1, Int32(-1), Int32(0)), CellRow(Int32(0), Int32(0), Int32(0)),
            CellRow(n1, Int32(1), Int32(0)),
            CellRow(n12 - n1, Int32(-1), Int32(1)), CellRow(n12, Int32(0), Int32(1)),
            CellRow(n12 + n1, Int32(1), Int32(1)))
end

"""
Lazy rows of a grid with reach `R >= 2`: `(2R+1)^(D-1)` rows in the same
order as the tuples above (last coordinate outermost). A tuple would unroll
the pair loop body 25 times in 3D, which costs instruction cache; this
iterates with a counter instead.
"""
struct CellRows{D, R}
    n1::Int32
    n12::Int32
end

Base.length(::CellRows{D, R}) where {D, R} = (2R + 1)^(D - 1)
Base.eltype(::Type{<:CellRows}) = CellRow
@inline Base.iterate(r::CellRows) = iterate(r, Int32(0))
@inline function Base.iterate(r::CellRows{D, R}, k::Int32) where {D, R}
    W = Int32(2R + 1)
    k >= W^(D - 1) && return nothing
    dy = D == 2 ? k - Int32(R) : k % W - Int32(R)
    dz = D == 2 ? Int32(0)     : k ÷ W - Int32(R)
    return CellRow(dy * r.n1 + dz * r.n12, dy, dz), k + Int32(1)
end

@inline cell_rows(grid::CellGrid{D, R}) where {D, R} = CellRows{D, R}(grid.dims[1], grid.dims[1] * grid.dims[2])

"""
    row_offsets(grid)

Linear index offsets of the rows of `cell_rows(grid)` (a tuple for `R = 1`,
lazy otherwise).
"""
@inline row_offsets(grid::CellGrid{D, 1}) where {D} = map(r -> r.off, cell_rows(grid))
@inline row_offsets(grid::CellGrid) = Iterators.map(r -> r.off, cell_rows(grid))

"""
    row_range(grid, CellStart, c, off) -> (jlo, jhi)

The (1-based, inclusive) particle index range of the row of `2R+1` cells
centred on cell `c + off`, where `off` is one of `row_offsets(grid)` or a
`CellRow` of `cell_rows(grid)`.
"""
@inline function row_range(grid::CellGrid{D, R}, CellStart, c::Int32, off::Int32) where {D, R}
    row0 = c + off - Int32(R)
    @inbounds jlo = CellStart[row0] + Int32(1)
    @inbounds jhi = CellStart[row0 + Int32(2R + 1)]
    return jlo, jhi
end

@inline row_range(grid::CellGrid, CellStart, c::Int32, row::CellRow) = row_range(grid, CellStart, c, row.off)

#---------------------------------------------------------------
# Cell relative positions (DualSPHysics `poscell`)
#---------------------------------------------------------------

"""
    PosCell{D, T}

Position of a particle relative to the anchor of its cell list cell, in the
working precision `T`, together with the linear index of that cell. Built
from the (possibly higher precision) position by `pos_cell`; pair vectors are
formed with `pair_vector`. The anchor is `cell * cellsize` in the global cell
coordinates, so the relative position is bounded by the cell edge plus the
displacement since the last rebuild, and rounding it to `T` costs an ulp of
the cell edge instead of an ulp of the position itself. In 3D the struct is
16 bytes, the `float4` of DualSPHysics.
"""
struct PosCell{D, T}
    rel::SVector{D, T}
    cell::Int32
end

"""
    cell_size(grid, H) -> H / R

Edge of a cell of `grid` for the support radius `H` (in the precision of `H`;
exact for `R = 1, 2, 4`). The pair vectors of `PosCell`s are formed with this
value, so it must be the same in `pos_cell` and `pair_vector`.
"""
@inline cell_size(::CellGrid{D, R}, H::T) where {D, R, T} = H / T(R)

"""
    cell_anchor(c, s, TP) -> SVector{D, TP}

Anchor point of the cell with global coordinates `c` and edge `s`, in the
position precision `TP`: `c * s` evaluated in `TP`, exact for the cell counts
of any realistic grid.
"""
@inline cell_anchor(c::NTuple{D, Int32}, s::T, ::Type{TP}) where {D, T, TP} =
    SVector{D, TP}(ntuple(d -> TP(c[d]) * TP(s), Val(D)))

"""
    pos_cell(x, c, grid, s) -> PosCell
    pos_cell(x, cg, s)      -> PosCell

Cell relative form of the position `x` for the linear cell `c` of `grid`
(the cell the particle was binned into) and cell edge `s`, or, for a point
that is not a particle (a ghost node), for the global cell coordinates `cg`
(the stored cell index is then meaningless and set to zero).
"""
@inline function pos_cell(x::SVector{D, TP}, c::Int32, grid::CellGrid{D}, s::T) where {D, TP, T}
    return PosCell{D, T}(SVector{D, T}(x - cell_anchor(global_coords(grid, c), s, TP)), c)
end

@inline function pos_cell(x::SVector{D, TP}, cg::NTuple{D, Int32}, s::T) where {D, TP, T}
    return PosCell{D, T}(SVector{D, T}(x - cell_anchor(cg, s, TP)), Int32(0))
end

@inline cell_shift(::Val{2}, δx::Int32, row::CellRow, ::Type{T}) where {T} = SVector{2, T}(T(δx), T(-row.dy))
@inline cell_shift(::Val{3}, δx::Int32, row::CellRow, ::Type{T}) where {T} =
    SVector{3, T}(T(δx), T(-row.dy), T(-row.dz))

"""
    pair_vector(pos, pᵢ, j, rowcell, row, s) -> xᵢ - xⱼ

Vector from particle `j` to the reference `pᵢ` (`pos[i]` for a particle),
where `j` lies in the stencil row `row` and `rowcell` is the linear index of
the cell of that row with the first coordinate of the reference. With plain
positions this is `pᵢ - pos[j]` (`rowcell`, `row` and `s` are ignored). With
`PosCell`s it is `(relᵢ - relⱼ) + s * (cellᵢ - cellⱼ)`: the cell difference is
`(rowcell - cellⱼ, -row.dy, -row.dz)`, at most the reach of the grid per axis,
so the whole computation stays in the working precision.
"""
@inline function pair_vector(Position::AbstractVector{<:SVector}, xᵢ::SVector, j::Int32, rowcell::Int32,
                             row::CellRow, s)
    @inbounds return xᵢ - Position[j]
end

@inline function pair_vector(PosCells::AbstractVector{PosCell{D, T}}, pᵢ::PosCell{D, T}, j::Int32,
                             rowcell::Int32, row::CellRow, s::T) where {D, T}
    @inbounds pⱼ = PosCells[j]
    return (pᵢ.rel - pⱼ.rel) + s * cell_shift(Val(D), rowcell - pⱼ.cell, row, T)
end

"""
Device side buffers of the cell list. Capacities grow on demand when the
bounding box of the particles grows; `generation` counts those
reallocations so that holders of raw device pointers (captured graphs) can
notice. `grid_dev` is a one element device copy of `grid` for the kernels.
The bounding-box reduction and grid validation finish on the device; the host
consumes the status/header through the necessary timestep readback, or through
a dedicated header read for standalone calls, before launching the histogram.
`T` is the element type of the positions that are binned (`Float64` with
`GPUDoublePosition`), used by the bounding box reduction.
"""
mutable struct CellListWorkspace{D, T, R}
    grid::CellGrid{D, R}
    grid_dev::CuVector{CellGrid{D, R}}
    grid_state_dev::CuVector{GridBuildState{D, R}}
    grid_state_host::Vector{GridBuildState{D, R}}
    grid_prepared::Bool
    generation::Int
    CellStart::CuVector{Int32}       # capacity >= ncells + 1
    Counts::CuVector{Int32}          # capacity >= ncells
    Perm::CuVector{Int32}            # length n, new index -> old index
    CellIDScratch::CuVector{Int32}   # length n, cell of particle (old order)
    bbox_ws::BoundingBoxWorkspace{D, T}
    scan_ws::IntScanWorkspace
    max_cells::Int
    deterministic::Bool
    nrebuilds::Int
    grid_status_readbacks::Int
    capacity_growths::Int
    bbox_host_readbacks::Int
end

"""
    CellListWorkspace{D, T}(n; reach = 1, max_cells, deterministic)

Cell list buffers for `n` particles whose positions are `SVector{D, T}`.
`reach` selects the grid: cells of edge `H / reach` and a `(2 reach + 1)^D`
stencil (see the module documentation).
"""
function CellListWorkspace{D, T}(n::Integer; reach::Integer = 1, max_cells::Integer = 50_000_000,
                                 deterministic::Bool = true) where {D, T}
    reach >= 1 || throw(ArgumentError("the cell list reach must be at least 1, got $reach"))
    reach <= typemax(Int32) || throw(ArgumentError("the cell list reach is too large: $reach"))
    0 <= n <= CELL_INDEX_LIMIT || throw(ArgumentError("the cell-list particle count must be between 0 and $CELL_INDEX_LIMIT, got $n"))
    max_cells >= 1 || throw(ArgumentError("GPUMaxCells must be at least 1, got $max_cells"))
    R = Int(reach)
    bbox_ws = BoundingBoxWorkspace{D, T}(n)
    grid = CellGrid{D, R}()
    counts_capacity = 1024
    grid_state = GridBuildState{D, R}(grid, GRID_NOT_NEEDED)
    return CellListWorkspace{D, T, R}(
        grid,
        CuArray([grid]),
        CuArray([grid_state]),
        [grid_state],
        false,
        0,
        CuVector{Int32}(undef, counts_capacity + 1),
        CuVector{Int32}(undef, counts_capacity),
        CuVector{Int32}(undef, n),
        CuVector{Int32}(undef, n),
        bbox_ws,
        IntScanWorkspace(max(n, counts_capacity)),
        Int(max_cells),
        deterministic,
        0,
        0,
        0,
        0,
    )
end

#---------------------------------------------------------------
# Kernels
#---------------------------------------------------------------

@inline function rounded_coordinate_valid(x, inv_cell)
    q = muladd(abs(x), inv_cell, typeof(x)(0.5))
    q_limit = typeof(q)(typemax(Int32)) + one(q)
    return isfinite(q) & (q >= zero(q)) & (q < q_limit)
end

@inline bbox_finite(b::SVector{4, T}) where {T} =
    isfinite(b[1]) & isfinite(b[2]) & isfinite(b[3]) & isfinite(b[4])
@inline bbox_finite(b::SVector{6, T}) where {T} =
    isfinite(b[1]) & isfinite(b[2]) & isfinite(b[3]) &
    isfinite(b[4]) & isfinite(b[5]) & isfinite(b[6])
@inline bbox_finite(b::NTuple{4, T}) where {T} =
    isfinite(b[1]) & isfinite(b[2]) & isfinite(b[3]) & isfinite(b[4])
@inline bbox_finite(b::NTuple{6, T}) where {T} =
    isfinite(b[1]) & isfinite(b[2]) & isfinite(b[3]) &
    isfinite(b[4]) & isfinite(b[5]) & isfinite(b[6])

@inline publish_grid_state!(::Nothing, state) = nothing
@inline function publish_grid_state!(step::DeviceStep, state::GridBuildState{D}) where {D}
    @inbounds begin
        step.i[I_GRID_STATUS] = state.status
        for d in 1:3
            step.i[I_GRID_ORIGIN + d - 1] = d <= D ? state.grid.origin[d] : Int32(0)
            step.i[I_GRID_DIMS + d - 1] = d <= D ? state.grid.dims[d] : Int32(1)
        end
        step.i[I_GRID_NCELLS] = state.grid.ncells
    end
    return nothing
end

@inline function store_grid_state!(state_dev, grid, status, step)
    state = GridBuildState(grid, status)
    @inbounds state_dev[1] = state
    publish_grid_state!(step, state)
    return nothing
end

@inline function write_grid_result!(grid_dev, state_dev, bbox,
                                    inv_cell, max_cells::Int64, capacity::Int64,
                                    reach_i::Int32, step, ::Val{D}, ::Val{R}) where {D, R}
    old_grid = load_grid(grid_dev)
    status = GRID_OK

    if !bbox_finite(bbox)
        status = GRID_NONFINITE
        store_grid_state!(state_dev, old_grid, status, step)
        return nothing
    end

    valid = isfinite(inv_cell) & (inv_cell > zero(inv_cell))
    for d in 1:D
        valid &= rounded_coordinate_valid(bbox[d], inv_cell)
        valid &= rounded_coordinate_valid(bbox[D + d], inv_cell)
    end
    if !valid
        store_grid_state!(state_dev, old_grid, GRID_COORD_OVERFLOW, step)
        return nothing
    end

    cmin = ntuple(d -> Int64(map_floor(bbox[d], inv_cell)) - Int64(R), Val(D))
    cmax = ntuple(d -> Int64(map_floor(bbox[D + d], inv_cell)) + Int64(R), Val(D))
    coord_limit_lo = Int64(typemin(Int32)) + Int64(R)
    coord_limit_hi = Int64(typemax(Int32)) - Int64(R)
    for d in 1:D
        valid &= (cmin[d] >= coord_limit_lo) & (cmax[d] <= coord_limit_hi)
    end
    if !valid
        store_grid_state!(state_dev, old_grid, GRID_COORD_OVERFLOW, step)
        return nothing
    end

    dims64 = ntuple(d -> cmax[d] - cmin[d] + Int64(1), Val(D))
    ncells64 = Int64(1)
    overflow = false
    int32_limit = Int64(CELL_INDEX_LIMIT)
    for d in 1:D
        dim = dims64[d]
        if dim <= Int64(0) || dim > int32_limit || ncells64 > int32_limit ÷ dim
            overflow = true
        else
            ncells64 *= dim
        end
    end
    if overflow
        store_grid_state!(state_dev, old_grid, GRID_CELL_OVERFLOW, step)
        return nothing
    end
    if ncells64 > max_cells
        store_grid_state!(state_dev, old_grid, GRID_MAX_CELLS, step)
        return nothing
    end

    dims = ntuple(d -> Int32(dims64[d]), Val(D))
    origin = ntuple(d -> Int32(cmin[d]), Val(D))
    grid = CellGrid{D, R}(origin, dims, Int32(ncells64))
    status = ncells64 > capacity ? GRID_CAPACITY : GRID_OK
    @inbounds grid_dev[1] = grid
    store_grid_state!(state_dev, grid, status, step)
    return nothing
end

@inline grid_build_needed(::Nothing) = true
@inline grid_build_needed(step::DeviceStep) = @inbounds step.i[I_STOP] == STOP_REBUILD

function bbox_reduce_scalar_kernel!(partial, invalid, Position, n::Int32,
                                    lo_init::T, hi_init::T, step, ::Val{2}) where {T}
    grid_build_needed(step) || return nothing
    i = Int64(thread_index())
    stride = Int64(blockDim().x) * Int64(gridDim().x)
    lo1 = lo_init; lo2 = lo_init; hi1 = hi_init; hi2 = hi_init
    bad = zero(T)
    @inbounds while i <= n
        j = Int64(2) * (i - Int64(1)) + Int64(1)
        x = Position[j]; y = Position[j + Int64(1)]
        bad = (isfinite(x) & isfinite(y)) ? bad : one(T)
        lo1 = min(lo1, x); lo2 = min(lo2, y)
        hi1 = max(hi1, x); hi2 = max(hi2, y)
        i += stride
    end
    tid = threadIdx().x
    B = 256
    shared = CuStaticSharedArray(T, 5 * B)
    @inbounds begin
        shared[tid] = lo1; shared[B + tid] = lo2
        shared[2B + tid] = hi1; shared[3B + tid] = hi2
        shared[4B + tid] = bad
    end
    sync_threads()
    s = Int32(B ÷ 2)
    while s >= Int32(1)
        if tid <= s
            @inbounds begin
                shared[tid] = min(shared[tid], shared[tid + s])
                shared[B + tid] = min(shared[B + tid], shared[B + tid + s])
                shared[2B + tid] = max(shared[2B + tid], shared[2B + tid + s])
                shared[3B + tid] = max(shared[3B + tid], shared[3B + tid + s])
                shared[4B + tid] = max(shared[4B + tid], shared[4B + tid + s])
            end
        end
        sync_threads()
        s ÷= Int32(2)
    end
    if tid == Int32(1)
        base = (blockIdx().x - Int32(1)) * Int32(4)
        @inbounds begin
            partial[base + 1] = shared[1]; partial[base + 2] = shared[B + 1]
            partial[base + 3] = shared[2B + 1]; partial[base + 4] = shared[3B + 1]
            invalid[blockIdx().x] = Int32(shared[4B + 1])
        end
    end
    return nothing
end

function bbox_reduce_scalar_kernel!(partial, invalid, Position, n::Int32,
                                    lo_init::T, hi_init::T, step, ::Val{3}) where {T}
    grid_build_needed(step) || return nothing
    i = Int64(thread_index())
    stride = Int64(blockDim().x) * Int64(gridDim().x)
    lo1 = lo_init; lo2 = lo_init; lo3 = lo_init
    hi1 = hi_init; hi2 = hi_init; hi3 = hi_init
    bad = zero(T)
    @inbounds while i <= n
        j = Int64(3) * (i - Int64(1)) + Int64(1)
        x = Position[j]; y = Position[j + Int64(1)]; z = Position[j + Int64(2)]
        bad = (isfinite(x) & isfinite(y) & isfinite(z)) ? bad : one(T)
        lo1 = min(lo1, x); lo2 = min(lo2, y); lo3 = min(lo3, z)
        hi1 = max(hi1, x); hi2 = max(hi2, y); hi3 = max(hi3, z)
        i += stride
    end
    tid = threadIdx().x
    B = 256
    shared = CuStaticSharedArray(T, 7 * B)
    @inbounds begin
        shared[tid] = lo1; shared[B + tid] = lo2; shared[2B + tid] = lo3
        shared[3B + tid] = hi1; shared[4B + tid] = hi2; shared[5B + tid] = hi3
        shared[6B + tid] = bad
    end
    sync_threads()
    s = Int32(B ÷ 2)
    while s >= Int32(1)
        if tid <= s
            @inbounds begin
                shared[tid] = min(shared[tid], shared[tid + s])
                shared[B + tid] = min(shared[B + tid], shared[B + tid + s])
                shared[2B + tid] = min(shared[2B + tid], shared[2B + tid + s])
                shared[3B + tid] = max(shared[3B + tid], shared[3B + tid + s])
                shared[4B + tid] = max(shared[4B + tid], shared[4B + tid + s])
                shared[5B + tid] = max(shared[5B + tid], shared[5B + tid + s])
                shared[6B + tid] = max(shared[6B + tid], shared[6B + tid + s])
            end
        end
        sync_threads()
        s ÷= Int32(2)
    end
    if tid == Int32(1)
        base = (blockIdx().x - Int32(1)) * Int32(6)
        @inbounds begin
            partial[base + 1] = shared[1]; partial[base + 2] = shared[B + 1]
            partial[base + 3] = shared[2B + 1]; partial[base + 4] = shared[3B + 1]
            partial[base + 5] = shared[4B + 1]; partial[base + 6] = shared[5B + 1]
            invalid[blockIdx().x] = Int32(shared[6B + 1])
        end
    end
    return nothing
end

function finish_grid_metadata_kernel!(grid_dev, state_dev, partial, invalid, nblocks::Int32,
                                      lo_init::T, hi_init::T, inv_cell,
                                      max_cells::Int64, capacity::Int64, reach_i::Int32,
                                      step, ::Val{2}, ::Val{R}) where {T, R}
    tid = threadIdx().x
    if !grid_build_needed(step)
        if tid == Int32(1)
            store_grid_state!(state_dev, load_grid(grid_dev), GRID_NOT_NEEDED, step)
        end
        return nothing
    end
    lo1 = lo_init; lo2 = lo_init; hi1 = hi_init; hi2 = hi_init
    b = tid
    @inbounds while b <= nblocks
        base = (b - Int32(1)) * Int32(4)
        lo1 = min(lo1, partial[base + 1]); lo2 = min(lo2, partial[base + 2])
        hi1 = max(hi1, partial[base + 3]); hi2 = max(hi2, partial[base + 4])
        b += blockDim().x
    end
    B = 256
    shared = CuStaticSharedArray(T, 4 * B)
    @inbounds begin
        shared[tid] = lo1; shared[B + tid] = lo2
        shared[2B + tid] = hi1; shared[3B + tid] = hi2
    end
    sync_threads()
    s = Int32(B ÷ 2)
    while s >= Int32(1)
        if tid <= s
            @inbounds begin
                shared[tid] = min(shared[tid], shared[tid + s])
                shared[B + tid] = min(shared[B + tid], shared[B + tid + s])
                shared[2B + tid] = max(shared[2B + tid], shared[2B + tid + s])
                shared[3B + tid] = max(shared[3B + tid], shared[3B + tid + s])
            end
        end
        sync_threads()
        s ÷= Int32(2)
    end
    tid == Int32(1) || return nothing
    @inbounds for b in Int32(1):nblocks
        if invalid[b] != Int32(0)
            store_grid_state!(state_dev, load_grid(grid_dev), GRID_NONFINITE, step)
            return nothing
        end
    end
    bbox = (shared[1], shared[B + 1], shared[2B + 1], shared[3B + 1])
    write_grid_result!(grid_dev, state_dev, bbox, inv_cell, max_cells,
                       capacity, reach_i, step, Val(2), Val(R))
    return nothing
end

function finish_grid_metadata_kernel!(grid_dev, state_dev, partial, invalid, nblocks::Int32,
                                      lo_init::T, hi_init::T, inv_cell,
                                      max_cells::Int64, capacity::Int64, reach_i::Int32,
                                      step, ::Val{3}, ::Val{R}) where {T, R}
    tid = threadIdx().x
    if !grid_build_needed(step)
        if tid == Int32(1)
            store_grid_state!(state_dev, load_grid(grid_dev), GRID_NOT_NEEDED, step)
        end
        return nothing
    end
    lo1 = lo_init; lo2 = lo_init; lo3 = lo_init
    hi1 = hi_init; hi2 = hi_init; hi3 = hi_init
    b = tid
    @inbounds while b <= nblocks
        base = (b - Int32(1)) * Int32(6)
        lo1 = min(lo1, partial[base + 1]); lo2 = min(lo2, partial[base + 2]); lo3 = min(lo3, partial[base + 3])
        hi1 = max(hi1, partial[base + 4]); hi2 = max(hi2, partial[base + 5]); hi3 = max(hi3, partial[base + 6])
        b += blockDim().x
    end
    B = 256
    shared = CuStaticSharedArray(T, 6 * B)
    @inbounds begin
        shared[tid] = lo1; shared[B + tid] = lo2; shared[2B + tid] = lo3
        shared[3B + tid] = hi1; shared[4B + tid] = hi2; shared[5B + tid] = hi3
    end
    sync_threads()
    s = Int32(B ÷ 2)
    while s >= Int32(1)
        if tid <= s
            @inbounds begin
                shared[tid] = min(shared[tid], shared[tid + s])
                shared[B + tid] = min(shared[B + tid], shared[B + tid + s])
                shared[2B + tid] = min(shared[2B + tid], shared[2B + tid + s])
                shared[3B + tid] = max(shared[3B + tid], shared[3B + tid + s])
                shared[4B + tid] = max(shared[4B + tid], shared[4B + tid + s])
                shared[5B + tid] = max(shared[5B + tid], shared[5B + tid + s])
            end
        end
        sync_threads()
        s ÷= Int32(2)
    end
    tid == Int32(1) || return nothing
    @inbounds for b in Int32(1):nblocks
        if invalid[b] != Int32(0)
            store_grid_state!(state_dev, load_grid(grid_dev), GRID_NONFINITE, step)
            return nothing
        end
    end
    bbox = (shared[1], shared[B + 1], shared[2B + 1], shared[3B + 1],
            shared[4B + 1], shared[5B + 1])
    write_grid_result!(grid_dev, state_dev, bbox, inv_cell, max_cells,
                       capacity, reach_i, step, Val(3), Val(R))
    return nothing
end

# One block scans 256 values, stores the block totals, and writes either the
# exclusive or inclusive result.  A second scan over the block totals supplies
# the offsets for all following blocks.
function scan_block_kernel!(out, input, sums, n::Int32, inclusive::Bool)
    tid = threadIdx().x
    i = (blockIdx().x - Int32(1)) * blockDim().x + tid
    value = Int32(0)
    if i <= n
        @inbounds value = input[i]
    end
    shared = CuStaticSharedArray(Int32, SCAN_THREADS)
    @inbounds shared[tid] = value
    sync_threads()

    offset = Int32(1)
    while offset < Int32(SCAN_THREADS)
        add = Int32(0)
        if tid > offset
            @inbounds add = shared[tid - offset]
        end
        sync_threads()
        @inbounds shared[tid] += add
        sync_threads()
        offset <<= Int32(1)
    end

    if i <= n
        if inclusive
            @inbounds out[i] = shared[tid]
        else
            @inbounds out[i] = shared[tid] - value
        end
    end
    if tid == Int32(SCAN_THREADS)
        @inbounds sums[blockIdx().x] = shared[tid]
    end
    return nothing
end

function scan_add_offsets_kernel!(out, offsets, n::Int32)
    i = thread_index()
    i > n && return nothing
    block = (i - Int32(1)) ÷ Int32(SCAN_THREADS) + Int32(1)
    @inbounds out[i] += offsets[block]
    return nothing
end

function scan_level!(ws::IntScanWorkspace, out, input, n::Int, level::Int, inclusive::Bool)
    n == 0 && return nothing
    blocks = cld(n, SCAN_THREADS)
    sums = ws.sums[level]
    @cuda threads=SCAN_THREADS blocks=blocks scan_block_kernel!(out, input, sums,
                                                                Int32(n), inclusive)
    if blocks > 1
        offsets = ws.offsets[level]
        scan_level!(ws, offsets, sums, blocks, level + 1, false)
        @cuda threads=SCAN_THREADS blocks=blocks scan_add_offsets_kernel!(out, offsets, Int32(n))
    end
    return nothing
end

function scan_int32!(ws::IntScanWorkspace, out, input; inclusive::Bool = false)
    n = length(input)
    length(out) == n || throw(DimensionMismatch("scan input and output lengths differ"))
    n == 0 && return out
    ensure_scan_capacity!(ws, n)
    scan_level!(ws, out, input, n, 1, inclusive)
    return out
end

# Cell of every particle and histogram of cell occupation.
function cellid_hist_kernel!(CellIDScratch, Counts, Position, InverseCutOff, gridarg, n::Int32)
    i = thread_index()
    i > n && return nothing
    @inbounds begin
        grid = load_grid(gridarg)
        c = linear_cell(grid, cell_coords(Position[i], InverseCutOff))
        CellIDScratch[i] = c
        CUDA.atomic_add!(pointer(Counts, c), Int32(1))
    end
    return nothing
end

# Counting sort scatter: every particle claims a slot inside its cell range.
function scatter_kernel!(Perm, Counts, CellStart, CellIDScratch, n::Int32)
    i = thread_index()
    i > n && return nothing
    @inbounds begin
        c    = CellIDScratch[i]
        k    = CUDA.atomic_add!(pointer(Counts, c), Int32(1))
        slot = CellStart[c] + k + Int32(1)
        Perm[slot] = i
    end
    return nothing
end

# Insertion sort of the particle indices inside every cell. This makes the
# ordering independent of the (non deterministic) order in which atomics
# were served, so repeated runs produce bit identical results.
function cell_sort_kernel!(Perm, CellStart, gridarg)
    grid = load_grid(gridarg)
    c = thread_index()
    c > grid.ncells && return nothing
    @inbounds begin
        lo = CellStart[c] + Int32(1)
        hi = CellStart[c + Int32(1)]
        k = lo + Int32(1)
        while k <= hi
            v = Perm[k]
            m = k - Int32(1)
            while m >= lo && Perm[m] > v
                Perm[m + Int32(1)] = Perm[m]
                m -= Int32(1)
            end
            Perm[m + Int32(1)] = v
            k += Int32(1)
        end
    end
    return nothing
end

@inline gather_fields!(i, old, ::Tuple{}, ::Tuple{}) = nothing
@inline function gather_fields!(i, old, srcs::Tuple, dsts::Tuple)
    @inbounds dsts[1][i] = srcs[1][old]
    gather_fields!(i, old, Base.tail(srcs), Base.tail(dsts))
end

"""
Permute every array in `srcs` into the matching array of `dsts` following
`perm` (`dst[i] = src[perm[i]]`). One launch for all fields.
"""
function gather_kernel!(perm, srcs::Tuple, dsts::Tuple, n::Int32)
    i = thread_index()
    i > n && return nothing
    @inbounds old = perm[i]
    gather_fields!(i, old, srcs, dsts)
    return nothing
end

# Stream compaction: flag the non-zero entries, rank them with an inclusive
# scan and scatter every flagged index to its rank.
function nonzero_flag_kernel!(flags, x, predicate, n::Int32)
    i = thread_index()
    i > n && return nothing
    @inbounds flags[i] = Int32(predicate(x[i]))
    return nothing
end

function compact_kernel!(out, flags, ranks, n::Int32)
    i = thread_index()
    i > n && return nothing
    @inbounds flagged = flags[i] != Int32(0)
    if flagged
        # bounds checked on purpose: a rank beyond `out` means the caller's
        # count of non-zero entries is wrong
        @inbounds r = ranks[i]
        out[r] = i
    end
    return nothing
end

#---------------------------------------------------------------
# Host driver
#---------------------------------------------------------------

function ensure_capacity!(ws::CellListWorkspace, ncells::Integer)
    1 <= ncells <= min(ws.max_cells, CELL_INDEX_LIMIT) ||
        throw(ArgumentError("cell capacity must be between 1 and $(min(ws.max_cells, CELL_INDEX_LIMIT)), got $ncells"))
    if length(ws.Counts) < ncells
        old_capacity = length(ws.Counts)
        new_capacity = min(CELL_INDEX_LIMIT, max(Int(ncells), max(old_capacity + 1, (3 * old_capacity) ÷ 2)))
        # The caller has synchronized the previous users. Allocate replacements
        # before releasing the current buffers so allocation failure is safe.
        cell_start = CuVector{Int32}(undef, new_capacity + 1)
        counts = CuVector{Int32}(undef, new_capacity)
        ensure_scan_capacity!(ws.scan_ws, max(length(ws.Perm), new_capacity))
        old_start, old_counts = ws.CellStart, ws.Counts
        ws.CellStart, ws.Counts = cell_start, counts
        ws.generation += 1
        ws.capacity_growths += 1
        CUDA.unsafe_free!(old_start)
        CUDA.unsafe_free!(old_counts)
    end
    return nothing
end

function grid_build_error(ws::CellListWorkspace, state::GridBuildState)
    status = state.status
    if status == GRID_NONFINITE
        error("Non-finite particle position encountered while building the cell list.")
    elseif status == GRID_COORD_OVERFLOW
        error("Particle position is outside the representable cell-coordinate range while building the cell list.")
    elseif status == GRID_CELL_OVERFLOW
        error("Cell grid dimensions or cell count overflow the Int32 cell-list index range.")
    elseif status == GRID_MAX_CELLS
        error("Cell grid would need more than the allowed $(ws.max_cells) cells. " *
              "A particle probably escaped the domain. Increase `GPUMaxCells` if this is expected.")
    end
    error("Unknown device cell-grid status $(status).")
end

"""
    prepare_cell_grid!(ws, Position, InverseCutOff; step = nothing)

Launch the device bounding-box reduction and grid calculation. No host data is
read here. With `step`, preparation runs only when the timestep stop flag asks
for a rebuild and publishes the status/header into its existing integer state.
"""
function prepare_cell_grid!(ws::CellListWorkspace{D, T, R},
                            Position::CuVector{SVector{D, T}}, InverseCutOff;
                            step::Union{Nothing, StepState} = nothing) where {D, T, R}
    n = length(Position)
    1 <= n <= CELL_INDEX_LIMIT || error("The cell-list particle count must be between 1 and $CELL_INDEX_LIMIT.")
    n == length(ws.Perm) || throw(DimensionMismatch("positions and cell-list workspace particle counts differ"))
    ws.grid_prepared = false
    inv_cell = bin_scale(CellGrid{D, R}(), InverseCutOff)
    PositionScalar = reinterpret(T, Position)
    lo = T(Inf)
    hi = T(-Inf)
    @cuda threads=256 blocks=ws.bbox_ws.nblocks bbox_reduce_scalar_kernel!(
        ws.bbox_ws.partial, ws.bbox_ws.invalid, PositionScalar, Int32(n), lo, hi, step, Val(D))
    @cuda threads=256 blocks=1 finish_grid_metadata_kernel!(
        ws.grid_dev, ws.grid_state_dev, ws.bbox_ws.partial, ws.bbox_ws.invalid,
        Int32(ws.bbox_ws.nblocks), lo, hi, inv_cell, Int64(ws.max_cells),
        Int64(length(ws.Counts)), Int32(R), step, Val(D), Val(R))
    return nothing
end

"""
    sync_cell_grid!(ws) -> Bool

Copy and consume the one-element device grid status after a standalone
preparation launch. This adds a dedicated synchronization; the simulation
driver instead uses `consume_cell_grid!` after its necessary timestep readback.
"""
function sync_cell_grid!(ws::CellListWorkspace{D, T, R}) where {D, T, R}
    copyto!(ws.grid_state_host, ws.grid_state_dev)
    ws.grid_status_readbacks += 1
    state = @inbounds ws.grid_state_host[1]
    return consume_cell_grid!(ws, state)
end

function consume_cell_grid!(ws::CellListWorkspace, state::GridBuildState)
    ws.grid_prepared = false
    status = state.status
    status == GRID_NOT_NEEDED && return false
    if status == GRID_OK || status == GRID_CAPACITY
        1 <= Int(state.grid.ncells) <= min(ws.max_cells, CELL_INDEX_LIMIT) ||
            error("The prepared cell grid has an invalid cell count.")
        status == GRID_CAPACITY && ensure_capacity!(ws, Int(state.grid.ncells))
        Int(state.grid.ncells) <= length(ws.Counts) || error("The prepared cell grid exceeds the allocated capacity.")
        ws.grid = state.grid
        ws.grid_prepared = true
        return true
    end
    grid_build_error(ws, state)
end

"""
    consume_cell_grid!(ws, host_step_integers) -> Bool

Validate and consume the header returned by the existing timestep readback,
updating host geometry and growing capacity before histogram/scatter launches.
The caller must have completed `readback!(step)` after conditional preparation.
This function does not transfer data or synchronize the device. Capacity
growth changes `generation`; callers must invalidate their captured graphs.
"""
function consume_cell_grid!(ws::CellListWorkspace{D, T, R}, ih::AbstractVector{Int32}) where {D, T, R}
    length(ih) >= I_GRID_NCELLS || throw(DimensionMismatch("the timestep header has no cell-grid state"))
    origin = ntuple(d -> ih[I_GRID_ORIGIN + d - 1], Val(D))
    dims = ntuple(d -> ih[I_GRID_DIMS + d - 1], Val(D))
    grid = CellGrid{D, R}(origin, dims, ih[I_GRID_NCELLS])
    return consume_cell_grid!(ws, GridBuildState{D, R}(grid, ih[I_GRID_STATUS]))
end

"""
    update_cell_list!(ws, Position, InverseCutOff, srcs, dsts; prepared = false) -> grid

Rebuild the cell list from `Position` (device array) and reorder the arrays
in `srcs` into `dsts` by cell. `srcs`/`dsts` are tuples of device arrays of
equal length; the caller is expected to swap them afterwards. The sorted
cell id of every particle is written to `dsts[end]` when `srcs[end]` is the
workspace scratch cell id array, so include `(ws.CellIDScratch => CellID)`
as the last pair. `InverseCutOff` is used in its own precision (the working
precision), so positions of a higher precision are binned exactly like the
ghost nodes in the kernels.

With `prepared = true`, the caller has already prepared and consumed a valid
grid header, so no dedicated grid-status synchronization is performed. The
enqueue phase can be captured with fixed grid dimensions and stable buffers;
conditional preparation and host capacity growth remain outside that capture.
"""
function update_cell_list!(ws::CellListWorkspace{D, T, R}, Position::CuVector{SVector{D, T}},
                           InverseCutOff, srcs::Tuple, dsts::Tuple; prepared::Bool = false) where {D, T, R}
    n = length(Position)
    n == length(ws.Perm) || throw(DimensionMismatch("positions and cell-list workspace particle counts differ"))

    if !prepared
        prepare_cell_grid!(ws, Position, InverseCutOff)
        sync_cell_grid!(ws)
    end
    ws.grid_prepared || error("The cell grid must be prepared and its valid status consumed before rebuilding.")
    ws.grid_prepared = false

    grid = ws.grid
    ncells = grid.ncells
    ncells > Int32(0) || error("The device returned an empty cell grid.")
    inv_cell = bin_scale(grid, InverseCutOff)
    Counts    = view(ws.Counts, 1:ncells)

    threads = SORT_THREADS
    blocks  = cld(n, threads)

    fill!(Counts, Int32(0))
    @cuda threads=threads blocks=blocks cellid_hist_kernel!(ws.CellIDScratch, ws.Counts, Position,
                                                             inv_cell, ws.grid_dev, Int32(n))

    # Exclusive prefix sum with leading zero.
    fill!(view(ws.CellStart, 1:1), Int32(0))
    scan_int32!(ws.scan_ws, view(ws.CellStart, 2:(ncells + 1)), Counts; inclusive = true)

    fill!(Counts, Int32(0))
    @cuda threads=threads blocks=blocks scatter_kernel!(ws.Perm, ws.Counts, ws.CellStart,
                                                         ws.CellIDScratch, Int32(n))

    if ws.deterministic
        @cuda threads=threads blocks=cld(ncells, threads) cell_sort_kernel!(ws.Perm, ws.CellStart,
                                                                               ws.grid_dev)
    end

    @cuda threads=threads blocks=blocks gather_kernel!(ws.Perm, srcs, dsts, Int32(n))

    ws.nrebuilds += 1
    return grid
end

"""
    compact_nonzero!(ws, x, out; predicate = !iszero) -> out

Write the (1-based, ascending) indices of the non-zero entries of the device
vector `x` into `out`, an `Int32` vector whose length must equal their
number. `x` must have one entry per particle. Uses the `Perm` and
`CellIDScratch` buffers of `ws` as scratch space, so call it only after
`update_cell_list!` is done with them, i.e. after the reorder. The driver
uses it to list the boundary particles that own a ghost node in cell order
after every rebuild, so that the mDBC kernel is launched over those alone.
Pass `predicate` to select entries by a different condition.
"""
function compact_nonzero!(ws::CellListWorkspace, x::CuVector, out::CuVector{Int32};
                          predicate = !iszero)
    n = length(x)
    n == length(ws.Perm) || throw(ArgumentError("`x` must have one entry per particle"))
    n == 0 && return out
    flags   = ws.CellIDScratch
    ranks   = ws.Perm
    threads = SORT_THREADS
    blocks  = cld(n, threads)
    @cuda threads=threads blocks=blocks nonzero_flag_kernel!(flags, x, predicate, Int32(n))
    scan_int32!(ws.scan_ws, ranks, flags; inclusive = true)
    @cuda threads=threads blocks=blocks compact_kernel!(out, flags, ranks, Int32(n))
    return out
end

"""
    unique_cells_host(grid, CellID_host) -> Vector{CartesianIndex{D}}

Global cell coordinates of the occupied cells, in ascending linear order, from
the (sorted) per particle cell ids copied to the host. Used for the optional
cell grid export.
"""
function unique_cells_host(grid::CellGrid{D}, CellID_host::AbstractVector{Int32}) where {D}
    cells = CartesianIndex{D}[]
    isempty(CellID_host) && return cells
    last = Int32(-1)
    for c in CellID_host
        if c != last
            l = local_coords(grid, c)
            push!(cells, CartesianIndex(ntuple(d -> Int(l[d] + grid.origin[d]), Val(D))))
            last = c
        end
    end
    return cells
end

end # module
