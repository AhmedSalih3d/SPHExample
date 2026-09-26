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
"""
module GPUCellGrid

using CUDA
using StaticArrays
using ..GPUReductions

export CellGrid, CellListWorkspace, update_cell_list!, compact_nonzero!, unique_cells_host,
       map_floor, cell_coords, linear_cell, local_coords, row_offsets, row_range, in_grid,
       reach, bin_scale, gather_kernel!, thread_index, load_grid

const SORT_THREADS = 256

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
Linear index offsets of the neighbouring rows of an interior cell (`2R+1` rows
in 2D, `(2R+1)^2` in 3D). Each row holds the `2R+1` consecutive cells
`(cx-R .. cx+R)`; the offset points at the centre cell of the row. For
`R = 1` a tuple (the kernels' row loops unroll), for `R >= 2` a lazy iterator.
"""
@inline row_offsets(grid::CellGrid{2, 1}) = (-grid.dims[1], Int32(0), grid.dims[1])

@inline function row_offsets(grid::CellGrid{3, 1})
    n1  = grid.dims[1]
    n12 = grid.dims[1] * grid.dims[2]
    return (-n12 - n1, -n12, -n12 + n1, -n1, Int32(0), n1, n12 - n1, n12, n12 + n1)
end

"""
Lazy row offsets of a grid with reach `R >= 2`: `(2R+1)^(D-1)` offsets in
the same order as the tuples above (last coordinate outermost). A tuple
would unroll the pair loop body 25 times in 3D, which costs instruction
cache; this iterates with a counter instead.
"""
struct RowOffsets{D, R}
    n1::Int32
    n12::Int32
end

Base.length(::RowOffsets{D, R}) where {D, R} = (2R + 1)^(D - 1)
Base.eltype(::Type{<:RowOffsets}) = Int32
@inline Base.iterate(r::RowOffsets) = iterate(r, Int32(0))
@inline function Base.iterate(r::RowOffsets{D, R}, k::Int32) where {D, R}
    W = Int32(2R + 1)
    k >= W^(D - 1) && return nothing
    off = D == 2 ? (k - Int32(R)) * r.n1 : (k ÷ W - Int32(R)) * r.n12 + (k % W - Int32(R)) * r.n1
    return off, k + Int32(1)
end

@inline row_offsets(grid::CellGrid{D, R}) where {D, R} = RowOffsets{D, R}(grid.dims[1], grid.dims[1] * grid.dims[2])

"""
    row_range(grid, CellStart, c, off) -> (jlo, jhi)

The (1-based, inclusive) particle index range of the row of `2R+1` cells
centred on cell `c + off`, where `off` is one of `row_offsets(grid)`.
"""
@inline function row_range(grid::CellGrid{D, R}, CellStart, c::Int32, off::Int32) where {D, R}
    row0 = c + off - Int32(R)
    @inbounds jlo = CellStart[row0] + Int32(1)
    @inbounds jhi = CellStart[row0 + Int32(2R + 1)]
    return jlo, jhi
end

"""
Device side buffers of the cell list. Capacities grow on demand when the
bounding box of the particles grows; `generation` counts those
reallocations so that holders of raw device pointers (captured graphs) can
notice. `grid_dev` is a one element device copy of `grid` for the kernels.
"""
mutable struct CellListWorkspace{D, T, R, W <: ReductionWorkspace}
    grid::CellGrid{D, R}
    grid_dev::CuVector{CellGrid{D, R}}
    generation::Int
    CellStart::CuVector{Int32}       # capacity >= ncells + 1
    Counts::CuVector{Int32}          # capacity >= ncells
    Perm::CuVector{Int32}            # length n, new index -> old index
    CellIDScratch::CuVector{Int32}   # length n, cell of particle (old order)
    bbox_ws::W
    max_cells::Int
    deterministic::Bool
    nrebuilds::Int
end

"""
    CellListWorkspace{D, T}(n; reach = 1, max_cells, deterministic)

Cell list buffers for `n` particles. `reach` selects the grid: cells of edge
`H / reach` and a `(2 reach + 1)^D` stencil (see the module documentation).
"""
function CellListWorkspace{D, T}(n::Integer; reach::Integer = 1, max_cells::Integer = 50_000_000,
                                 deterministic::Bool = true) where {D, T}
    reach >= 1 || throw(ArgumentError("the cell list reach must be at least 1, got $reach"))
    R = Int(reach)
    bbox_ws = ReductionWorkspace{SVector{2D, T}}(n)
    return CellListWorkspace{D, T, R, typeof(bbox_ws)}(
        CellGrid{D, R}(),
        CuArray([CellGrid{D, R}()]),
        0,
        CuVector{Int32}(undef, 1024),
        CuVector{Int32}(undef, 1024),
        CuVector{Int32}(undef, n),
        CuVector{Int32}(undef, n),
        bbox_ws,
        Int(max_cells),
        deterministic,
        0,
    )
end

#---------------------------------------------------------------
# Kernels
#---------------------------------------------------------------

@inline function bbox_map(i, Position)
    @inbounds p = Position[i]
    return vcat(p, p)
end

@inline function bbox_reduce(a::SVector{N, T}, b::SVector{N, T}) where {N, T}
    D = N ÷ 2
    return SVector{N, T}(ntuple(k -> k <= D ? min(a[k], b[k]) : max(a[k], b[k]), Val(N)))
end

# Cell of every particle and histogram of cell occupation.
function cellid_hist_kernel!(CellIDScratch, Counts, Position, InverseCutOff, grid, n::Int32)
    i = thread_index()
    i > n && return nothing
    @inbounds begin
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
function cell_sort_kernel!(Perm, CellStart, ncells::Int32)
    c = thread_index()
    c > ncells && return nothing
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
function nonzero_flag_kernel!(flags, x, n::Int32)
    i = thread_index()
    i > n && return nothing
    @inbounds flags[i] = Int32(!iszero(x[i]))
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
    if length(ws.CellStart) < ncells + 1
        newlen = max(ncells + 1, ceil(Int, 1.5 * length(ws.CellStart)))
        CUDA.unsafe_free!(ws.CellStart)
        CUDA.unsafe_free!(ws.Counts)
        ws.CellStart = CuVector{Int32}(undef, newlen)
        ws.Counts    = CuVector{Int32}(undef, newlen)
        ws.generation += 1
    end
    return nothing
end

"""
    update_cell_list!(ws, Position, InverseCutOff, srcs, dsts) -> grid

Rebuild the cell list from `Position` (device array) and reorder the arrays
in `srcs` into `dsts` by cell. `srcs`/`dsts` are tuples of device arrays of
equal length; the caller is expected to swap them afterwards. The sorted
cell id of every particle is written to `dsts[end]` when `srcs[end]` is the
workspace scratch cell id array, so include `(ws.CellIDScratch => CellID)`
as the last pair.
"""
function update_cell_list!(ws::CellListWorkspace{D, T, R}, Position::CuVector{SVector{D, T}},
                           InverseCutOff, srcs::Tuple, dsts::Tuple) where {D, T, R}
    n = length(Position)

    # Cells have edge H / R: bin with R / H (exact for R = 1, 2).
    inv_cell = bin_scale(ws.grid, T(InverseCutOff))

    # Bounding box of all particles -> grid with an R cell margin.
    init = SVector{2D, T}(ntuple(k -> k <= D ? T(Inf) : T(-Inf), Val(2D)))
    bbox = reduce_svector(ws.bbox_ws, bbox_map, bbox_reduce, init, n, Position)
    all(isfinite, bbox) || error("Non-finite particle position encountered while building the cell list.")

    cmin = ntuple(d -> map_floor(bbox[d],     inv_cell) - Int32(R), Val(D))
    cmax = ntuple(d -> map_floor(bbox[D + d], inv_cell) + Int32(R), Val(D))
    dims = ntuple(d -> cmax[d] - cmin[d] + Int32(1), Val(D))
    ncells_big = prod(Int64.(dims))
    if ncells_big > ws.max_cells
        error("Cell grid would need $(ncells_big) cells (dims = $(dims)), more than the " *
              "allowed $(ws.max_cells). A particle probably escaped the domain. " *
              "Increase `GPUMaxCells` in SimulationMetaData if this is expected.")
    end
    ncells = Int32(ncells_big)
    grid   = CellGrid{D, R}(cmin, dims, ncells)
    ws.grid = grid
    fill!(ws.grid_dev, grid)

    ensure_capacity!(ws, ncells)
    Counts    = view(ws.Counts, 1:ncells)
    CellStart = view(ws.CellStart, 1:(ncells + 1))

    threads = SORT_THREADS
    blocks  = cld(n, threads)

    fill!(Counts, Int32(0))
    @cuda threads=threads blocks=blocks cellid_hist_kernel!(ws.CellIDScratch, ws.Counts, Position,
                                                             inv_cell, grid, Int32(n))

    # Exclusive prefix sum with leading zero.
    fill!(view(ws.CellStart, 1:1), Int32(0))
    accumulate!(+, view(ws.CellStart, 2:(ncells + 1)), Counts)

    fill!(Counts, Int32(0))
    @cuda threads=threads blocks=blocks scatter_kernel!(ws.Perm, ws.Counts, ws.CellStart,
                                                         ws.CellIDScratch, Int32(n))

    if ws.deterministic
        @cuda threads=threads blocks=cld(ncells, threads) cell_sort_kernel!(ws.Perm, ws.CellStart, ncells)
    end

    @cuda threads=threads blocks=blocks gather_kernel!(ws.Perm, srcs, dsts, Int32(n))

    ws.nrebuilds += 1
    return grid
end

"""
    compact_nonzero!(ws, x, out) -> out

Write the (1-based, ascending) indices of the non-zero entries of the device
vector `x` into `out`, an `Int32` vector whose length must equal their
number. `x` must have one entry per particle. Uses the `Perm` and
`CellIDScratch` buffers of `ws` as scratch space, so call it only after
`update_cell_list!` is done with them, i.e. after the reorder. The driver
uses it to list the boundary particles that own a ghost node in cell order
after every rebuild, so that the mDBC kernel is launched over those alone.
"""
function compact_nonzero!(ws::CellListWorkspace, x::CuVector, out::CuVector{Int32})
    n = length(x)
    n == length(ws.Perm) || throw(ArgumentError("`x` must have one entry per particle"))
    n == 0 && return out
    flags   = ws.CellIDScratch
    ranks   = ws.Perm
    threads = SORT_THREADS
    blocks  = cld(n, threads)
    @cuda threads=threads blocks=blocks nonzero_flag_kernel!(flags, x, Int32(n))
    accumulate!(+, ranks, flags)
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
