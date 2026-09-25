"""
GPU cell list (uniform grid neighbour search).

Particles are binned into a dense grid of cubic cells with edge length `H`
(the kernel support radius). The grid covers the bounding box of all
particles plus a one cell margin on every side, so that the 3 (2D) or 9 (3D)
rows of three neighbouring cells around any occupied cell are always valid
indices. Particles are physically reordered by cell (counting sort) so that
a particle's neighbours are contiguous in memory.

Cell indices are linear with the first coordinate varying fastest. The three
cells `(cx-1, cy, cz)`, `(cx, cy, cz)`, `(cx+1, cy, cz)` therefore form one
contiguous particle range, which the interaction kernels exploit.

`CellStart` stores exclusive prefix sums with a leading zero: particles of
cell `c` occupy the (1-based) index range `CellStart[c]+1 : CellStart[c+1]`.
"""
module GPUCellGrid

using CUDA
using StaticArrays
using ..GPUReductions

export CellGrid, CellListWorkspace, update_cell_list!, unique_cells_host,
       map_floor, cell_coords, linear_cell, local_coords, row_offsets, in_grid,
       gather_kernel!, thread_index

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
grid cell (including the margin) and `dims` the number of cells per axis.
"""
struct CellGrid{D}
    origin::NTuple{D, Int32}
    dims::NTuple{D, Int32}
    ncells::Int32
end

CellGrid{D}() where {D} = CellGrid{D}(ntuple(_ -> Int32(0), Val(D)), ntuple(_ -> Int32(1), Val(D)), Int32(1))

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
clamped to the interior of the grid (one cell away from the margin) so that a
corrupt position can never produce an out of bounds neighbour range.
"""
@inline function linear_cell(grid::CellGrid{2}, c::NTuple{2, Int32})
    l1 = clamp(c[1] - grid.origin[1], Int32(1), grid.dims[1] - Int32(2))
    l2 = clamp(c[2] - grid.origin[2], Int32(1), grid.dims[2] - Int32(2))
    return Int32(1) + l1 + grid.dims[1] * l2
end

@inline function linear_cell(grid::CellGrid{3}, c::NTuple{3, Int32})
    l1 = clamp(c[1] - grid.origin[1], Int32(1), grid.dims[1] - Int32(2))
    l2 = clamp(c[2] - grid.origin[2], Int32(1), grid.dims[2] - Int32(2))
    l3 = clamp(c[3] - grid.origin[3], Int32(1), grid.dims[3] - Int32(2))
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
Linear index offsets of the neighbouring rows of an interior cell. Each row
holds three consecutive cells `(cx-1 .. cx+1)`; the offset points at the
centre cell of the row.
"""
@inline row_offsets(grid::CellGrid{2}) = (-grid.dims[1], Int32(0), grid.dims[1])

@inline function row_offsets(grid::CellGrid{3})
    n1  = grid.dims[1]
    n12 = grid.dims[1] * grid.dims[2]
    return (-n12 - n1, -n12, -n12 + n1, -n1, Int32(0), n1, n12 - n1, n12, n12 + n1)
end

"""
Device side buffers of the cell list. Capacities grow on demand when the
bounding box of the particles grows.
"""
mutable struct CellListWorkspace{D, T, R <: ReductionWorkspace}
    grid::CellGrid{D}
    CellStart::CuVector{Int32}       # capacity >= ncells + 1
    Counts::CuVector{Int32}          # capacity >= ncells
    Perm::CuVector{Int32}            # length n, new index -> old index
    CellIDScratch::CuVector{Int32}   # length n, cell of particle (old order)
    bbox_ws::R
    max_cells::Int
    deterministic::Bool
    nrebuilds::Int
end

function CellListWorkspace{D, T}(n::Integer; max_cells::Integer = 50_000_000,
                                 deterministic::Bool = true) where {D, T}
    bbox_ws = ReductionWorkspace{SVector{2D, T}}(n)
    return CellListWorkspace{D, T, typeof(bbox_ws)}(
        CellGrid{D}(),
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
function update_cell_list!(ws::CellListWorkspace{D, T}, Position::CuVector{SVector{D, T}},
                           InverseCutOff, srcs::Tuple, dsts::Tuple) where {D, T}
    n = length(Position)

    # Bounding box of all particles -> grid with one cell margin.
    init = SVector{2D, T}(ntuple(k -> k <= D ? T(Inf) : T(-Inf), Val(2D)))
    bbox = reduce_svector(ws.bbox_ws, bbox_map, bbox_reduce, init, n, Position)
    all(isfinite, bbox) || error("Non-finite particle position encountered while building the cell list.")

    cmin = ntuple(d -> map_floor(bbox[d],     InverseCutOff) - Int32(1), Val(D))
    cmax = ntuple(d -> map_floor(bbox[D + d], InverseCutOff) + Int32(1), Val(D))
    dims = ntuple(d -> cmax[d] - cmin[d] + Int32(1), Val(D))
    ncells_big = prod(Int64.(dims))
    if ncells_big > ws.max_cells
        error("Cell grid would need $(ncells_big) cells (dims = $(dims)), more than the " *
              "allowed $(ws.max_cells). A particle probably escaped the domain. " *
              "Increase `GPUMaxCells` in SimulationMetaData if this is expected.")
    end
    ncells = Int32(ncells_big)
    grid   = CellGrid{D}(cmin, dims, ncells)
    ws.grid = grid

    ensure_capacity!(ws, ncells)
    Counts    = view(ws.Counts, 1:ncells)
    CellStart = view(ws.CellStart, 1:(ncells + 1))

    threads = SORT_THREADS
    blocks  = cld(n, threads)

    fill!(Counts, Int32(0))
    @cuda threads=threads blocks=blocks cellid_hist_kernel!(ws.CellIDScratch, ws.Counts, Position,
                                                             T(InverseCutOff), grid, Int32(n))

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
