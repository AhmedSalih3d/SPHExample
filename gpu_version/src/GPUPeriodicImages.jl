# Dense, cell-sorted periodic sources. Only the original targets are integrated.
mutable struct PeriodicImageWorkspace{A,C,P,O,IC}
    raw::A
    sorted::A
    tags::CuVector{Int32}
    targets::CuVector{Int32}
    cells::CuVector{Int32}
    pairs::P
    count::CuVector{Int32}
    invalid::CuVector{Int32}
    builds::CuVector{Int32}
    order::O
    inverse_density::IC
    pressure::IC
    cl::C
    initialized::Bool
end

function PeriodicImageWorkspace{D,T,TP}(n; reach=1, max_cells=50_000_000, deterministic=true,
        preserve_pair_order=false, cache_fields=false) where {D,T,TP}
    0 < n <= GPUCellGrid.CELL_INDEX_LIMIT ÷ 2 || throw(ArgumentError("Periodic source capacity exceeds Int32 indexing"))
    arrays() = (; Position=CuVector{SVector{D,TP}}(undef,2n),
        Velocity=CuVector{SVector{D,T}}(undef,2n), Density=CuVector{T}(undef,2n),
        Type=CuVector{ParticleType}(undef,2n))
    raw, sorted = arrays(), arrays()
    cl = CellListWorkspace{D,TP}(2n; reach, max_cells, deterministic)
    return PeriodicImageWorkspace(raw,sorted,CuVector{Int32}(undef,2n),CuVector{Int32}(undef,n),
        CuVector{Int32}(undef,2n),CuVector{PosCell{D,T}}(undef,2n),
        CuArray(Int32[n]),CUDA.zeros(Int32,1),CUDA.zeros(Int32,1),
        preserve_pair_order ? (; owners=CuVector{Int32}(undef,2n), cells=CuVector{Int32}(undef,2n)) : nothing,
        cache_fields ? CuVector{T}(undef,2n) : nothing,
        cache_fields ? CuVector{T}(undef,2n) : nothing,cl,false)
end

# Grow outside graph capture. Retaining an enclosing grid keeps graph launch
# dimensions and pointers stable while the original particle grid shrinks.
function configure_periodic_images!(images,periodicity,original_grid,kernel)
    D = length(original_grid.dims)
    R = Int(GPUCellGrid.reach(images.cl.grid))
    s = Float64(cell_size(original_grid,kernel.H))
    lower = ntuple(d -> (original_grid.origin[d]+GPUCellGrid.reach(original_grid)-1)*s - 2kernel.H,D)
    upper = ntuple(d -> (original_grid.origin[d]+original_grid.dims[d]-GPUCellGrid.reach(original_grid))*s + 2kernel.H,D)
    bounds = (; domain_min=lower,domain_max=upper)
    # The initializer adds transverse image margins itself.
    if !images.initialized
        return initialize_periodic_images!(images,periodicity,bounds,kernel)
    end
    grid = images.cl.grid
    scale = R*kernel.H⁻¹
    coordinate_limit = Float64(typemax(Int32))-2R-4
    all(d -> abs((lower[d]-abs(periodicity.translation[d]))*scale) < coordinate_limit &&
        abs((upper[d]+abs(periodicity.translation[d]))*scale) < coordinate_limit,1:D) ||
        throw(ArgumentError("Periodic force-source grid coordinates exceed Int32 indexing"))
    lo = ntuple(d -> d == periodicity.axis ? grid.origin[d] :
        min(grid.origin[d],GPUCellGrid.map_floor(bounds.domain_min[d]-abs(periodicity.translation[d]),scale)-Int32(R+1)),D)
    hi = ntuple(d -> d == periodicity.axis ? grid.origin[d]+grid.dims[d]-Int32(1) :
        max(grid.origin[d]+grid.dims[d]-Int32(1),GPUCellGrid.map_floor(bounds.domain_max[d]+abs(periodicity.translation[d]),scale)+Int32(R+1)),D)
    dims64 = ntuple(d -> Int64(hi[d])-Int64(lo[d])+1,D)
    ncells = prod(Float64.(dims64))
    0 < ncells <= min(images.cl.max_cells,GPUCellGrid.CELL_INDEX_LIMIT) ||
        throw(ArgumentError("Periodic force-source grid exceeds GPUMaxCells"))
    fresh = typeof(grid)(lo,Int32.(dims64),Int32(ncells))
    if fresh != grid
        GPUCellGrid.ensure_capacity!(images.cl,Int(ncells))
        images.cl.grid = fresh
        copyto!(images.cl.grid_dev,[fresh])
        images.cl.generation += 1
    end
    return nothing
end

function initialize_periodic_images!(images, periodicity, scheme, kernel)
    images.initialized && return nothing
    periodicity === nothing && throw(ArgumentError("Periodic images require a periodic boundary"))
    D = length(images.cl.grid.dims)
    # A particle can own at most one image when the two face strips do not overlap.
    periodicity.upper - periodicity.lower > 2kernel.H ||
        throw(ArgumentError("Periodic images require a period greater than twice the support radius"))
    all(d -> isfinite(scheme.domain_min[d]) && isfinite(scheme.domain_max[d]) &&
        scheme.domain_min[d] < scheme.domain_max[d],1:D) ||
        throw(ArgumentError("Periodic images require finite, ordered domain bounds"))
    R = Int(GPUCellGrid.reach(images.cl.grid))
    scale = R * kernel.H⁻¹
    lower = ntuple(D) do d
        d == periodicity.axis ? Float64(periodicity.lower) - kernel.H :
            scheme.domain_min[d] - abs(periodicity.translation[d])
    end
    upper = ntuple(D) do d
        d == periodicity.axis ? Float64(periodicity.upper) + kernel.H :
            scheme.domain_max[d] + abs(periodicity.translation[d])
    end
    coordinate_limit = Float64(typemax(Int32)) - 2R - 4
    all(d -> abs(lower[d]*scale) < coordinate_limit && abs(upper[d]*scale) < coordinate_limit,1:D) ||
        throw(ArgumentError("Periodic source grid coordinates exceed Int32 indexing"))
    lo = ntuple(d -> GPUCellGrid.map_floor(lower[d],scale) - Int32(R+1),D)
    hi = ntuple(d -> GPUCellGrid.map_floor(upper[d],scale) + Int32(R+1),D)
    dims64 = ntuple(d -> Int64(hi[d]) - Int64(lo[d]) + 1,D)
    limit = min(images.cl.max_cells,GPUCellGrid.CELL_INDEX_LIMIT)
    ncells = prod(Float64.(dims64))
    all(>(0),dims64) && 0 < ncells <= limit ||
        throw(ArgumentError("Periodic source grid requires $ncells cells; increase GPUMaxCells or narrow the domain bounds"))
    dims = Int32.(dims64)
    grid = typeof(images.cl.grid)(lo,dims,Int32(ncells))
    GPUCellGrid.ensure_capacity!(images.cl,Int(ncells))
    images.cl.grid = grid
    copyto!(images.cl.grid_dev,[grid])
    images.initialized = true
    images.cl.generation += 1
    return nothing
end

@inline function periodic_source_inside(x,grid,scale)
    valid = all(isfinite,x)
    for d in 1:length(x)
        # Check before converting coordinates to Int32.
        value = x[d] * scale
        valid &= grid.origin[d] + GPUCellGrid.reach(grid) + 0.5 <= value <=
            grid.origin[d] + grid.dims[d] - GPUCellGrid.reach(grid) - 1.5
    end
    return valid
end

function periodic_reset_kernel!(count,invalid,builds,n,step)
    step_active(step) || return nothing
    @inbounds count[1],invalid[1],builds[1] = n,Int32(0),builds[1]+Int32(1)
    return nothing
end

function periodic_status_kernel!(invalid,step)
    @inbounds if step_active(step) && invalid[1] != 0
        step.i[I_STOP] = STOP_PERIODIC_IMAGES
    end
    return nothing
end
publish_periodic_status!(invalid,::HostStep) = nothing
function publish_periodic_status!(invalid,step::StepState)
    @cuda threads=1 blocks=1 periodic_status_kernel!(invalid,step)
    return nothing
end

function periodic_sources_kernel!(raw,tags,count,invalid,x,v,rho,types,periodicity,H,grid,scale,n,step)
    step_active(step) || return nothing
    i = thread_index()
    i > n && return nothing
    @inbounds begin
        kind = types[i]
        point = kind == Frozen ? x[i] : wrap_periodic_position(x[i],periodicity)
        if kind != Frozen && !periodic_source_inside(point,grid,scale)
            CUDA.atomic_xchg!(pointer(invalid),Int32(1))
            kind = Frozen
        end
        raw.Position[i],raw.Velocity[i],raw.Density[i],raw.Type[i] = point,v[i],rho[i],kind
        tags[i] = i
        if kind != Frozen
            axial = point[periodicity.axis]
            direction = axial < periodicity.lower + H ? -1 : axial > periodicity.upper - H ? 1 : 0
            if direction != 0
                copy_point = point + direction * periodicity.translation
                slot = CUDA.atomic_add!(pointer(count),Int32(1)) + Int32(1)
                copy_kind = kind
                if !periodic_source_inside(copy_point,grid,scale)
                    CUDA.atomic_xchg!(pointer(invalid),Int32(1))
                    copy_kind = Frozen
                end
                raw.Position[slot],raw.Velocity[slot],raw.Density[slot],raw.Type[slot] = copy_point,v[i],rho[i],copy_kind
                tags[slot] = n+i
            end
        end
    end
    return nothing
end

function periodic_hist_kernel!(ids,counts,raw,count,scale,grid,capacity,step)
    step_active(step) || return nothing
    i = thread_index()
    @inbounds if i <= capacity && i <= count[1]
        c = raw.Type[i] == Frozen ? Int32(1) : GPUCellGrid.linear_cell(grid,GPUCellGrid.cell_coords(raw.Position[i],scale))
        ids[i] = c
        CUDA.atomic_add!(pointer(counts,c),Int32(1))
    end
    return nothing
end

function periodic_scatter_kernel!(perm,counts,starts,ids,count,capacity,step)
    step_active(step) || return nothing
    i = thread_index()
    @inbounds if i <= capacity && i <= count[1]
        c = ids[i]
        slot = starts[c] + CUDA.atomic_add!(pointer(counts,c),Int32(1)) + Int32(1)
        perm[slot] = i
    end
    return nothing
end

function periodic_sort_kernel!(perm,starts,tags,grid,step)
    step_active(step) || return nothing
    c = thread_index()
    # Inactive particles share the reserved corner cell; its order is irrelevant.
    (c <= 1 || c > grid.ncells) && return nothing
    @inbounds begin
        lo,hi = starts[c]+Int32(1),starts[c+Int32(1)]
        k = lo+Int32(1)
        while k <= hi
            value = perm[k]
            m = k-Int32(1)
            while m >= lo && tags[perm[m]] > tags[value]
                perm[m+Int32(1)] = perm[m]
                m -= Int32(1)
            end
            perm[m+Int32(1)] = value
            k += Int32(1)
        end
    end
    return nothing
end

function periodic_gather_kernel!(sorted,pairs,cells,targets,raw,tags,perm,ids,count,grid,s,n,capacity,
        order,original_cells,inverse_density,pressure,constants,step)
    step_active(step) || return nothing
    i = thread_index()
    @inbounds if i <= capacity && i <= count[1]
        old = perm[i]
        GPUCellGrid.gather_fields!(i,old,Tuple(raw),Tuple(sorted))
        c = ids[old]
        cells[i] = c
        pairs[i] = pos_cell(raw.Position[old],c,grid,s)
        tag = tags[old]
        tag <= n && (targets[tag] = i)
        if order !== nothing
            owner = tag > n ? tag-n : tag
            order.owners[i] = owner
            order.cells[i] = original_cells[owner]
        end
        if inverse_density !== nothing
            density = sorted.Density[i]
            GPUKernels.store_inverse_density!(inverse_density,i,density)
            GPUKernels.store_pressure!(pressure,i,EquationOfStateGamma7(density,constants.c₀,constants.ρ₀))
        end
    end
    return nothing
end

function build_periodic_images!(images,x,v,rho,types,periodicity,scheme,kernel;
        original_cells=nothing,constants=nothing,step=HostStep(0.0,0.0))
    # The initial batch stops for an original-grid rebuild before any force
    # evaluation. Its graph preparation must remain a harmless no-op.
    !images.initialized && scheme === nothing && return nothing
    images.initialized || initialize_periodic_images!(images,periodicity,scheme,kernel)
    images.order === nothing || original_cells !== nothing || throw(ArgumentError("Periodic sources require original cell IDs for reference pair orientation"))
    images.inverse_density === nothing || constants !== nothing || throw(ArgumentError("Cached periodic sources require simulation constants"))
    n,capacity = length(types),length(images.tags)
    cl = images.cl
    grid = cl.grid
    scale = GPUCellGrid.bin_scale(grid,kernel.H⁻¹)
    threads = 256
    @cuda threads=1 blocks=1 periodic_reset_kernel!(images.count,images.invalid,images.builds,Int32(n),step)
    @cuda threads=threads blocks=cld(n,threads) periodic_sources_kernel!(images.raw,images.tags,
        images.count,images.invalid,x,v,rho,types,periodicity,kernel.H,grid,scale,Int32(n),step)
    publish_periodic_status!(images.invalid,step)
    counts = view(cl.Counts,1:grid.ncells)
    fill!(counts,Int32(0))
    @cuda threads=threads blocks=cld(capacity,threads) periodic_hist_kernel!(cl.CellIDScratch,cl.Counts,
        images.raw,images.count,scale,grid,Int32(capacity),step)
    fill!(view(cl.CellStart,1:1),Int32(0))
    GPUCellGrid.scan_int32!(cl.scan_ws,view(cl.CellStart,2:grid.ncells+1),counts;inclusive=true)
    fill!(counts,Int32(0))
    @cuda threads=threads blocks=cld(capacity,threads) periodic_scatter_kernel!(cl.Perm,cl.Counts,
        cl.CellStart,cl.CellIDScratch,images.count,Int32(capacity),step)
    if cl.deterministic
        @cuda threads=threads blocks=cld(grid.ncells,threads) periodic_sort_kernel!(cl.Perm,cl.CellStart,images.tags,grid,step)
    end
    @cuda threads=threads blocks=cld(capacity,threads) periodic_gather_kernel!(images.sorted,images.pairs,
        images.cells,images.targets,images.raw,images.tags,cl.Perm,cl.CellIDScratch,images.count,
        grid,cell_size(grid,kernel.H),Int32(n),Int32(capacity),images.order,original_cells,
        images.inverse_density,images.pressure,constants,step)
    return nothing
end
