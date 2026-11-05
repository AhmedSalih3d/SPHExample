"""Native one-way planar open-channel boundaries with GPU particle emission."""
module OpenBoundaries
using CUDA,StaticArrays,LinearAlgebra
using ..SimulationGeometry
using ..SimulationConstantsConfiguration
using ..GPUCellGrid
using ..GPUKernels: store_pos_cell!,ELEMENTWISE_THREADS
using ..GPUStepState
using ..SPHKernels

export OpenChannel,open_channel_geometry,prepare_open_boundary,launch_open_boundary_values!,
       launch_open_boundary_events!,open_boundary_state

"""
    OpenChannel{T}(; Inlet, Outlet, WaterLevel, Velocity, Spacing, Duration,
        InletLayers=5, OutletLayers=8, FluidMarker=1, InletMarker=2,
        OutletMarker=3, ReserveMarker=4)

One-way X-directed 2D flow with prescribed velocity and zeroth-order ghost
density extrapolation. `open_channel_geometry` expands the seed lines into
buffer layers and preallocates inactive inlet particles for `Duration`.
Emission, conversion and retirement stay on CUDA; opposite ends have no
periodic neighbour interactions. Pass the configuration as `SimOpenBoundaries`.
Currently requires nonperiodic, expanding-grid 2D DBC and standard stepping.
Inactive reserves use `Frozen`; distinguish them from lost physical particles.
"""
struct OpenChannel{T}
    inlet::Float64
    outlet::Float64
    water_level::Float64
    velocity::T
    spacing::Float64
    duration::Float64
    inlet_layers::Int32
    outlet_layers::Int32
    fluid_marker::Int32
    inlet_marker::Int32
    outlet_marker::Int32
    reserve_marker::Int32
end

function OpenChannel{T}(;Inlet,Outlet,WaterLevel,Velocity,Spacing,Duration,InletLayers = 5,OutletLayers = 8,
        FluidMarker = 1,InletMarker = 2,OutletMarker = 3,ReserveMarker = 4) where {T<:AbstractFloat}
    left,right,level,duration = Float64.((Inlet,Outlet,WaterLevel,Duration))
    speed,spacing = T(Velocity),Float64(Spacing)
    all(isfinite,(left,right,level,duration,speed,spacing)) && right > left && speed > 0 && spacing > 0 && duration > 0 ||
        throw(ArgumentError("open-channel bounds must be finite, Outlet > Inlet, and velocity/spacing/duration positive"))
    markers = Int32.((FluidMarker,InletMarker,OutletMarker,ReserveMarker))
    all(>(0),markers) && length(unique(markers)) == 4 || throw(ArgumentError("open-channel markers must be positive and distinct"))
    InletLayers >= 2 && OutletLayers >= 2 || throw(ArgumentError("open boundaries need at least two buffer layers"))
    return OpenChannel(left,right,level,speed,spacing,duration,Int32(InletLayers),Int32(OutletLayers),markers...)
end

function open_channel_geometry(geometry::Vector{SPHGeometry{2,T}},flow::OpenChannel{T}) where T
    any(g -> g.GroupMarker == flow.reserve_marker,geometry) && throw(ArgumentError("ReserveMarker already belongs to input geometry"))
    incoming = only(filter(g -> g.GroupMarker == flow.inlet_marker && g.Type == Fluid,geometry))
    outgoing = only(filter(g -> g.GroupMarker == flow.outlet_marker && g.Type == Fluid,geometry))
    select(g) = findall(p -> p[2] <= flow.water_level+1e-6*flow.spacing,g.Particles.Position)
    inlet_rows,outlet_rows = select(incoming),select(outgoing)
    isempty(inlet_rows) && throw(ArgumentError("inlet seed line has no points below WaterLevel"))
    length(inlet_rows) == length(outlet_rows) || throw(ArgumentError("this one-way channel requires matching inlet/outlet row counts"))
    all(p -> abs(p[1]-flow.inlet) < 1e-5*flow.spacing,incoming.Particles.Position) &&
        all(p -> abs(p[1]-flow.outlet) < 1e-5*flow.spacing,outgoing.Particles.Position) || throw(ArgumentError("seed-line positions differ from the configured inlet/outlet"))
    vector = SVector{2,T}(flow.velocity,0)
    function region(points,density,marker,kind)
        velocity = fill(kind == Fluid ? vector : zero(vector),length(points))
        return SPHGeometry{2,T}(Particles = particle_struct_array(SVector{2,Float64}.(points),T(1000);
            Density = T.(density),Velocity = velocity),GroupMarker = Int(marker),Type = kind)
    end
    result = SPHGeometry{2,T}[]
    for g in geometry
        if g.Type == Fluid && g.GroupMarker in (flow.inlet_marker,flow.outlet_marker)
            active = select(g)
            inactive = setdiff(eachindex(g.Particles.Position),active)
            push!(result,region(g.Particles.Position[active],g.Particles.Density[active],g.GroupMarker,Fluid))
            isempty(inactive) || push!(result,region(g.Particles.Position[inactive],g.Particles.Density[inactive],g.GroupMarker,Frozen))
        elseif g.Type == Fluid
            push!(result,region(g.Particles.Position,g.Particles.Density,g.GroupMarker,Fluid))
        else
            push!(result,g)
        end
    end
    for (g,indices,layers,sign) in ((incoming,inlet_rows,flow.inlet_layers,-1),(outgoing,outlet_rows,flow.outlet_layers,1))
        points = [SVector{2,Float64}(g.Particles.Position[i])+SVector(sign*Float64(flow.spacing)*layer,0.) for layer in 1:layers-1 for i in indices]
        densities = [g.Particles.Density[i] for _ in 1:layers-1 for i in indices]
        push!(result,region(points,densities,g.GroupMarker,Fluid))
    end
    events = ceil(Int,flow.duration*flow.velocity/flow.spacing)+1
    events*length(inlet_rows) <= typemax(Int32) || throw(ArgumentError("inlet reserve does not fit Int32 indexing"))
    points = [SVector{2,Float64}(incoming.Particles.Position[i])-SVector(Float64(flow.spacing)*(flow.inlet_layers-1),0.) for _ in 1:events for i in inlet_rows]
    push!(result,region(points,fill(T(1000),length(points)),flow.reserve_marker,Frozen))
    return result
end

prepare_open_boundary(::Nothing,particles,constants,kernel,metadata) = nothing
function prepare_open_boundary(flow::OpenChannel{T},particles,constants,kernel,metadata) where T
    flow.duration >= metadata.SimulationTime || throw(ArgumentError("inlet reserve duration is shorter than SimulationTime"))
    T(flow.spacing) == constants.dx || throw(ArgumentError("open-channel spacing differs from simulation dx"))
    (min(flow.inlet_layers,flow.outlet_layers)-1)*flow.spacing >= kernel.H || throw(ArgumentError("open-channel buffer thickness must cover the SPH support"))
    sort(Int.(particles.ID)) == collect(1:length(particles)) || throw(ArgumentError("open boundaries require dense stable IDs 1:N"))
    reserve = sort!(findall(==(flow.reserve_marker),particles.GroupMarker);by = i -> particles.ID[i])
    !isempty(reserve) && all(i -> particles.Type[i] == Frozen,reserve) || throw(ArgumentError("call open_channel_geometry before allocation"))
    events = ceil(Int,flow.duration*flow.velocity/flow.spacing)+1
    length(reserve)%events == 0 || throw(ArgumentError("inlet reserve layout differs from its duration"))
    rows = length(reserve)÷events
    times = fill(Inf,length(particles))
    positions = zeros(SVector{2,Float64},length(particles))
    for (k,i) in enumerate(reserve)
        id = particles.ID[i]
        times[id] = (1+(k-1)÷rows)*Float64(flow.spacing)/Float64(flow.velocity)
        positions[id] = particles.Position[i]
    end
    return (;configuration = flow,emission_time = CuArray(times),emission_position = CuArray(positions),
        counters = CUDA.zeros(Int32,2),reserve_particles = length(reserve),initial_active_fluid = count(==(Fluid),particles.Type))
end

@inline function open_ghost_density(point,side,flow,positions,density,types,groups,starts,grid,kernel,constants)
    T = eltype(density)
    plane = side == flow.inlet_marker ? flow.inlet : flow.outlet
    ghost = SVector(2plane-point[1],point[2])
    cg = cell_coords(ghost,bin_scale(grid,kernel.H⁻¹))
    lx,ly = cg[1]-grid.origin[1],cg[2]-grid.origin[2]
    R = reach(grid)
    xlo,xhi = max(Int32(0),lx-R),min(grid.dims[1]-1,lx+R)
    weight,perturbation = zero(T),zero(T)
    @inbounds if xlo <= xhi
        for dy in -R:R
            y = ly+dy
            if 0 <= y < grid.dims[2]
                first_cell = Int32(1)+xlo+grid.dims[1]*y
                for j in starts[first_cell]+Int32(1):starts[first_cell+(xhi-xlo)+Int32(1)]
                    if types[j] == Fluid && groups[j] == flow.fluid_marker
                        delta = SVector{2,T}(positions[j]-ghost)
                        distance2 = dot(delta,delta)
                        if distance2 <= kernel.H²
                            w = SPHKernels.Wᵢⱼ(kernel,sqrt(distance2)*kernel.h⁻¹)*constants.m₀
                            volume_weight = w/density[j]
                            weight += volume_weight
                            perturbation += volume_weight*(density[j]-constants.ρ₀)
                        end
                    end
                end
            end
        end
    end
    value = weight > eps(T) ? constants.ρ₀+perturbation/weight : constants.ρ₀
    return isfinite(value) && value > 0 ? value : constants.ρ₀
end

function open_values_kernel!(positions,velocity,density,types,groups,starts,gridarg,kernel,constants,flow,step,n)
    step_active(step) || return nothing
    i = thread_index()
    i > n && return nothing
    @inbounds begin
        buffer = types[i] == Fluid && (groups[i] == flow.inlet_marker || groups[i] == flow.outlet_marker)
        # Outside the open planes the DBC floor has a truncated fluid
        # neighbourhood. Extend ghost density there instead of accumulating
        # an artificial end-wall pressure through its continuity equation.
        wall = types[i] == Fixed && (positions[i][1] <= flow.inlet || positions[i][1] >= flow.outlet)
        if buffer || wall
            side = wall ? (positions[i][1] <= flow.inlet ? flow.inlet_marker : flow.outlet_marker) : groups[i]
            density[i] = open_ghost_density(positions[i],side,flow,positions,density,types,groups,starts,load_grid(gridarg),kernel,constants)
            buffer && (velocity[i] = SVector(flow.velocity,zero(flow.velocity)))
        end
    end
    return nothing
end

launch_open_boundary_values!(::Nothing,args...) = nothing
function launch_open_boundary_values!(state,position,velocity,density,types,groups,starts,grid,kernel,constants,step)
    @cuda threads=128 blocks=cld(length(position),128) open_values_kernel!(position,velocity,density,types,groups,starts,grid,kernel,constants,state.configuration,step,Int32(length(position)))
    return nothing
end

function open_events_kernel!(positions,velocity,density,types,groups,ids,cells,pos_cells,grid,state,kernel,constants,step,n)
    step_active(step) || return nothing
    i = thread_index()
    i > n && return nothing
    flow = state.configuration
    time = step_time(step)+Float64(step_dt(step))
    changed = false
    @inbounds begin
        id = ids[i]
        if types[i] == Frozen && groups[i] == flow.reserve_marker && state.emission_time[id] <= time+max(64eps(max(1.,time)),8eps(eltype(density))*flow.spacing)
            positions[i] = state.emission_position[id]+SVector((time-state.emission_time[id])*flow.velocity,0.)
            velocity[i] = SVector(flow.velocity,zero(flow.velocity))
            depth = max(0.,flow.water_level-positions[i][2])
            density[i] = eltype(density)(constants.ρ₀*(1+7constants.g*depth/constants.c₀^2)^(1/7))
            groups[i],types[i] = flow.inlet_marker,Fluid
            store_pos_cell!(pos_cells,i,positions[i],cells,grid,kernel.H)
            CUDA.@atomic state.counters[1] += Int32(1)
            changed = true
        elseif types[i] == Fluid
            x = positions[i][1]
            if (groups[i] == flow.fluid_marker && x < flow.inlet) || x > flow.outlet+flow.outlet_layers*flow.spacing
                types[i] = Frozen
                velocity[i] = zero(eltype(velocity))
                CUDA.@atomic state.counters[2] += Int32(1)
                # Retirement only removes a source. Existing sorted ranges
                # remain conservative and every reader skips Frozen entries.
                # Activation below needs a rebuild; removal does not.
            elseif groups[i] == flow.inlet_marker && x > flow.inlet
                groups[i] = flow.fluid_marker
            elseif groups[i] == flow.fluid_marker && x >= flow.outlet
                groups[i] = flow.outlet_marker
            end
        end
        if changed
            # Force a rebuild before any subsequent force evaluation sees a
            # changed active set. The existing graph stop/readback handles it.
            CUDA.@atomic step.f[F_DX] = max(step.f[F_DX],Float64(2kernel.H))
        end
    end
    return nothing
end

launch_open_boundary_events!(::Nothing,args...) = nothing
function launch_open_boundary_events!(state,gpu,support,cl,kernel,constants,step)
    @cuda threads=128 blocks=cld(length(gpu),128) open_events_kernel!(gpu.Position,gpu.Velocity,gpu.Density,gpu.Type,
        gpu.GroupMarker,gpu.ID,gpu.CellID,support.PosCells,cl.grid_dev,state,kernel,constants,step,Int32(length(gpu)))
    return nothing
end

open_boundary_state(::Nothing,particles) = nothing
function open_boundary_state(state,particles)
    counters = Array(state.counters)
    return (;emitted_particles = counters[1],retired_particles = counters[2],active_fluid = count(==(Fluid),particles.Type),
        reserve_particles = state.reserve_particles,initial_active_fluid = state.initial_active_fluid,
        remaining_reserve = count(i -> particles.Type[i] == Frozen && particles.GroupMarker[i] == state.configuration.reserve_marker,eachindex(particles.ID)))
end
end
