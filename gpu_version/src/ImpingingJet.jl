"""
    ImpingingJet{T}(; Spacing, Duration, Velocity, Inlet, Outlets, OutletHeight, Layers, VerticalLimits)

Planar downward inlet and lateral outlet conditions for a 2D mDBC jet.
`Inlet=(xmin,xmax,height)` and `Outlets=(left,right)` define the open zones;
`VerticalLimits=(bottom,top)` bounds escaped fluid. All lengths are in metres.
Both inlet and outlets fix density to the simulation reference density.
The inlet fixes velocity; outlet velocity uses zeroth-order ghost extrapolation.
Construct all particle geometry, wall ghosts and inactive reserves in the
simulation file, then pass this configuration as `SimOpenBoundaries`.
"""
struct ImpingingJet{T}
    spacing::Float64
    duration::Float64
    velocity::T
    inlet::SVector{3,Float64}
    outlets::SVector{2,Float64}
    outlet_height::Float64
    layers::Int32
    vertical_limits::SVector{2,Float64}
    reserve_marker::Int32
end
function ImpingingJet{T}(;Spacing,Duration,Velocity,Inlet,Outlets,OutletHeight,Layers,VerticalLimits) where {T<:AbstractFloat}
    all(x -> isfinite(x) && x > 0,(Spacing,Duration,T(Velocity),OutletHeight)) || throw(ArgumentError("jet spacing, duration, velocity and outlet height must be finite and positive"))
    inlet,outlets,limits = SVector{3,Float64}(Inlet),SVector{2,Float64}(Outlets),SVector{2,Float64}(VerticalLimits)
    all(isfinite,inlet) && all(isfinite,outlets) && all(isfinite,limits) &&
        outlets[1] < inlet[1] < inlet[2] < outlets[2] && limits[1] < 0 < OutletHeight < inlet[3] < limits[2] ||
        throw(ArgumentError("jet planes and limits must be finite and ordered"))
    2 <= Layers <= typemax(Int32) || throw(ArgumentError("jet buffers require at least two layers"))
    return ImpingingJet(Float64(Spacing),Float64(Duration),T(Velocity),inlet,outlets,Float64(OutletHeight),Int32(Layers),limits,Int32(5))
end

function prepare_open_boundary(flow::ImpingingJet{T},particles,constants,kernel,metadata) where T
    flow.duration >= metadata.SimulationTime || throw(ArgumentError("jet reserve duration is shorter than SimulationTime"))
    T(flow.spacing) == constants.dx || throw(ArgumentError("jet spacing differs from dx"))
    (flow.layers-1)*flow.spacing >= kernel.H || throw(ArgumentError("jet buffer layers must cover kernel support"))
    sort(Int.(particles.ID)) == collect(1:length(particles)) || throw(ArgumentError("jet requires dense stable IDs"))
    reserve = sort!(findall(==(flow.reserve_marker),particles.GroupMarker);by=i -> particles.ID[i])
    events = ceil(Int,flow.duration*flow.velocity/flow.spacing)+1
    rows = round(Int,(flow.inlet[2]-flow.inlet[1])/flow.spacing)+1
    length(reserve) == events*rows && all(i -> particles.Type[i] == Frozen,reserve) || throw(ArgumentError("jet geometry must contain the complete inactive inlet reserve"))
    times,positions = fill(Inf,length(particles)),zeros(SVector{2,Float64},length(particles))
    for (k,i) in enumerate(reserve)
        times[particles.ID[i]] = (1+(k-1)÷rows)*flow.spacing/flow.velocity
        positions[particles.ID[i]] = particles.Position[i]
    end
    return (;configuration=flow,emission_time=CuArray(times),emission_position=CuArray(positions),
        counters=CUDA.zeros(Int32,2),reserve_particles=length(reserve),initial_active_fluid=count(==(Fluid),particles.Type))
end

@inline function jet_ghost_velocity(point,side,previous_velocity,flow,positions,velocity,density,types,groups,starts,grid,kernel,constants)
    T = eltype(density)
    plane = side == 3 ? flow.outlets[1] : flow.outlets[2]
    ghost = SVector(2plane-point[1],point[2])
    cg = cell_coords(ghost,bin_scale(grid,kernel.H⁻¹))
    lx,ly = cg[1]-grid.origin[1],cg[2]-grid.origin[2]
    R = reach(grid)
    Rx = x_reach(grid)
    weight,v = zero(T),zero(eltype(velocity))
    for dy in -R:R, dx in -Rx:Rx
        x,y = lx+dx,ly+dy
        if 0 <= x < grid.dims[1] && 0 <= y < grid.dims[2]
            cell = Int32(1)+x+grid.dims[1]*y
            for j in starts[cell]+Int32(1):starts[cell+Int32(1)]
                if types[j] == Fluid && groups[j] == 1
                    delta = SVector{2,T}(positions[j]-ghost)
                    distance2 = dot(delta,delta)
                    if T(1e-18) <= distance2 <= kernel.H²
                        w = SPHKernels.Wᵢⱼ(kernel,sqrt(distance2)*kernel.h⁻¹)*constants.m₀/density[j]
                        weight += w
                        v += w*velocity[j]
                    end
                end
            end
        end
    end
    return weight > zero(T) ? v/weight : previous_velocity
end

function jet_values_kernel!(positions,velocity,density,types,groups,starts,gridarg,kernel,constants,flow,step,n)
    step_active(step) || return nothing
    i = thread_index()
    i > n && return nothing
    @inbounds if types[i] == Fluid
        if groups[i] == 2
            velocity[i] = SVector(zero(flow.velocity),-flow.velocity)
            density[i] = constants.ρ₀
        elseif groups[i] == 3 || groups[i] == 4
            density[i] = constants.ρ₀
            velocity[i] = jet_ghost_velocity(positions[i],groups[i],velocity[i],flow,positions,velocity,density,types,groups,starts,load_grid(gridarg),kernel,constants)
        end
    end
    return nothing
end

function jet_events_kernel!(positions,velocity,density,types,groups,ids,cells,pos_cells,grid,state,kernel,constants,step,n)
    step_active(step) || return nothing
    i = thread_index()
    i > n && return nothing
    flow = state.configuration
    time = step_time(step)+Float64(step_dt(step))
    @inbounds begin
        id = ids[i]
        if types[i] == Frozen && groups[i] == flow.reserve_marker && state.emission_time[id] <= time+64eps(max(1.,time))
            positions[i] = state.emission_position[id]+SVector(0.,-(time-state.emission_time[id])*flow.velocity)
            velocity[i] = SVector(zero(flow.velocity),-flow.velocity)
            density[i] = constants.ρ₀
            groups[i],types[i] = Int32(2),Fluid
            store_pos_cell!(pos_cells,i,positions[i],cells,grid,kernel.H)
            CUDA.@atomic state.counters[1] += Int32(1)
            CUDA.@atomic step.f[F_DX] = max(step.f[F_DX],Float64(2kernel.H))
        elseif types[i] == Fluid
            x,y = positions[i]
            outside_domain = x < flow.outlets[1]-flow.layers*flow.spacing || x > flow.outlets[2]+flow.layers*flow.spacing ||
                y < flow.vertical_limits[1] || y > flow.vertical_limits[2]
            inlet_backflow = groups[i] == 1 && y >= flow.inlet[3] && flow.inlet[1] <= x <= flow.inlet[2]
            if outside_domain || inlet_backflow
                types[i] = Frozen
                velocity[i] = zero(eltype(velocity))
                CUDA.@atomic state.counters[2] += Int32(1)
            elseif groups[i] == 2 && y < flow.inlet[3]
                groups[i] = Int32(1)
            elseif (groups[i] == 3 && x > flow.outlets[1]) || (groups[i] == 4 && x < flow.outlets[2])
                groups[i] = Int32(1)
            elseif groups[i] == 1 && x <= flow.outlets[1] && 0 <= y <= flow.outlet_height
                groups[i] = Int32(3)
            elseif groups[i] == 1 && x >= flow.outlets[2] && 0 <= y <= flow.outlet_height
                groups[i] = Int32(4)
            end
        end
    end
    return nothing
end
