"""
    ImpingingJet{T}(; Spacing=0.0005, Duration=0.1, Velocity=20)

Downward 2D jet based on DualSPHysics 5.4 `inletoutlet/08_ImpingingJet`.
Uses a fixed-density inlet and two lateral ghost-extrapolated outlets.
Currently uses DBC and standard stepping; this is not an mDBC reference reproduction.
Generate particles with `impinging_jet_geometry` and pass as `SimOpenBoundaries`.
"""
struct ImpingingJet{T}
    spacing::Float64
    duration::Float64
    velocity::T
    reserve_marker::Int32
end
function ImpingingJet{T}(;Spacing=0.0005,Duration=0.1,Velocity=20) where {T<:AbstractFloat}
    all(x -> isfinite(x) && x > 0,(Spacing,Duration,T(Velocity))) || throw(ArgumentError("jet spacing, duration and velocity must be finite and positive"))
    return ImpingingJet(Float64(Spacing),Float64(Duration),T(Velocity),Int32(5))
end

"""Generate the plate, initial jet, four-layer inlet and lateral outlet buffers."""
function impinging_jet_geometry(flow::ImpingingJet{T}) where T
    dx = flow.spacing
    # Preserve all reference dimensions on the particle lattice.
    all(x -> isapprox(x/dx,round(x/dx);atol=1e-8), (0.2,0.015,0.005,0.125,0.03)) ||
        throw(ArgumentError("jet spacing must divide the reference dimensions"))
    points(xs,ys) = [SVector{2,Float64}(x,y) for x in xs for y in ys]
    region(p,marker,kind;velocity=zero(SVector{2,T})) = SPHGeometry{2,T}(;Particles=particle_struct_array(p,T(1000);Velocity=fill(velocity,length(p))),Type=kind,GroupMarker=Int(marker))
    xs = collect(-0.015:dx:0.015)
    geometry = [region(points(-0.2-6dx:dx:0.2+5dx,-3dx:dx:0.),10,Fixed),
        region(points(xs,0.12:dx:0.125-dx),1,Fluid;velocity=SVector{2,T}(0,-flow.velocity)),
        region(points(xs,0.125:dx:0.125+3dx),2,Fluid;velocity=SVector{2,T}(0,-flow.velocity)),
        region(points(-0.2-3dx:dx:-0.2,dx:dx:0.03),3,Fluid),
        region(points(0.2:dx:0.2+3dx,dx:dx:0.03),4,Fluid)]
    events = ceil(Int,flow.duration*flow.velocity/dx)+1
    reserve = [SVector{2,Float64}(x,0.125+3dx) for _ in 1:events for x in xs]
    push!(geometry,region(reserve,flow.reserve_marker,Frozen))
    return geometry
end

function prepare_open_boundary(flow::ImpingingJet{T},particles,constants,kernel,metadata) where T
    flow.duration >= metadata.SimulationTime || throw(ArgumentError("jet reserve duration is shorter than SimulationTime"))
    T(flow.spacing) == constants.dx || throw(ArgumentError("jet spacing differs from dx"))
    3flow.spacing >= kernel.H || throw(ArgumentError("four jet buffer layers must cover kernel support"))
    sort(Int.(particles.ID)) == collect(1:length(particles)) || throw(ArgumentError("jet requires dense stable IDs"))
    reserve = sort!(findall(==(flow.reserve_marker),particles.GroupMarker);by=i -> particles.ID[i])
    events = ceil(Int,flow.duration*flow.velocity/flow.spacing)+1
    rows = round(Int,0.03/flow.spacing)+1
    length(reserve) == events*rows && all(i -> particles.Type[i] == Frozen,reserve) || throw(ArgumentError("call impinging_jet_geometry before allocation"))
    times,positions = fill(Inf,length(particles)),zeros(SVector{2,Float64},length(particles))
    for (k,i) in enumerate(reserve)
        times[particles.ID[i]] = (1+(k-1)÷rows)*flow.spacing/flow.velocity
        positions[particles.ID[i]] = particles.Position[i]
    end
    return (;configuration=flow,emission_time=CuArray(times),emission_position=CuArray(positions),
        counters=CUDA.zeros(Int32,2),reserve_particles=length(reserve),initial_active_fluid=count(==(Fluid),particles.Type))
end

@inline function jet_ghost_values(point,side,positions,velocity,density,types,groups,starts,grid,kernel,constants)
    T = eltype(density)
    plane = side == 3 ? -0.2 : 0.2
    ghost = SVector(2plane-point[1],point[2])
    cg = cell_coords(ghost,bin_scale(grid,kernel.H⁻¹))
    lx,ly = cg[1]-grid.origin[1],cg[2]-grid.origin[2]
    R = reach(grid)
    weight,rho,v = zero(T),zero(T),zero(eltype(velocity))
    for dy in -R:R, dx in -R:R
        x,y = lx+dx,ly+dy
        if 0 <= x < grid.dims[1] && 0 <= y < grid.dims[2]
            cell = Int32(1)+x+grid.dims[1]*y
            for j in starts[cell]+Int32(1):starts[cell+Int32(1)]
                if types[j] == Fluid && groups[j] == 1
                    delta = SVector{2,T}(positions[j]-ghost)
                    distance2 = dot(delta,delta)
                    if distance2 <= kernel.H²
                        w = SPHKernels.Wᵢⱼ(kernel,sqrt(distance2)*kernel.h⁻¹)*constants.m₀/density[j]
                        weight += w
                        rho += w*density[j]
                        v += w*velocity[j]
                    end
                end
            end
        end
    end
    return weight > eps(T) ? (rho/weight,v/weight) : (constants.ρ₀,zero(v))
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
            density[i],velocity[i] = jet_ghost_values(positions[i],groups[i],positions,velocity,density,types,groups,starts,load_grid(gridarg),kernel,constants)
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
            if abs(x) > 0.2+4flow.spacing || y < -4flow.spacing || (groups[i] == 1 && y > 0.125)
                types[i] = Frozen
                velocity[i] = zero(eltype(velocity))
                CUDA.@atomic state.counters[2] += Int32(1)
            elseif groups[i] == 2 && y < 0.125
                groups[i] = Int32(1)
            elseif groups[i] == 1 && x <= -0.2
                groups[i] = Int32(3)
            elseif groups[i] == 1 && x >= 0.2
                groups[i] = Int32(4)
            end
        end
    end
    return nothing
end
