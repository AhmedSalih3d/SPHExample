# Included in GPUKernels. This path keeps the CPU-reference integrators intact.
mutable struct DualStepSupport{T}
    viscous::CuVector{T}
    exclusions::CuVector{UInt8}
    reduction::ReductionWorkspace{SVector{4,T}}
    time::Float64
    dt::Float64
    travel::Float64
    initialized::Bool
    excluded_position::Int
    excluded_density::Int
end
DualStepSupport{T}(n) where {T} = DualStepSupport(CUDA.zeros(T,n), CUDA.zeros(UInt8,n), ReductionWorkspace{SVector{4,T}}(n), 0.0, 0.0, 0.0, false, 0, 0)

@inline dual_max(a, b) = max.(a, b)

@inline function dual_pressure(density,constants,::Val{Fast}) where {Fast}
    ratio = density * inv(constants.ρ₀)
    power = Fast ? Base.FastMath.pow_fast(ratio,constants.γ) : ratio^constants.γ
    return constants.Cb * (power - one(density))
end

function dual_criteria_kernel!(partial, velocity, acceleration, types, viscous, n)
    value = zero(eltype(partial))
    i = thread_index()
    @inbounds while i <= n
        if types[i] != Frozen
            v = velocity[i]
            a = types[i] == Fluid ? acceleration[i] : zero(v)
            value = dual_max(value, SVector(dot(a,a), dot(v,v), viscous[i], zero(v[1])))
        end
        i += blockDim().x * gridDim().x
    end
    block_reduce_store!(partial, value, dual_max)
    return nothing
end

function launch_dual_criteria!(support, velocity, acceleration, types)
    @cuda threads=REDUCE_THREADS blocks=support.reduction.nblocks dual_criteria_kernel!(
        support.reduction.partial, velocity, acceleration, types, support.viscous, Int32(length(types)))
    return finish_reduction(support.reduction, dual_max, zero(eltype(support.reduction.partial)))
end

@inline function dual_outside(position,scheme)
    outside = false
    for d in 1:length(position)
        outside |= position[d] < scheme.domain_min[d] || position[d] > scheme.domain_max[d]
    end
    return outside
end

function dual_predictor_kernel!(xp, vp, rp, x, v, rho, acceleration, rate, types, dt, constants, exclusions, periodicity, scheme, n)
    i = thread_index()
    i > n && return nothing
    T = eltype(rho)
    @inbounds begin
        kind = types[i]
        xp[i], vp[i], rp[i] = x[i], v[i], rho[i]
        kind == Frozen && return nothing
        density = T(Float64(rho[i]) + (dt / 2) * Float64(rate[i]))
        rp[i] = is_wall(kind) ? max(constants.ρ₀,density) : density
        if kind == Fluid
            gravity = ConstructGravitySVector(v[i], -constants.g)
            total = SVector{length(v[i]),Float64}(acceleration[i]) + SVector{length(v[i]),Float64}(gravity)
            xp[i] = x[i] + SVector{length(v[i]),Float64}(v[i]) * (dt / 2)
            vp[i] = SVector{length(v[i]),T}(SVector{length(v[i]),Float64}(v[i]) + total * (dt / 2))
            if exclusions !== nothing
                # Keep predictor coordinates continuous relative to their old
                # cell anchors. Wrapping here would inflate global search
                # bounds by an entire period until the next accepted rebuild.
                checked_position = periodicity === nothing ? xp[i] : GPUCellGrid.wrap_periodic_position(xp[i],periodicity)
                out_position = dual_outside(checked_position,scheme)
                out_density = density < scheme.density_min || density > scheme.density_max
                if out_position || out_density
                    exclusions[i] = out_position ? UInt8(1) : UInt8(2)
                    types[i] = Frozen
                    x[i],rho[i],v[i],acceleration[i] = checked_position,density,zero(v[i]),zero(acceleration[i])
                end
            end
        end
    end
    return nothing
end

function launch_dual_predictor!(xp,vp,rp,x,v,rho,acceleration,rate,types,dt,constants;
        exclusions=nothing,periodicity=nothing,scheme=DualSPHysicsSymplectic())
    needs_exclusions = any(isfinite,scheme.domain_min) || any(isfinite,scheme.domain_max) || isfinite(scheme.density_min) || isfinite(scheme.density_max)
    needs_exclusions && exclusions===nothing && throw(ArgumentError("Reference exclusions require their event buffer"))
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(length(types),ELEMENTWISE_THREADS) dual_predictor_kernel!(
        xp,vp,rp,x,v,rho,acceleration,rate,types,dt,constants,exclusions,periodicity,scheme,Int32(length(types)))
end

function dual_corrector_kernel!(x,v,rho,xp,rp,acceleration,rate,types,dt,constants,n)
    i = thread_index()
    i > n && return nothing
    T = eltype(rho)
    @inbounds begin
        kind = types[i]
        kind == Frozen && return nothing
        epsilon = -Float64(rate[i]) / Float64(rp[i]) * dt
        density = T(Float64(rho[i]) * (2 - epsilon) / (2 + epsilon))
        rho[i] = is_wall(kind) ? max(constants.ρ₀,density) : density
        if kind == Fluid
            old_velocity = SVector{length(v[i]),Float64}(v[i])
            gravity = ConstructGravitySVector(v[i], -constants.g)
            total = SVector{length(v[i]),Float64}(acceleration[i]) + SVector{length(v[i]),Float64}(gravity)
            velocity = SVector{length(v[i]),T}(old_velocity + total * dt)
            xp[i] = x[i] # predictor position is no longer needed; retain the old accepted position
            x[i] += (old_velocity + SVector{length(v[i]),Float64}(velocity)) * (dt / 2)
            v[i] = velocity
        end
    end
    return nothing
end

function launch_dual_corrector!(x,v,rho,xp,rp,acceleration,rate,types,dt,constants)
    @cuda threads=ELEMENTWISE_THREADS blocks=cld(length(types),ELEMENTWISE_THREADS) dual_corrector_kernel!(
        x,v,rho,xp,rp,acceleration,rate,types,dt,constants,Int32(length(types)))
end

@inline dual_finish_merge(a,b) = SVector(a[1]+b[1],a[2]+b[2],zero(a[3]),max(a[4],b[4]))

function dual_motion_finish_kernel!(partial,x,v,rho,previous,types,exclusions,dt,periodicity,domain_min,domain_max,rho_min,rho_max,n)
    value = zero(eltype(partial))
    T = eltype(eltype(v))
    i = thread_index()
    @inbounds while i <= n
        if types[i] == Frozen
            event = exclusions[i]
            value = dual_finish_merge(value,SVector(T(event==1),T(event==2),zero(T),zero(T)))
            exclusions[i] = 0
        else
            if periodicity !== nothing && types[i] == Fluid
                x[i] = GPUCellGrid.wrap_periodic_position(x[i],periodicity)
            end
            displacement = x[i] - previous[i]
            if types[i] == Moving
                v[i] = SVector{length(displacement),T}(displacement / dt)
            end
            out_position = false
            if types[i] == Fluid
                for d in 1:length(displacement)
                    out_position |= x[i][d] < domain_min[d] || x[i][d] > domain_max[d]
                end
            end
            out_density = types[i] == Fluid && (rho[i] < rho_min || rho[i] > rho_max)
            if out_position || out_density
                types[i] = Frozen
                v[i] = zero(v[i])
            end
            value = dual_finish_merge(value, SVector(T(out_position),T(out_density && !out_position),zero(T),T(norm(displacement))))
        end
        i += blockDim().x * gridDim().x
    end
    block_reduce_store!(partial,value,dual_finish_merge)
    return nothing
end

function launch_dual_motion_finish!(support,x,v,rho,previous,types,dt,periodicity,scheme)
    @cuda threads=REDUCE_THREADS blocks=support.reduction.nblocks dual_motion_finish_kernel!(
        support.reduction.partial,x,v,rho,previous,types,support.exclusions,dt,periodicity,
        scheme.domain_min,scheme.domain_max,scheme.density_min,scheme.density_max,Int32(length(types)))
    return finish_reduction(support.reduction,dual_finish_merge,zero(eltype(support.reduction.partial)))
end
