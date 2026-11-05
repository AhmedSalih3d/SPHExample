# Velocity programs are flattened once. A missing program allocates no buffers.
struct DeviceVelocityKey{V}
    time::Float64
    value::V
    active::Bool
end
function append_velocity_keys!(keys,series,::Type{V}) where V
    start = Int32(length(keys)+1)
    if series !== nothing
        for (time,value) in zip(series.times,series.values)
            if value !== nothing && ((V <: Real) != (value isa Real) || (!(V <: Real) && length(value) != length(zero(V))))
                throw(ArgumentError("velocity key dimension must match $V"))
            end
            converted = value === nothing ? zero(V) : V(value)
            all(isfinite,converted) || throw(ArgumentError("velocity keys must be representable in $V"))
            push!(keys,DeviceVelocityKey(time,converted,value !== nothing))
        end
    end
    return SVector(start,Int32(length(keys)))
end
function attach_rigid_velocities(f,specification)
    isempty(specification.velocities) && return f
    T,D = eltype(f.mass),floating_dimension(f.center)
    V,W = SVector{D,T},D == 2 ? T : SVector{3,T}
    linear,angular = DeviceVelocityKey{V}[],DeviceVelocityKey{W}[]
    spans_l,spans_a = fill(SVector{2,Int32}(1,0),Int(f.nbodies)),fill(SVector{2,Int32}(1,0),Int(f.nbodies))
    body = Array(f.body)
    speed_bounds = zeros(SVector{2,T},Int(f.nbodies))
    for program in specification.velocities
        program.group <= length(body) && body[program.group] > 0 || throw(ArgumentError("velocity marker must identify a floating body"))
        b = body[program.group]
        spans_l[b] = append_velocity_keys!(linear,program.linear,V)
        spans_a[b] = append_velocity_keys!(angular,program.angular,W)
        maximum_speed(series) = series === nothing ? zero(T) : maximum(v -> v === nothing ? zero(T) : T(norm(v)),series.values)
        speed_bounds[b] = SVector(maximum_speed(program.linear),maximum_speed(program.angular))
    end
    motors = (;speed_bounds = CuArray(speed_bounds),linear = CuArray(linear),angular = CuArray(angular),linear_spans = CuArray(spans_l),angular_spans = CuArray(spans_a),
        linear_active = CUDA.zeros(Bool,Int(f.nbodies)),angular_active = CUDA.zeros(Bool,Int(f.nbodies)),
        linear_value = CUDA.zeros(V,Int(f.nbodies)),angular_value = CUDA.zeros(W,Int(f.nbodies)))
    return merge(f,(;motors,initial_velocity = true))
end
@inline function velocity_sample_impl(keys::AbstractVector{DeviceVelocityKey{V}},first,last,time) where V
    first > last && return false,zero(V)
    lo,hi = first,last+Int32(1)
    while lo < hi
        mid = lo+(hi-lo)÷Int32(2)
        @inbounds if keys[mid].time <= time
            lo = mid+Int32(1)
        else
            hi = mid
        end
    end
    end_index = max(first,lo-Int32(1))
    @inbounds lower = keys[end_index]
    lower.active || return false,zero(V)
    if end_index < last
        @inbounds upper = keys[end_index+Int32(1)]
        if upper.active && upper.time > lower.time
            fraction = clamp((time-lower.time)/(upper.time-lower.time),0.,1.)
            return true,lower.value+eltype(V)(fraction)*(upper.value-lower.value)
        end
    end
    return true,lower.value
end
function sample_rigid_velocities_kernel!(motors,frozen,pause,step,fraction,n)
    step_active(step) || return nothing
    b = thread_index()
    b > n && return nothing
    @inbounds begin
        time = step_time(step)+fraction*Float64(step_dt(step))
        active = frozen[b] == 0 && time >= pause[b]
        linear,value = velocity_sample_impl(motors.linear,motors.linear_spans[b]...,time)
        angular,spin = velocity_sample_impl(motors.angular,motors.angular_spans[b]...,time)
        motors.linear_active[b],motors.angular_active[b] = active && linear,active && angular
        motors.linear_value[b],motors.angular_value[b] = value,spin
    end
    return nothing
end
function sample_rigid_velocities!(f,step,fraction)
    hasproperty(f,:motors) || return nothing
    @cuda threads=128 blocks=cld(Int(f.nbodies),128) sample_rigid_velocities_kernel!(f.motors,f.frozen,f.pause,step,Float64(fraction),f.nbodies)
    return nothing
end
@inline motor_linear_active(f,b) = hasproperty(f,:motors) && b != 0 && @inbounds(f.motors.linear_active[b])
@inline motor_angular_active(f,b) = hasproperty(f,:motors) && b != 0 && @inbounds(f.motors.angular_active[b])
@inline function apply_rigid_motor!(f,velocity,omega,b)
    if hasproperty(f,:motors)
        @inbounds begin
            motor_linear_active(f,b) && (velocity[b] = f.motors.linear_value[b])
            motor_angular_active(f,b) && (omega[b] = f.motors.angular_value[b])
        end
    end
    return nothing
end
@inline function rigid_motor_point_velocity(f,velocity,omega,b,arm)
    b == 0 && return zero(eltype(velocity))
    @inbounds v = motor_linear_active(f,b) ? f.motors.linear_value[b] : velocity[b]
    @inbounds w = motor_angular_active(f,b) ? f.motors.angular_value[b] : omega[b]
    return v+angular_cross(w,arm)
end
