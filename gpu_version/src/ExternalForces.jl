module ExternalForces

using Adapt
using CUDA
using CSV
using StaticArrays
using LinearAlgebra: cross
using ..SimulationGeometry: Fluid
using ..GPUStepState: step_active, step_time, step_dt

export ExternalForce, external_acceleration

"""
    ExternalForce{D,T}(group_marker; times, linear_acceleration = nothing,
        angular_acceleration = nothing, center = zeros(D), global_gravity = true,
        start_time = 0, end_time = Inf)
    ExternalForce{D,T}(group_marker, csv_path; kwargs...)

Prescribed acceleration of a fluid `GroupMarker`, in SI units, following
DualSPHysics 5.4 `accinput`. Supply at least two strictly increasing, nonnegative
`times`. Linear samples have `D` components; angular samples have three in 3D
or are scalars in 2D (positive counterclockwise). Omitted histories are zero.
Samples are linearly interpolated and held at their endpoints outside the table.
The force, including its gravity override, is active on `[start_time, end_time]`.

The angular contribution includes tangential, centripetal and Coriolis terms
about the fixed `center`. For compatibility with the reference, frame velocities
start at zero at the first sample, accumulate `a[k] * (t[k] - t[k-1])`, and are
then linearly interpolated. Angular terms run only when interpolated angular
acceleration is nonzero. The Coriolis components follow the reference's
component-wise velocity subtraction (see `external_acceleration`). These are
the 5.4 conventions, rather than exact integration of a piecewise linear signal.

CSV input uses the reference's seven semicolon-separated columns:
`time; ax; ay; az; alpha_x; alpha_y; alpha_z`, with optional `#` comments.
In 2D, reference x/z map to simulation x/y and angular y changes sign; out-of-plane
components must be zero. Force values are accelerations, not forces in newtons.
Pass one force or a tuple/vector as `SimExternalForces` to `RunSimulation`.
Only fluid groups are supported; prescribed boundaries and floating bodies are
rejected. Each fluid marker can have at most one force history.
"""
struct ExternalForce{D,T,A<:AbstractVector{T},V<:AbstractVector{SVector{3,T}}}
    group_marker::Int
    times::A
    linear::V
    angular::V
    linear_velocity::V
    angular_velocity::V
    center::SVector{3,Float64}
    global_gravity::Bool
    start_time::T
    end_time::T
end

function Adapt.adapt_structure(to, force::ExternalForce{D,T}) where {D,T}
    times = Adapt.adapt(to, force.times)
    linear = Adapt.adapt(to, force.linear)
    return ExternalForce{D,T,typeof(times),typeof(linear)}(force.group_marker, times, linear,
        Adapt.adapt(to, force.angular), Adapt.adapt(to, force.linear_velocity),
        Adapt.adapt(to, force.angular_velocity), force.center, force.global_gravity,
        force.start_time, force.end_time)
end

@inline embed(v::SVector{3}) = v
@inline embed(v::SVector{2,T}) where {T} = SVector{3,T}(v[1], zero(T), v[2])
@inline project(v, ::Val{3}) = v
@inline project(v, ::Val{2}) = SVector(v[1], v[3])

function ExternalForce{D,T}(group_marker::Integer; times, linear_acceleration = nothing,
        angular_acceleration = nothing, center = zeros(D), global_gravity::Bool = true,
        start_time = 0, end_time = Inf) where {D,T<:AbstractFloat}
    D in (2, 3) || throw(ArgumentError("external forces require 2D or 3D"))
    ts = T.(collect(times))
    length(ts) >= 2 || throw(ArgumentError("external forces require at least two time samples"))
    all(isfinite, ts) && first(ts) >= 0 && all(>(zero(T)), diff(ts)) ||
        throw(ArgumentError("force times must be finite, nonnegative and strictly increasing in $T"))
    start, stop = T(start_time), T(end_time)
    isfinite(start) && start >= 0 && !isnan(stop) && start <= stop ||
        throw(ArgumentError("force interval must satisfy 0 <= start_time <= end_time"))
    length(center) == D || throw(DimensionMismatch("force center requires $D components"))
    c = embed(SVector{D,Float64}(center))
    all(isfinite, c) || throw(ArgumentError("force center must be finite"))
    linear = fill(zero(SVector{3,T}), length(ts))
    angular = copy(linear)
    for (samples, output, isangular) in ((linear_acceleration, linear, false), (angular_acceleration, angular, true))
        samples === nothing && continue
        length(samples) == length(ts) || throw(DimensionMismatch("force samples must match times"))
        for k in eachindex(ts)
            value = samples[k]
            if isangular && D == 2
                value isa Real || throw(ArgumentError("2D angular acceleration samples must be scalars"))
                output[k] = SVector{3,T}(0, -value, 0)
            else
                length(value) == D || throw(DimensionMismatch("force sample requires $D components"))
                output[k] = embed(SVector{D,T}(value))
            end
            all(isfinite, output[k]) || throw(ArgumentError("force samples must be finite in $T"))
        end
    end
    velocities = map((linear, angular)) do samples
        v = fill(zero(SVector{3,T}), length(ts))
        for k in 2:length(ts)
            v[k] = v[k-1] + samples[k] * (ts[k] - ts[k-1])
            all(isfinite, v[k]) || throw(ArgumentError("integrated force velocity overflows $T"))
        end
        v
    end
    return ExternalForce{D,T,typeof(ts),typeof(linear)}(Int(group_marker), ts, linear, angular,
        velocities[1], velocities[2], c, global_gravity, start, stop)
end

function ExternalForce{D,T}(group_marker::Integer, path::AbstractString; kwargs...) where {D,T<:AbstractFloat}
    D in (2, 3) || throw(ArgumentError("external forces require 2D or 3D"))
    rows = collect(CSV.File(path; header = false, delim = ';', comment = "#", types = Float64, strict = true))
    length(rows) >= 2 || throw(ArgumentError("force CSV requires at least two rows"))
    all(row -> length(row) == 7 && all(x -> x isa Real && isfinite(x), row), rows) ||
        throw(ArgumentError("force CSV requires seven finite numeric columns"))
    if D == 2
        all(row -> iszero(row[3]) && iszero(row[5]) && iszero(row[7]), rows) ||
            throw(ArgumentError("2D force CSV requires zero linear y and angular x/z components"))
        linear = [SVector(row[2], row[4]) for row in rows]
        angular = [-row[6] for row in rows]
    else
        linear = [SVector(row[2], row[3], row[4]) for row in rows]
        angular = [SVector(row[5], row[6], row[7]) for row in rows]
    end
    return ExternalForce{D,T}(group_marker; times = [row[1] for row in rows],
        linear_acceleration = linear, angular_acceleration = angular, kwargs...)
end

# A device-safe binary search also handles endpoint holding without extrapolation.
@inline function sample_interval(times, time)
    lo, hi = 1, length(times)
    while lo < hi
        mid = (lo + hi + 1) >>> 1
        if @inbounds times[mid] <= time
            lo = mid
        else
            hi = mid - 1
        end
    end
    k = min(lo, length(times) - 1)
    @inbounds fraction = clamp((time - times[k]) / (times[k+1] - times[k]), zero(eltype(times)), one(eltype(times)))
    return k, fraction
end

@inline interpolate(values, k, fraction) = @inbounds values[k] + fraction * (values[k+1] - values[k])

struct ExternalForceSample{T}
    active::Bool
    linear::SVector{3,T}
    angular::SVector{3,T}
    angular_velocity::SVector{3,T}
    linear_velocity::SVector{3,T}
end

@inline function sample_force(force::ExternalForce{D,T}, time) where {D,T}
    z = zero(SVector{3,T})
    force.start_time <= time <= force.end_time || return ExternalForceSample(false, z, z, z, z)
    k, fraction = sample_interval(force.times, time)
    acc = interpolate(force.linear, k, fraction)
    alpha = interpolate(force.angular, k, fraction)
    z = zero(acc)
    omega = iszero(alpha) ? z : interpolate(force.angular_velocity, k, fraction)
    frame_velocity = iszero(alpha) ? z : interpolate(force.linear_velocity, k, fraction)
    return ExternalForceSample(true, acc, alpha, omega, frame_velocity)
end

"""
    external_acceleration(force, position, velocity, time, g)

Evaluate the additional acceleration for one particle belonging to `force`.
The solver adds its normal gravity separately; when `global_gravity = false`,
this result cancels fluid gravity along the last coordinate. Works on host
histories and adapted device histories, preserving higher-precision positions.
"""
@inline function external_acceleration(force::ExternalForce{D,T}, position::SVector{D}, velocity::SVector{D}, time, g) where {D,T}
    return sampled_acceleration(force, position, velocity, sample_force(force, time), g)
end

@inline function sampled_acceleration(force::ExternalForce{D,T}, position::SVector{D}, velocity::SVector{D}, sample, g) where {D,T}
    sample.active || return zero(SVector{D,T})
    acc, alpha = sample.linear, sample.angular
    if !iszero(alpha)
        omega, frame_velocity = sample.angular_velocity, sample.linear_velocity
        r = embed(position) - force.center
        v = embed(velocity)
        # Preserve DualSPHysics 5.4's component-wise Coriolis convention.
        coriolis = 2SVector(omega[2]*v[3] - omega[3]*(v[2] - frame_velocity[2]),
            omega[3]*v[1] - omega[1]*(v[3] - frame_velocity[3]),
            omega[1]*v[2] - omega[2]*(v[1] - frame_velocity[1]))
        acc += cross(alpha, r) + cross(omega, cross(omega, r)) + coriolis
    end
    if !force.global_gravity
        acc += SVector(zero(T), zero(T), T(g))
    end
    return SVector{D,T}(project(acc, Val(D)))
end

function prepare_external_forces(forces, particles, ::Val{D}, ::Type{T}; sample_once::Bool = true) where {D,T}
    forces === nothing && return ()
    histories = forces isa ExternalForce ? (forces,) : Tuple(forces)
    markers = Int[]
    for force in histories
        force isa ExternalForce{D,T} || throw(ArgumentError("SimExternalForces must contain ExternalForce{$D,$T} histories"))
        force.group_marker in markers && throw(ArgumentError("duplicate external force for GroupMarker $(force.group_marker)"))
        push!(markers, force.group_marker)
        selected = particles.GroupMarker .== force.group_marker
        any(selected) || throw(ArgumentError("external force GroupMarker $(force.group_marker) is absent"))
        all(==(Fluid), particles.Type[selected]) || throw(ArgumentError("external forces require an exclusively fluid GroupMarker"))
    end
    return map(histories) do force
        history = Adapt.adapt(CuArray, force)
        sample_once ? (; history, sample = CuArray([sample_force(force, zero(T))])) : history
    end
end

function external_force_kernel!(acceleration, position, velocity, types, groups, force, step, g, half_time, n)
    step_active(step) || return nothing
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds if types[i] == Fluid && groups[i] == force.group_marker
            time = step_time(step) + (half_time ? step_dt(step) / 2 : zero(step_time(step)))
            acceleration[i] += external_acceleration(force, position[i], velocity[i], time, g)
        end
    end
    return nothing
end

function force_sample_kernel!(sample, force, step, half_time)
    step_active(step) || return nothing
    time = step_time(step) + (half_time ? step_dt(step) / 2 : zero(step_time(step)))
    @inbounds sample[1] = sample_force(force, time)
    return nothing
end

function sampled_force_kernel!(acceleration, position, velocity, types, groups, force, sample, step, g, n)
    step_active(step) || return nothing
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    @inbounds if i <= n && types[i] == Fluid && groups[i] == force.group_marker
        acceleration[i] += sampled_acceleration(force, position[i], velocity[i], sample[1], g)
    end
    return nothing
end

function launch_external_force!(force::ExternalForce, acceleration, position, velocity, types, groups, step, g, half_time, n)
    @cuda threads=256 blocks=cld(n, 256) external_force_kernel!(
        acceleration, position, velocity, types, groups, force, step, g, half_time, n)
    return nothing
end

function launch_external_force!(prepared::NamedTuple, acceleration, position, velocity, types, groups, step, g, half_time, n)
    (; history, sample) = prepared
    @cuda threads=1 blocks=1 force_sample_kernel!(sample, history, step, half_time)
    @cuda threads=256 blocks=cld(n, 256) sampled_force_kernel!(
        acceleration, position, velocity, types, groups, history, sample, step, g, n)
    return nothing
end

function launch_external_forces!(forces::Tuple, acceleration, position, velocity, types, groups, step, g; half_time = false)
    n = length(position)
    n == 0 && return nothing
    for force in forces
        launch_external_force!(force, acceleration, position, velocity, types, groups, step, g, half_time, n)
    end
    return nothing
end

end
