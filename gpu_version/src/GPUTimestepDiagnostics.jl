# Included inside GPUKernels. Keep IDs in their integer type, including IDs
# outside the exactly representable range of the simulation's float type.
struct TimestepLimit{T, I}
    visc::T
    force::T
    visc_id::I
    force_id::I
    valid::Bool
end

@inline neutral_limit(::Type{T}, ::Type{I}) where {T, I} =
    TimestepLimit(zero(T), T(Inf), zero(I), zero(I), false)

@inline function limit_reduce(a::TimestepLimit, b::TimestepLimit)
    a.valid || return b
    b.valid || return a
    visc_b = b.visc > a.visc || (b.visc == a.visc && b.visc_id < a.visc_id)
    force_b = b.force < a.force || (b.force == a.force && b.force_id < a.force_id)
    return TimestepLimit(max(a.visc, b.visc), min(a.force, b.force),
        visc_b ? b.visc_id : a.visc_id, force_b ? b.force_id : a.force_id, true)
end

struct TimestepRecord{T, I}
    iteration::Int32
    time::Float64
    dt::T
    force_dt::T
    acoustic_viscous_dt::T
    particle_id::I
    criterion::Int32 # 1: force, 2: acoustic/viscous, 3: output deadline
    has_particle::Bool
end

"Optional provenance of the reduction consumed by the timestep selector."
struct TimestepDiagnostics{T, I}
    reduction::ReductionWorkspace{TimestepLimit{T, I}}
    result::CuVector{TimestepRecord{T, I}}
    host::Vector{TimestepRecord{T, I}}
end

function TimestepDiagnostics{T, I}(n::Integer) where {T, I}
    initial = TimestepRecord(Int32(0), 0.0, zero(T), T(Inf), T(Inf), zero(I), Int32(0), false)
    return TimestepDiagnostics{T, I}(ReductionWorkspace{TimestepLimit{T, I}}(n), CuArray([initial]), [initial])
end

@inline diagnostic_init(::Nothing) = nothing
@inline diagnostic_init(::AbstractVector{TimestepLimit{T, I}}) where {T, I} = neutral_limit(T, I)
@inline diagnostic_accumulate(::Nothing, values, ids, i) = nothing
@inline function diagnostic_accumulate(acc::TimestepLimit, values, ids, i)
    id = @inbounds ids[i]
    return limit_reduce(acc, TimestepLimit(values[1], values[2], id, id, true))
end
@inline diagnostic_store!(::Nothing, acc) = nothing
@inline diagnostic_store!(partial, acc) = block_reduce_store!(partial, acc, limit_reduce)

@inline function timestep_limit_map(i, Position, Velocity, Acceleration, PositionHalf, ids, h, eta2)
    values = step_map(i, Position, Velocity, Acceleration, PositionHalf, h, eta2)
    id = @inbounds ids[i]
    return TimestepLimit(values[1], values[2], id, id, true)
end

function launch_timestep_diagnostic_reduction!(diagnostics::TimestepDiagnostics{T, I},
        Position, Velocity, Acceleration, PositionHalf, ids, kernel) where {T, I}
    launch_reduce!(diagnostics.reduction, timestep_limit_map, limit_reduce, neutral_limit(T, I),
        length(Position), Position, Velocity, Acceleration, PositionHalf, ids, T(kernel.h), T(kernel.η²))
    return nothing
end

@inline finish_diagnostic(::Nothing, nblocks) = nothing
@inline function finish_diagnostic(partial, nblocks)
    acc = diagnostic_init(partial)
    b = threadIdx().x
    @inbounds while b <= nblocks
        acc = limit_reduce(acc, partial[b])
        b += blockDim().x
    end
    return block_reduce(acc, limit_reduce)
end

@inline record_diagnostic!(::Nothing, limit, step, dt, force_dt, acoustic_dt, force) = nothing
@inline function record_diagnostic!(result, limit, step, dt, force_dt, acoustic_dt, force)
    clipped = dt < min(force_dt, acoustic_dt)
    id = force ? limit.force_id : limit.visc_id
    @inbounds result[1] = TimestepRecord(step.i[I_ITER] + Int32(1), step.f[F_TIME], oftype(force_dt, dt),
        force_dt, acoustic_dt, id, clipped ? Int32(3) : force ? Int32(1) : Int32(2), limit.valid && !clipped)
    return nothing
end

function write_timestep_diagnostic!(io::IO, diagnostics::TimestepDiagnostics, output_time)
    copyto!(diagnostics.host, diagnostics.result)
    record = only(diagnostics.host)
    record.iteration == 0 && return nothing
    criterion = ("force", "acoustic_viscous", "output")[record.criterion]
    println(io, join((output_time, record.iteration, record.time, record.dt, criterion,
        record.has_particle ? record.particle_id : "", record.force_dt, record.acoustic_viscous_dt), ","))
    flush(io)
    return nothing
end
