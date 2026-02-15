module FluidAcceleration

export FluidAccelerationSeries, FluidAccelerationByGroup,
       FluidAccelerationInputState, FluidAccelerationInputSeries, FluidAccelerationInputByGroup,
       LoadFluidAccelerationSeriesCSV, LoadFluidAccelerationByGroupCSV,
       LoadFluidAccelerationInputSeriesCSV, LoadFluidAccelerationInputByGroupCSV,
       EvaluateFluidAcceleration

using CSV
using StaticArrays: SVector

mutable struct FluidAccelerationSeries{D,T<:AbstractFloat}
    Times::Vector{T}
    Values::Vector{SVector{D,T}}
    Cursor::Int
end

function FluidAccelerationSeries(Times::AbstractVector{T}, Values::AbstractVector{SVector{D,T}}) where {D,T<:AbstractFloat}
    @assert !isempty(Times) "Fluid acceleration timeline cannot be empty."
    @assert length(Times) == length(Values) "Fluid acceleration times and values must have the same length."
    times_vector = collect(Times)
    values_vector = collect(Values)
    @assert issorted(times_vector) "Fluid acceleration times must be sorted in ascending order."
    return FluidAccelerationSeries{D,T}(times_vector, values_vector, 1)
end

mutable struct FluidAccelerationByGroup{D,T<:AbstractFloat}
    Models::Vector{Union{Nothing,FluidAccelerationSeries{D,T}}}
    CurrentValues::Vector{SVector{D,T}}
end

function FluidAccelerationByGroup(Models::AbstractVector{Union{Nothing,FluidAccelerationSeries{D,T}}}) where {D,T<:AbstractFloat}
    models_vector = collect(Models)
    current_values = fill(zero(SVector{D,T}), length(models_vector))
    return FluidAccelerationByGroup{D,T}(models_vector, current_values)
end

struct FluidAccelerationInputState{D,T<:AbstractFloat}
    LinearAcceleration::SVector{D,T}
    AngularAcceleration::SVector{D,T}
    Centre::SVector{D,T}
    AngularVelocity::SVector{D,T}
    LinearVelocity::SVector{D,T}
    GlobalGravityEnabled::Bool
end

mutable struct FluidAccelerationInputSeries{D,T<:AbstractFloat}
    Times::Vector{T}
    LinearAccelerationValues::Vector{SVector{D,T}}
    AngularAccelerationValues::Vector{SVector{D,T}}
    LinearVelocityValues::Vector{SVector{D,T}}
    AngularVelocityValues::Vector{SVector{D,T}}
    Centre::SVector{D,T}
    GlobalGravityEnabled::Bool
    Cursor::Int
end

mutable struct FluidAccelerationInputByGroup{D,T<:AbstractFloat}
    Models::Vector{Union{Nothing,FluidAccelerationInputSeries{D,T}}}
    CurrentValues::Vector{FluidAccelerationInputState{D,T}}
end

@inline function ZeroFluidAccelerationInputState(::Val{D}, ::Type{T}) where {D,T<:AbstractFloat}
    zero_vector = zero(SVector{D,T})
    return FluidAccelerationInputState{D,T}(
        zero_vector,
        zero_vector,
        zero_vector,
        zero_vector,
        zero_vector,
        true,
    )
end

@inline function ToAccelerationCentre(Value, ::Val{D}, ::Type{T}) where {D,T<:AbstractFloat}
    return SVector{D,T}(ntuple(i -> T(Value[i]), D))
end

@inline function IntegrateAccelerationTimeline(
    Times::AbstractVector{T},
    AccelerationValues::AbstractVector{SVector{D,T}},
) where {D,T<:AbstractFloat}
    VelocityValues = Vector{SVector{D,T}}(undef, length(Times))
    VelocityValues[1] = zero(SVector{D,T})
    @inbounds for i in 2:length(Times)
        dt = Times[i] - Times[i - 1]
        VelocityValues[i] = VelocityValues[i - 1] + AccelerationValues[i] * dt
    end
    return VelocityValues
end

function FluidAccelerationInputSeries(
    Times::AbstractVector{T},
    LinearAccelerationValues::AbstractVector{SVector{D,T}},
    AngularAccelerationValues::AbstractVector{SVector{D,T}},
    Centre::SVector{D,T};
    GlobalGravityEnabled::Bool=true,
) where {D,T<:AbstractFloat}
    @assert !isempty(Times) "Fluid acceleration input timeline cannot be empty."
    @assert length(Times) == length(LinearAccelerationValues) "Fluid acceleration input times and linear acceleration values must have the same length."
    @assert length(Times) == length(AngularAccelerationValues) "Fluid acceleration input times and angular acceleration values must have the same length."
    times_vector = collect(Times)
    linear_acceleration_vector = collect(LinearAccelerationValues)
    angular_acceleration_vector = collect(AngularAccelerationValues)
    @assert issorted(times_vector) "Fluid acceleration input times must be sorted in ascending order."
    linear_velocity_vector = IntegrateAccelerationTimeline(times_vector, linear_acceleration_vector)
    angular_velocity_vector = IntegrateAccelerationTimeline(times_vector, angular_acceleration_vector)
    return FluidAccelerationInputSeries{D,T}(
        times_vector,
        linear_acceleration_vector,
        angular_acceleration_vector,
        linear_velocity_vector,
        angular_velocity_vector,
        Centre,
        GlobalGravityEnabled,
        1,
    )
end

function FluidAccelerationInputByGroup(
    Models::AbstractVector{Union{Nothing,FluidAccelerationInputSeries{D,T}}},
) where {D,T<:AbstractFloat}
    models_vector = collect(Models)
    zero_state = ZeroFluidAccelerationInputState(Val(D), T)
    current_values = fill(zero_state, length(models_vector))
    return FluidAccelerationInputByGroup{D,T}(models_vector, current_values)
end

@inline DefaultLinearAccelerationColumns(::Val{2}) = (Symbol("LinearAccX"), Symbol("LinearAccZ"))
@inline DefaultLinearAccelerationColumns(::Val{3}) = (Symbol("LinearAccX"), Symbol("LinearAccY"), Symbol("LinearAccZ"))
@inline DefaultLinearAccelerationColumns(::Val{D}) where {D} = throw(ArgumentError("Default linear acceleration columns are only defined for D=2 or D=3; provide `AccelerationColumns` explicitly."))
@inline DefaultAngularAccelerationColumns(::Val{2}) = (Symbol("AngularAccX"), Symbol("AngularAccZ"))
@inline DefaultAngularAccelerationColumns(::Val{3}) = (Symbol("AngularAccX"), Symbol("AngularAccY"), Symbol("AngularAccZ"))
@inline DefaultAngularAccelerationColumns(::Val{D}) where {D} = throw(ArgumentError("Default angular acceleration columns are only defined for D=2 or D=3; provide `AngularAccelerationColumns` explicitly."))

function LoadFluidAccelerationSeriesCSV(
    FilePath::AbstractString,
    ::Val{D},
    ::Type{T};
    Delimiter=';',
    TimeColumn::Symbol=Symbol("#Time"),
    AccelerationColumns::Union{Nothing,NTuple{D,Symbol}}=nothing,
) where {D,T<:AbstractFloat}
    columns = isnothing(AccelerationColumns) ? DefaultLinearAccelerationColumns(Val(D)) : AccelerationColumns
    times = T[]
    values = SVector{D,T}[]

    for row in CSV.File(FilePath; delim=Delimiter)
        push!(times, T(getproperty(row, TimeColumn)))
        sample = ntuple(i -> T(getproperty(row, columns[i])), D)
        push!(values, SVector{D,T}(sample))
    end

    return FluidAccelerationSeries(times, values)
end

function LoadFluidAccelerationByGroupCSV(
    GroupFiles::AbstractDict{<:Integer,<:AbstractString},
    GroupModelCount::Integer,
    ::Val{D},
    ::Type{T};
    Delimiter=';',
    TimeColumn::Symbol=Symbol("#Time"),
    AccelerationColumns::Union{Nothing,NTuple{D,Symbol}}=nothing,
) where {D,T<:AbstractFloat}
    @assert GroupModelCount >= 1 "GroupModelCount must be at least 1."
    models = Vector{Union{Nothing,FluidAccelerationSeries{D,T}}}(undef, GroupModelCount)
    fill!(models, nothing)

    for (group, file_path) in GroupFiles
        group_index = Int(group)
        @assert 1 <= group_index <= GroupModelCount "Group marker $(group_index) is outside 1:$(GroupModelCount)."
        models[group_index] = LoadFluidAccelerationSeriesCSV(
            file_path, Val(D), T;
            Delimiter=Delimiter,
            TimeColumn=TimeColumn,
            AccelerationColumns=AccelerationColumns,
        )
    end

    return FluidAccelerationByGroup(models)
end

function LoadFluidAccelerationInputSeriesCSV(
    FilePath::AbstractString,
    ::Val{D},
    ::Type{T};
    Delimiter=';',
    TimeColumn::Symbol=Symbol("#Time"),
    LinearAccelerationColumns::Union{Nothing,NTuple{D,Symbol}}=nothing,
    AngularAccelerationColumns::Union{Nothing,NTuple{D,Symbol}}=nothing,
    AccelerationCentre=nothing,
    GlobalGravityEnabled::Bool=true,
) where {D,T<:AbstractFloat}
    linear_columns = isnothing(LinearAccelerationColumns) ? DefaultLinearAccelerationColumns(Val(D)) : LinearAccelerationColumns
    angular_columns = isnothing(AngularAccelerationColumns) ? DefaultAngularAccelerationColumns(Val(D)) : AngularAccelerationColumns
    centre = isnothing(AccelerationCentre) ? zero(SVector{D,T}) : ToAccelerationCentre(AccelerationCentre, Val(D), T)

    times = T[]
    linear_values = SVector{D,T}[]
    angular_values = SVector{D,T}[]

    for row in CSV.File(FilePath; delim=Delimiter)
        push!(times, T(getproperty(row, TimeColumn)))
        linear_sample = ntuple(i -> T(getproperty(row, linear_columns[i])), D)
        angular_sample = ntuple(i -> T(getproperty(row, angular_columns[i])), D)
        push!(linear_values, SVector{D,T}(linear_sample))
        push!(angular_values, SVector{D,T}(angular_sample))
    end

    return FluidAccelerationInputSeries(
        times,
        linear_values,
        angular_values,
        centre;
        GlobalGravityEnabled=GlobalGravityEnabled,
    )
end

function LoadFluidAccelerationInputByGroupCSV(
    GroupFiles::AbstractDict{<:Integer,<:AbstractString},
    GroupModelCount::Integer,
    ::Val{D},
    ::Type{T};
    Delimiter=';',
    TimeColumn::Symbol=Symbol("#Time"),
    LinearAccelerationColumns::Union{Nothing,NTuple{D,Symbol}}=nothing,
    AngularAccelerationColumns::Union{Nothing,NTuple{D,Symbol}}=nothing,
    DefaultAccelerationCentre=nothing,
    DefaultGlobalGravityEnabled::Bool=true,
    AccelerationCentreByGroup::AbstractDict{<:Integer,<:Any}=Dict{Int,Any}(),
    GlobalGravityEnabledByGroup::AbstractDict{<:Integer,<:Bool}=Dict{Int,Bool}(),
) where {D,T<:AbstractFloat}
    @assert GroupModelCount >= 1 "GroupModelCount must be at least 1."
    default_centre = isnothing(DefaultAccelerationCentre) ? zero(SVector{D,T}) : ToAccelerationCentre(DefaultAccelerationCentre, Val(D), T)
    models = Vector{Union{Nothing,FluidAccelerationInputSeries{D,T}}}(undef, GroupModelCount)
    fill!(models, nothing)

    for (group, file_path) in GroupFiles
        group_index = Int(group)
        @assert 1 <= group_index <= GroupModelCount "Group marker $(group_index) is outside 1:$(GroupModelCount)."
        centre = haskey(AccelerationCentreByGroup, group_index) ?
                 ToAccelerationCentre(AccelerationCentreByGroup[group_index], Val(D), T) :
                 default_centre
        global_gravity_enabled = get(GlobalGravityEnabledByGroup, group_index, DefaultGlobalGravityEnabled)
        models[group_index] = LoadFluidAccelerationInputSeriesCSV(
            file_path,
            Val(D),
            T;
            Delimiter=Delimiter,
            TimeColumn=TimeColumn,
            LinearAccelerationColumns=LinearAccelerationColumns,
            AngularAccelerationColumns=AngularAccelerationColumns,
            AccelerationCentre=centre,
            GlobalGravityEnabled=global_gravity_enabled,
        )
    end

    return FluidAccelerationInputByGroup(models)
end

@inline function EvaluateFluidAcceleration(::Nothing, ::Type{T}, ::Val{D}, _time) where {D,T}
    return zero(SVector{D,T})
end

function EvaluateFluidAcceleration(model::FluidAccelerationSeries{D,T}, ::Type{T}, ::Val{D}, time::T) where {D,T<:AbstractFloat}
    TimeSamples = model.Times
    AccelerationSamples = model.Values
    LastIndex = length(TimeSamples)

    if LastIndex == 1
        model.Cursor = 1
        return AccelerationSamples[LastIndex]
    end

    i = LocateTimeInterval(TimeSamples, model.Cursor, time)
    model.Cursor = i

    @inbounds begin
        t0 = TimeSamples[i]
        t1 = TimeSamples[i + 1]
        a0 = AccelerationSamples[i]
        a1 = AccelerationSamples[i + 1]
        dt = t1 - t0
        if iszero(dt)
            return a1
        end

        alpha = clamp((time - t0) / dt, zero(T), one(T))
        return a0 + (a1 - a0) * alpha
    end
end

function EvaluateFluidAcceleration(model::FluidAccelerationByGroup{D,T}, ::Type{T}, ::Val{D}, time::T) where {D,T<:AbstractFloat}
    dim_val = Val(D)
    zero_acc = zero(SVector{D,T})
    @inbounds for group_index in eachindex(model.Models)
        series = model.Models[group_index]
        model.CurrentValues[group_index] = series === nothing ? zero_acc : EvaluateFluidAcceleration(series, T, dim_val, time)
    end
    return model.CurrentValues
end

function EvaluateFluidAcceleration(model::FluidAccelerationInputSeries{D,T}, ::Type{T}, ::Val{D}, time::T) where {D,T<:AbstractFloat}
    TimeSamples = model.Times
    LastIndex = length(TimeSamples)

    if LastIndex == 1
        model.Cursor = 1
        @inbounds return FluidAccelerationInputState{D,T}(
            model.LinearAccelerationValues[1],
            model.AngularAccelerationValues[1],
            model.Centre,
            model.AngularVelocityValues[1],
            model.LinearVelocityValues[1],
            model.GlobalGravityEnabled,
        )
    end

    i = LocateTimeInterval(TimeSamples, model.Cursor, time)
    model.Cursor = i

    @inbounds begin
        t0 = TimeSamples[i]
        t1 = TimeSamples[i + 1]
        dt = t1 - t0
        if iszero(dt)
            return FluidAccelerationInputState{D,T}(
                model.LinearAccelerationValues[i + 1],
                model.AngularAccelerationValues[i + 1],
                model.Centre,
                model.AngularVelocityValues[i + 1],
                model.LinearVelocityValues[i + 1],
                model.GlobalGravityEnabled,
            )
        end

        alpha = clamp((time - t0) / dt, zero(T), one(T))
        linear_acceleration = model.LinearAccelerationValues[i] + (model.LinearAccelerationValues[i + 1] - model.LinearAccelerationValues[i]) * alpha
        angular_acceleration = model.AngularAccelerationValues[i] + (model.AngularAccelerationValues[i + 1] - model.AngularAccelerationValues[i]) * alpha
        linear_velocity = model.LinearVelocityValues[i] + (model.LinearVelocityValues[i + 1] - model.LinearVelocityValues[i]) * alpha
        angular_velocity = model.AngularVelocityValues[i] + (model.AngularVelocityValues[i + 1] - model.AngularVelocityValues[i]) * alpha
        return FluidAccelerationInputState{D,T}(
            linear_acceleration,
            angular_acceleration,
            model.Centre,
            angular_velocity,
            linear_velocity,
            model.GlobalGravityEnabled,
        )
    end
end

function EvaluateFluidAcceleration(model::FluidAccelerationInputByGroup{D,T}, ::Type{T}, ::Val{D}, time::T) where {D,T<:AbstractFloat}
    dim_val = Val(D)
    zero_state = ZeroFluidAccelerationInputState(dim_val, T)
    @inbounds for group_index in eachindex(model.Models)
        series = model.Models[group_index]
        model.CurrentValues[group_index] = series === nothing ? zero_state : EvaluateFluidAcceleration(series, T, dim_val, time)
    end
    return model.CurrentValues
end

@inline FluidAccelerationForGroup(FluidAcceleration::SVector{D,T}, _GroupMarker) where {D,T<:AbstractFloat} = FluidAcceleration

@inline function FluidAccelerationForGroup(FluidAcceleration::AbstractVector{SVector{D,T}}, GroupMarker::Integer) where {D,T<:AbstractFloat}
    @inbounds return FluidAcceleration[Int(GroupMarker)]
end

@inline FluidAccelerationForGroup(FluidAcceleration::SVector{D,T}, GroupMarker::Integer, _Position, _Velocity, _GravityVector) where {D,T<:AbstractFloat} = FluidAccelerationForGroup(FluidAcceleration, GroupMarker)

@inline function FluidAccelerationForGroup(FluidAcceleration::AbstractVector{SVector{D,T}}, GroupMarker::Integer, _Position, _Velocity, _GravityVector) where {D,T<:AbstractFloat}
    return FluidAccelerationForGroup(FluidAcceleration, GroupMarker)
end

@inline function HasAngularAcceleration(AngularAcceleration::SVector{D,T}) where {D,T<:AbstractFloat}
    @inbounds for i in eachindex(AngularAcceleration)
        if !iszero(AngularAcceleration[i])
            return true
        end
    end
    return false
end

@inline function FluidAccelerationAngularContribution3D(
    State::FluidAccelerationInputState{3,T},
    Position::SVector{3,T},
    Velocity::SVector{3,T},
) where {T<:AbstractFloat}
    if !HasAngularAcceleration(State.AngularAcceleration)
        return zero(SVector{3,T})
    end

    dc = Position - State.Centre
    angular_acceleration = State.AngularAcceleration
    angular_velocity = State.AngularVelocity
    linear_velocity = State.LinearVelocity

    # (dω/dt) × (rᵢ-r)
    term1 = SVector{3,T}(
        (angular_acceleration[2] * dc[3]) - (angular_acceleration[3] * dc[2]),
        (angular_acceleration[3] * dc[1]) - (angular_acceleration[1] * dc[3]),
        (angular_acceleration[1] * dc[2]) - (angular_acceleration[2] * dc[1]),
    )

    # ω × (ω × (rᵢ-r))
    inner = SVector{3,T}(
        (angular_velocity[2] * dc[3]) - (angular_velocity[3] * dc[2]),
        (angular_velocity[3] * dc[1]) - (angular_velocity[1] * dc[3]),
        (angular_velocity[1] * dc[2]) - (angular_velocity[2] * dc[1]),
    )
    term2 = SVector{3,T}(
        (angular_velocity[2] * inner[3]) - (angular_velocity[3] * inner[2]),
        (angular_velocity[3] * inner[1]) - (angular_velocity[1] * inner[3]),
        (angular_velocity[1] * inner[2]) - (angular_velocity[2] * inner[1]),
    )

    # 2ω × (vᵢ-v) with the same component form as DualSPHysics.
    two = T(2)
    term3 = SVector{3,T}(
        ((two * angular_velocity[2]) * Velocity[3]) - ((two * angular_velocity[3]) * (Velocity[2] - linear_velocity[2])),
        ((two * angular_velocity[3]) * Velocity[1]) - ((two * angular_velocity[1]) * (Velocity[3] - linear_velocity[3])),
        ((two * angular_velocity[1]) * Velocity[2]) - ((two * angular_velocity[2]) * (Velocity[1] - linear_velocity[1])),
    )

    return term1 + term2 + term3
end

@inline function To3DFrom2D(Value::SVector{2,T}) where {T<:AbstractFloat}
    return SVector{3,T}(Value[1], zero(T), Value[2])
end

@inline function To2DFrom3D(Value::SVector{3,T}) where {T<:AbstractFloat}
    return SVector{2,T}(Value[1], Value[3])
end

@inline function FluidAccelerationAngularContribution2D(
    State::FluidAccelerationInputState{2,T},
    Position::SVector{2,T},
    Velocity::SVector{2,T},
) where {T<:AbstractFloat}
    state3d = FluidAccelerationInputState{3,T}(
        To3DFrom2D(State.LinearAcceleration),
        To3DFrom2D(State.AngularAcceleration),
        To3DFrom2D(State.Centre),
        To3DFrom2D(State.AngularVelocity),
        To3DFrom2D(State.LinearVelocity),
        State.GlobalGravityEnabled,
    )
    return To2DFrom3D(FluidAccelerationAngularContribution3D(state3d, To3DFrom2D(Position), To3DFrom2D(Velocity)))
end

@inline function FluidAccelerationForGroup(
    FluidAcceleration::FluidAccelerationInputState{3,T},
    _GroupMarker::Integer,
    Position::SVector{3,T},
    Velocity::SVector{3,T},
    GravityVector::SVector{3,T},
) where {T<:AbstractFloat}
    extra_acceleration = FluidAcceleration.LinearAcceleration
    if !FluidAcceleration.GlobalGravityEnabled
        extra_acceleration -= GravityVector
    end
    extra_acceleration += FluidAccelerationAngularContribution3D(FluidAcceleration, Position, Velocity)
    return extra_acceleration
end

@inline function FluidAccelerationForGroup(
    FluidAcceleration::FluidAccelerationInputState{2,T},
    _GroupMarker::Integer,
    Position::SVector{2,T},
    Velocity::SVector{2,T},
    GravityVector::SVector{2,T},
) where {T<:AbstractFloat}
    extra_acceleration = FluidAcceleration.LinearAcceleration
    if !FluidAcceleration.GlobalGravityEnabled
        extra_acceleration -= GravityVector
    end
    extra_acceleration += FluidAccelerationAngularContribution2D(FluidAcceleration, Position, Velocity)
    return extra_acceleration
end

@inline function FluidAccelerationForGroup(
    FluidAcceleration::AbstractVector{FluidAccelerationInputState{D,T}},
    GroupMarker::Integer,
    Position::SVector{D,T},
    Velocity::SVector{D,T},
    GravityVector::SVector{D,T},
) where {D,T<:AbstractFloat}
    @inbounds return FluidAccelerationForGroup(FluidAcceleration[Int(GroupMarker)], GroupMarker, Position, Velocity, GravityVector)
end

@inline function LocateTimeInterval(TimeSamples::AbstractVector{T}, Cursor::Int, time::T) where {T<:AbstractFloat}
    LastInterval = length(TimeSamples) - 1
    IntervalIndex = clamp(Cursor, 1, LastInterval)
    @inbounds begin
        while IntervalIndex < LastInterval && time > TimeSamples[IntervalIndex + 1]
            IntervalIndex += 1
        end
        while IntervalIndex > 1 && time < TimeSamples[IntervalIndex]
            IntervalIndex -= 1
        end
    end
    return IntervalIndex
end

@inline function Rotate2D(Vector2::SVector{2,T}, CosTheta::T, SinTheta::T) where {T<:AbstractFloat}
    return SVector{2,T}(
        CosTheta * Vector2[1] - SinTheta * Vector2[2],
        SinTheta * Vector2[1] + CosTheta * Vector2[2],
    )
end

@inline function RotationalVelocity2D(RelativePosition::SVector{2,T}, AngularSpeed::T) where {T<:AbstractFloat}
    return SVector{2,T}(-AngularSpeed * RelativePosition[2], AngularSpeed * RelativePosition[1])
end

end # module FluidAcceleration
