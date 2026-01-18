module PointMeasurements

using LinearAlgebra
using StaticArrays

using ..SPHKernels: Wᵢⱼ

export PointMeasure, PointMeasureFieldNames, FillPointMeasureData!

"""
    PointMeasure(Name, Position, Variables)

Define a probe point at `Position` for measuring `Variables` during output.
"""
struct PointMeasure{Dimensions, FloatType}
    Name::String
    Position::SVector{Dimensions, FloatType}
    Variables::Vector{String}
end

@inline function To3DPoint(::Val{2}, Point::SVector{2, T}) where {T}
    return SVector{3, T}(Point[1], Point[2], zero(T))
end

@inline function To3DPoint(::Val{3}, Point::SVector{3, T}) where {T}
    return Point
end

function PointMeasureFieldNames(PointMeasures::Vector{<:PointMeasure})
    FieldNames = String[]
    for Measure in PointMeasures
        push!(FieldNames, "PointMeasure_$(Measure.Name)_Position")
        for Variable in Measure.Variables
            push!(FieldNames, "PointMeasure_$(Measure.Name)_$(Variable)")
        end
    end
    return FieldNames
end

function FillPointMeasureData!(FieldData, PointMeasures::Vector{<:PointMeasure},
                               SimParticles, SimKernel, Dimensions;
                               field_map, vector_fields)
    resize!(FieldData, 0)
    if isempty(PointMeasures)
        return FieldData
    end

    for Measure in PointMeasures
        AppendPointMeasureResults!(FieldData, Measure, SimParticles, SimKernel, Dimensions;
                                   field_map = field_map, vector_fields = vector_fields)
    end

    return FieldData
end

function AppendPointMeasureResults!(FieldData, Measure::PointMeasure{D, T},
                                    SimParticles, SimKernel, Dimensions;
                                    field_map, vector_fields) where {D, T}
    Positions = SimParticles.Position
    NumberOfPoints = length(Positions)
    if NumberOfPoints == 0
        error("PointMeasure cannot evaluate without particles.")
    end

    Position = Measure.Position
    push!(FieldData, To3DPoint(Val(Dimensions), Position))

    Variables = Measure.Variables
    Accumulators = Vector{Any}(undef, length(Variables))
    InterpolateMask = BitVector(undef, length(Variables))
    Sources = Vector{Any}(undef, length(Variables))

    for (Index, Variable) in pairs(Variables)
        Property = get(field_map, Variable, Symbol(Variable))
        if !hasproperty(SimParticles, Property)
            error("PointMeasure includes $(Variable) but SimParticles has no field $(Property).")
        end
        Source = getproperty(SimParticles, Property)
        Sources[Index] = Source
        if Variable in vector_fields
            Accumulators[Index] = zero(eltype(Source))
            InterpolateMask[Index] = eltype(eltype(Source)) <: AbstractFloat
        else
            Accumulators[Index] = zero(eltype(Source))
            InterpolateMask[Index] = eltype(Source) <: AbstractFloat
        end
    end

    NearestIndex = 1
    NearestDistance = typemax(T)
    WeightSum = zero(T)

    @inbounds for ParticleIndex in eachindex(Positions)
        Offset = Position - Positions[ParticleIndex]
        Distance = norm(Offset)
        if Distance < NearestDistance
            NearestDistance = Distance
            NearestIndex = ParticleIndex
        end
        q = clamp(Distance * SimKernel.h⁻¹, zero(T), T(2))
        if q <= T(2)
            Weight = Wᵢⱼ(SimKernel, q)
            if Weight != zero(T)
                WeightSum += Weight
                for VariableIndex in eachindex(Variables)
                    if InterpolateMask[VariableIndex]
                        Accumulators[VariableIndex] += Sources[VariableIndex][ParticleIndex] * Weight
                    end
                end
            end
        end
    end

    for (Index, Variable) in pairs(Variables)
        Source = Sources[Index]
        Value = if InterpolateMask[Index] && WeightSum > zero(T)
            Accumulators[Index] / WeightSum
        else
            Source[NearestIndex]
        end

        if Variable in vector_fields
            push!(FieldData, To3DPoint(Val(Dimensions), Value))
        else
            push!(FieldData, Value)
        end
    end

    return FieldData
end

end
