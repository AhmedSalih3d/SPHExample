module PointMeasurements

using LinearAlgebra
using StaticArrays

using ..SPHKernels: Wᵢⱼ
using ..SPHNeighborList: MapFloor

export PointMeasure, PointMeasureFieldNames, PointMeasureOutputVariableNames,
       FillPointMeasureData!, FillPointMeasureSnapshot!

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

function PointMeasureOutputVariableNames(PointMeasures::Vector{<:PointMeasure})
    OutputNames = String[]
    Seen = Set{String}()
    for Measure in PointMeasures
        for Variable in Measure.Variables
            if !(Variable in Seen)
                push!(OutputNames, Variable)
                push!(Seen, Variable)
            end
        end
    end
    return OutputNames
end

function FillPointMeasureData!(FieldData, PointMeasures::Vector{<:PointMeasure},
                               SimParticles, SimKernel, Dimensions;
                               field_map, vector_fields,
                               neighbor_data = nothing)
    resize!(FieldData, 0)
    if isempty(PointMeasures)
        return FieldData
    end
    if neighbor_data === nothing
        error("PointMeasure evaluation requires neighbor data; pass `neighbor_data` from the cell list.")
    end

    for Measure in PointMeasures
        AppendPointMeasureResults!(FieldData, Measure, SimParticles, SimKernel, Dimensions;
                                   field_map = field_map,
                                   vector_fields = vector_fields,
                                   neighbor_data = neighbor_data)
    end

    return FieldData
end

function AppendPointMeasureResults!(FieldData, Measure::PointMeasure{D, T},
                                    SimParticles, SimKernel, Dimensions;
                                    field_map, vector_fields,
                                    neighbor_data) where {D, T}
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

    cell_dict = neighbor_data.cell_dict
    particle_ranges = neighbor_data.particle_ranges
    full_stencil = neighbor_data.full_stencil
    cell_index = CartesianIndex(map(x -> MapFloor(x, SimKernel.H⁻¹), Tuple(Position)))

    HasCandidates = false
    @inbounds for offset in full_stencil
        neighbor_cell = cell_index + offset
        neighbor_index = get(cell_dict, neighbor_cell, 0)
        if neighbor_index != 0
            start_index = particle_ranges[neighbor_index]
            end_index = particle_ranges[neighbor_index + 1] - 1
            if start_index <= end_index
                HasCandidates = true
                for particle_index in start_index:end_index
                    Offset = Position - Positions[particle_index]
                    Distance = norm(Offset)
                    if Distance < NearestDistance
                        NearestDistance = Distance
                        NearestIndex = particle_index
                    end
                    q = clamp(Distance * SimKernel.h⁻¹, zero(T), T(2))
                    if q <= T(2)
                        Weight = Wᵢⱼ(SimKernel, q)
                        if Weight != zero(T)
                            WeightSum += Weight
                            for VariableIndex in eachindex(Variables)
                                if InterpolateMask[VariableIndex]
                                    Accumulators[VariableIndex] += Sources[VariableIndex][particle_index] * Weight
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    if !HasCandidates
        error("PointMeasure found no neighbor candidates; adjust probe position or cell list.")
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

function FillPointMeasureSnapshot!(snapshot_positions, snapshot_output_data,
                                   PointMeasures::Vector{<:PointMeasure},
                                   OutputVariables::Vector{String},
                                   SimParticles, SimKernel, Dimensions;
                                   field_map, vector_fields,
                                   neighbor_data = nothing)
    if isempty(PointMeasures)
        return snapshot_positions, snapshot_output_data
    end
    if neighbor_data === nothing
        error("PointMeasure evaluation requires neighbor data; pass `neighbor_data` from the cell list.")
    end

    resize!(snapshot_positions, length(PointMeasures))
    for Output in snapshot_output_data
        resize!(Output, length(PointMeasures))
        fill!(Output, zero(eltype(Output)))
    end

    output_index = Dict(name => idx for (idx, name) in pairs(OutputVariables))

    for (MeasureIndex, Measure) in pairs(PointMeasures)
        snapshot_positions[MeasureIndex] = To3DPoint(Val(Dimensions), Measure.Position)

        buffer = Any[]
        AppendPointMeasureResults!(buffer, Measure, SimParticles, SimKernel, Dimensions;
                                   field_map = field_map,
                                   vector_fields = vector_fields,
                                   neighbor_data = neighbor_data)

        data_index = 2
        for Variable in Measure.Variables
            index = output_index[Variable]
            snapshot_output_data[index][MeasureIndex] = buffer[data_index]
            data_index += 1
        end
    end

    return snapshot_positions, snapshot_output_data
end

end
