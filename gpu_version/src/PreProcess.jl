module PreProcess

export LoadBoundaryNormals, AllocateDataStructures, AllocateSupportDataStructures

using CSV
using StaticArrays
using StructArrays

using ..SimulationGeometry
using ..SimulationMetaDataConfiguration: SimulationMetaData, position_float_type

# `T` is the working precision (densities), `TP` the precision of the positions.
function LoadSpecificCSV(::Val{D}, ::Type{T}, ::Type{TP}, particle_type::ParticleType,
                         particle_group_marker::Int,
                         specific_csv::String) where {D, T, TP}
    file  = CSV.File(specific_csv)
    nrows = length(file)

    points       = Vector{SVector{D, TP}}(undef, nrows)
    density      = Vector{T}(undef, nrows)
    types        = Vector{ParticleType}(undef, nrows)
    group_marker = Vector{Int}(undef, nrows)
    idp          = Vector{Int}(undef, nrows)

    i = 1
    for row ∈ file
        P1   = row[Symbol("Points:0")]
        P2   = row[Symbol("Points:1")]
        P3   = row[Symbol("Points:2")]
        Rhop = row[Symbol("Rhop")]
        Idp  = row[Symbol("Idp")] + 1

        points[i] = if D == 3
            SVector{3,TP}(P1, P2, P3)
        else
            SVector{2,TP}(P1, P3)
        end

        density[i]      = Rhop
        types[i]        = particle_type
        group_marker[i] = particle_group_marker
        idp[i]          = Idp
        i += 1
    end

    return points, density, types, group_marker, idp
end

"""
    AllocateDataStructures(SimGeometry, SimMetaData)
    AllocateDataStructures(SimGeometry; position_type = FloatType)

Load the particles of every SPHGeometry into a host `StructArray`. The device
particle container always carries the ghost node and kernel output fields,
so the mode types of the meta data do not change the host allocation; the
meta data selects the precision of the positions (`Float64` with
`GPUDoublePosition`, see `position_float_type`). `Position` and
`GhostPoints` are stored in that precision, every other field in `FloatType`.
"""
AllocateDataStructures(SimGeometry::Vector{<:SPHGeometry{Dimensions, FloatType}},
                       SimMetaData::SimulationMetaData{Dimensions, FloatType}) where {Dimensions, FloatType} =
    AllocateDataStructures(SimGeometry; position_type = position_float_type(SimMetaData))

function AllocateDataStructures(SimGeometry::Vector{<:SPHGeometry{Dimensions, FloatType}};
                                position_type::Type{TP} = FloatType) where {Dimensions, FloatType, TP}
    Position    = Vector{SVector{Dimensions, TP}}()
    Density     = Vector{FloatType}()
    Types       = Vector{ParticleType}()
    GroupMarker = Vector{UInt}()
    Idp         = Vector{Int}()
    
    for geom in SimGeometry
        particle_type         = geom.Type
        particle_group_marker = geom.GroupMarker
        specific_csv          = geom.CSVFile

        points, density, types, group_marker, idp =
            LoadSpecificCSV(Val(Dimensions), FloatType, TP, particle_type,
                           particle_group_marker, specific_csv)

        sizehint!(Position,    length(Position)    + length(points))
        sizehint!(Density,     length(Density)     + length(density))
        sizehint!(Types,       length(Types)       + length(types))
        sizehint!(GroupMarker, length(GroupMarker) + length(group_marker))
        sizehint!(Idp,         length(Idp)         + length(idp))

        append!(Position,    points)
        append!(Density,     density)
        append!(Types,       types)
        append!(GroupMarker, group_marker)
        append!(Idp,         idp)
    end

    NumberOfPoints = length(Position)
    PositionType   = eltype(Position)
    VectorType     = SVector{Dimensions, FloatType}

    Acceleration    = zeros(VectorType, NumberOfPoints)
    Velocity        = zeros(VectorType, NumberOfPoints)
    Kernel          = zeros(FloatType, NumberOfPoints)
    KernelGradient  = zeros(VectorType, NumberOfPoints)
    GhostPoints     = zeros(PositionType, NumberOfPoints)
    GhostNormals    = zeros(VectorType, NumberOfPoints)

    Pressureᵢ      = zeros(FloatType, NumberOfPoints)
    
    Cells          = fill(zero(CartesianIndex{Dimensions}), NumberOfPoints)

    SimParticles = StructArray((Cells = Cells, Kernel = Kernel, KernelGradient = KernelGradient, Position=Position, Acceleration=Acceleration, Velocity=Velocity, Density=Density, Pressure=Pressureᵢ, ID = Idp , Type = Types, GroupMarker = GroupMarker, GhostPoints = GhostPoints, GhostNormals=GhostNormals))

    sort!(SimParticles, by = p -> p.ID)

    return SimParticles
end

function AllocateSupportDataStructures(Position)

    NumberOfPoints           = length(Position)
    PositionType             = eltype(Position)
    PositionUnderlyingType   = eltype(PositionType)

    dρdtI           = zeros(PositionUnderlyingType, NumberOfPoints)
    Velocityₙ⁺      = zeros(PositionType, NumberOfPoints)
    Positionₙ⁺      = zeros(PositionType, NumberOfPoints)
    ρₙ⁺             = zeros(PositionUnderlyingType, NumberOfPoints)

    ∇Cᵢ            = zeros(PositionType, NumberOfPoints)
    ∇◌rᵢ           = zeros(PositionUnderlyingType, NumberOfPoints)

    return dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ
end

function LoadBoundaryNormals(::Val{D}, ::Type{T}, path_mdbc) where {D, T}
    normals      = Vector{SVector{D,T}}()
    points       = Vector{SVector{D,T}}()
    ghost_points = Vector{SVector{D,T}}()

    for row ∈ CSV.File(path_mdbc)
        if D == 3
            normal = SVector{D,T}(row[Symbol("Normal:0")],
                                   row[Symbol("Normal:1")],
                                   row[Symbol("Normal:2")])
            point  = SVector{D,T}(row[Symbol("Points:0")],
                                   row[Symbol("Points:1")],
                                   row[Symbol("Points:2")])
        elseif D == 2
            normal = SVector{D,T}(row[Symbol("Normal:0")],
                                   row[Symbol("Normal:2")])
            point  = SVector{D,T}(row[Symbol("Points:0")],
                                   row[Symbol("Points:2")])
        end

        push!(normals, normal)
        push!(points, point)
        push!(ghost_points, point + normal)
    end

    return points, ghost_points, normals
end

end
