module PreProcess

export LoadBoundaryNormals, AllocateDataStructures, AllocateSupportDataStructures

using CSV
using StaticArrays
using StructArrays

using ..SimulationGeometry
using ..SimulationMetaDataConfiguration: SimulationMetaData, position_float_type

"""
    AllocateDataStructures(SimGeometry, SimMetaData)
    AllocateDataStructures(SimGeometry; position_type = FloatType)

Copy the stored particles of every SPHGeometry into a host `StructArray`. The device
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
    
    explicit_ids = Int[]
    for geom in SimGeometry
        hasproperty(geom.Particles, :ID) && append!(explicit_ids, geom.Particles.ID)
    end
    length(unique(explicit_ids)) == length(explicit_ids) ||
        throw(ArgumentError("particle IDs must be unique across geometry groups"))
    next_id = isempty(explicit_ids) ? 1 : maximum(explicit_ids) + 1
    for geom in SimGeometry
        particles = geom.Particles
        n = length(particles)
        append!(Position, SVector{Dimensions, TP}.(particles.Position))
        append!(Density, particles.Density)
        append!(Types, fill(geom.Type, n))
        append!(GroupMarker, fill(geom.GroupMarker, n))
        if hasproperty(particles, :ID)
            append!(Idp, particles.ID)
        else
            append!(Idp, next_id:(next_id + n - 1))
            next_id += n
        end
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

    offset = 0
    for geom in SimGeometry
        n = length(geom.Particles)
        for field in (:Velocity, :GhostPoints, :GhostNormals)
            if hasproperty(geom.Particles, field)
                copyto!(getproperty(SimParticles, field), offset + 1,
                        getproperty(geom.Particles, field), 1, n)
            end
        end
        offset += n
    end

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
