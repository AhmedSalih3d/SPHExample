module PreProcess

export LoadBoundaryNormals, AllocateDataStructures, AllocateSupportDataStructures

using CSV
using StaticArrays
using StructArrays

using ..SimulationGeometry
using ..SimulationMetaDataConfiguration: SimulationMetaData, StoreKernelOutput, NoMDBC, position_float_type

"""
    AllocateDataStructures(SimGeometry, SimMetaData)
    AllocateDataStructures(SimGeometry; position_type = FloatType, cells = true,
                           kernel_output = true, ghost_data = true)

Copy the stored particles of every `SPHGeometry` into a host `StructArray`.
`Position` and `GhostPoints` use `position_type` (`Float64` with
`GPUDoublePosition`, see `position_float_type`), and every other float field
uses `FloatType`.

The single-argument form retains all optional host fields (`Cells`, `Kernel`,
`KernelGradient`, `GhostPoints`, and `GhostNormals`) by default. With
`SimMetaData` and `GPUCompactBuffers = true`, inactive optional host columns are
omitted: `Cells` unless `ExportGridCells` is enabled, `Kernel` and
`KernelGradient` unless `KMode === StoreKernelOutput`, and `GhostPoints` and
`GhostNormals` unless `BMode !== NoMDBC` (or an input geometry explicitly
carries ghost fields). Setting `GPUCompactBuffers = false` retains all host
columns.
"""
function AllocateDataStructures(SimGeometry::Vector{<:SPHGeometry{Dimensions, FloatType}},
                                SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode}) where {Dimensions, FloatType, SMode, KMode, BMode}
    compact = SimMetaData.GPUCompactBuffers
    has_geom_ghosts = any(geom -> hasproperty(geom.Particles, :GhostPoints) ||
                                  hasproperty(geom.Particles, :GhostNormals), SimGeometry)
    return AllocateDataStructures(SimGeometry;
        position_type = position_float_type(SimMetaData),
        cells = !compact || SimMetaData.ExportGridCells,
        kernel_output = !compact || KMode === StoreKernelOutput,
        ghost_data = !compact || BMode !== NoMDBC || has_geom_ghosts)
end

function AllocateDataStructures(SimGeometry::Vector{<:SPHGeometry{Dimensions, FloatType}};
                                position_type::Type{TP} = FloatType,
                                cells::Bool = true,
                                kernel_output::Bool = true,
                                ghost_data::Bool = true) where {Dimensions, FloatType, TP}
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

    Acceleration = zeros(VectorType, NumberOfPoints)
    Velocity     = zeros(VectorType, NumberOfPoints)
    Pressureᵢ    = zeros(FloatType, NumberOfPoints)

    columns = (
        (cells ? (Cells = fill(zero(CartesianIndex{Dimensions}), NumberOfPoints),) : ())...,
        (kernel_output ? (Kernel = zeros(FloatType, NumberOfPoints),
                          KernelGradient = zeros(VectorType, NumberOfPoints)) : ())...,
        Position = Position,
        Acceleration = Acceleration,
        Velocity = Velocity,
        Density = Density,
        Pressure = Pressureᵢ,
        ID = Idp,
        Type = Types,
        GroupMarker = GroupMarker,
        (ghost_data ? (GhostPoints = zeros(PositionType, NumberOfPoints),
                       GhostNormals = zeros(VectorType, NumberOfPoints)) : ())...,
    )

    SimParticles = StructArray(columns)

    offset = 0
    for geom in SimGeometry
        n = length(geom.Particles)
        for field in (:Velocity, :GhostPoints, :GhostNormals)
            if hasproperty(geom.Particles, field) && hasproperty(SimParticles, field)
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
