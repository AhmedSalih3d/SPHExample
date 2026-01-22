module PreProcess

export LoadBoundaryNormals, LoadMDBCNormals!, AllocateDataStructures, AllocateSupportDataStructures, AllocateThreadedArrays

using CSV
using StaticArrays
using StructArrays

using ..SimulationGeometry
using ..SimulationMetaDataConfiguration

@inline function LoadCSVPoint(::Val{2}, ::Type{T}, row) where {T}
    P1 = getproperty(row, Symbol("Points:0"))
    P3 = getproperty(row, Symbol("Points:2"))
    return SVector{2,T}(P1, P3)
end

@inline function LoadCSVPoint(::Val{3}, ::Type{T}, row) where {T}
    P1 = getproperty(row, Symbol("Points:0"))
    P2 = getproperty(row, Symbol("Points:1"))
    P3 = getproperty(row, Symbol("Points:2"))
    return SVector{3,T}(P1, P2, P3)
end

function LoadSpecificCSV(::Val{D}, ::Type{T}, particle_type::ParticleType, particle_group_marker::Int, specific_csv::String) where {D, T}
    csv_file = CSV.File(specific_csv)

    nrows = length(csv_file)

    points       = Vector{SVector{D,T}}(undef, nrows)
    density      = Vector{T}(undef, nrows)
    types        = Vector{ParticleType}(undef, nrows)
    group_marker = Vector{Int}(undef, nrows)
    idp          = Vector{Int}(undef, nrows)

    for (i, row) ∈ enumerate(csv_file)
        Rhop = row.Rhop
        Idp  = row.Idp + 1
        points[i] = LoadCSVPoint(Val(D), T, row)

        density[i]      = Rhop
        types[i]        = particle_type
        group_marker[i] = particle_group_marker
        idp[i]          = Idp
    end

    return points, density, types, group_marker, idp
end

@inline function LoadBoundaryNormalPoint(::Val{2}, ::Type{T}, row) where {T}
    normal = SVector{2,T}(getproperty(row, Symbol("Normal:0")), getproperty(row, Symbol("Normal:2")))
    point  = SVector{2,T}(getproperty(row, Symbol("Points:0")), getproperty(row, Symbol("Points:2")))
    return normal, point
end

@inline function LoadBoundaryNormalPoint(::Val{3}, ::Type{T}, row) where {T}
    normal = SVector{3,T}(getproperty(row, Symbol("Normal:0")), getproperty(row, Symbol("Normal:1")), getproperty(row, Symbol("Normal:2")))
    point  = SVector{3,T}(getproperty(row, Symbol("Points:0")), getproperty(row, Symbol("Points:1")), getproperty(row, Symbol("Points:2")))
    return normal, point
end

function AllocateDataStructures(SimGeometry::Vector{<:Geometry{Dimensions, FloatType}}; RequireMDBC::Bool=false, RequireKernelOutput::Bool=false) where {Dimensions, FloatType}
    Position    = Vector{SVector{Dimensions, FloatType}}()
    Density     = Vector{FloatType}()
    Types       = Vector{ParticleType}()
    GroupMarker = Vector{UInt}()
    Idp         = Vector{Int}()
    
    for geom in SimGeometry
        particle_type         = geom.Type
        particle_group_marker = geom.GroupMarker
        specific_csv          = geom.CSVFile

        points, density, types, group_marker, idp =
            LoadSpecificCSV(Val(Dimensions), FloatType, particle_type,
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

    NumberOfPoints           = length(Position)
    PositionType             = eltype(Position)
    PositionUnderlyingType   = eltype(PositionType)

    sort_perm = sortperm(Idp)
    Position = Position[sort_perm]
    Density = Density[sort_perm]
    Types = Types[sort_perm]
    GroupMarker = GroupMarker[sort_perm]
    Idp = Idp[sort_perm]

    Acceleration    = zeros(PositionType, NumberOfPoints)
    Velocity        = zeros(PositionType, NumberOfPoints)
    Pressureᵢ      = zeros(PositionUnderlyingType, NumberOfPoints)
    
    Cells          = fill(zero(CartesianIndex{Dimensions}), NumberOfPoints)
    
    ParticleFields = (;
        Cells = Cells,
        Position = Position,
        Acceleration = Acceleration,
        Velocity = Velocity,
        Density = Density,
        Pressure = Pressureᵢ,
        Type = Types,
        GroupMarker = GroupMarker,
    )
    if RequireKernelOutput
        Kernel = zeros(PositionUnderlyingType, NumberOfPoints)
        KernelGradient = zeros(PositionType, NumberOfPoints)
        ParticleFields = merge(ParticleFields, (; Kernel = Kernel, KernelGradient = KernelGradient))
    end
    if RequireMDBC
        GhostPoints = zeros(PositionType, NumberOfPoints)
        ParticleFields = merge(ParticleFields, (; GhostPoints = GhostPoints))
    end
    if RequireMDBC
        GhostNormals = zeros(PositionType, NumberOfPoints)
        ParticleFields = merge(ParticleFields, (; GhostNormals = GhostNormals))
    end
    ParticleFields = merge(ParticleFields, (; ID = Idp))

    SimParticles = StructArray(ParticleFields)

    return SimParticles
end

function AllocateDataStructures(SimGeometry::Vector{<:Geometry{Dimensions, FloatType}}, SimMetaData::SimulationMetaData{Dimensions, FloatType}) where {Dimensions, FloatType}
    RequireMDBC = !(SimMetaData isa SimulationMetaData{Dimensions, FloatType, SMode, KMode, NoMDBC, LMode} where {SMode, KMode, LMode})
    RequireKernelOutput = !(SimMetaData isa SimulationMetaData{Dimensions, FloatType, SMode, NoKernelOutput, BMode, LMode} where {SMode, BMode, LMode})
    return AllocateDataStructures(SimGeometry; RequireMDBC = RequireMDBC, RequireKernelOutput = RequireKernelOutput)
end

function AllocateSupportDataStructures(::SimulationMetaData{D,T,NoShifting,K,B,L}, Position) where {D,T,K<:KernelOutputMode,
                                                                                                    B<:MDBCMode,
                                                                                                    L<:LogMode}

    NumberOfPoints         = length(Position)
    PositionType           = eltype(Position)
    PositionUnderlyingType = eltype(PositionType)

    dρdtI      = zeros(PositionUnderlyingType, NumberOfPoints)
    Velocityₙ⁺ = zeros(PositionType, NumberOfPoints)
    Positionₙ⁺ = zeros(PositionType, NumberOfPoints)
    ρₙ⁺        = zeros(PositionUnderlyingType, NumberOfPoints)

    ∇Cᵢ  = Vector{PositionType}(undef, 0)
    ∇◌rᵢ = Vector{PositionUnderlyingType}(undef, 0)

    return dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ
end

function AllocateSupportDataStructures(::SimulationMetaData{D,T,S,K,B,L}, Position) where {D,T,S<:ShiftingMode,
                                                                                           K<:KernelOutputMode,
                                                                                           B<:MDBCMode,
                                                                                           L<:LogMode}

    NumberOfPoints         = length(Position)
    PositionType           = eltype(Position)
    PositionUnderlyingType = eltype(PositionType)

    dρdtI      = zeros(PositionUnderlyingType, NumberOfPoints)
    Velocityₙ⁺ = zeros(PositionType, NumberOfPoints)
    Positionₙ⁺ = zeros(PositionType, NumberOfPoints)
    ρₙ⁺        = zeros(PositionUnderlyingType, NumberOfPoints)

    ∇Cᵢ  = zeros(PositionType, NumberOfPoints)
    ∇◌rᵢ = zeros(PositionUnderlyingType, NumberOfPoints)

    return dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, ∇Cᵢ, ∇◌rᵢ
end

function LoadBoundaryNormals(::Val{D}, ::Type{T}, path_mdbc) where {D, T}
    # Read the CSV file
    csv_file = CSV.File(path_mdbc)

    normals       = Vector{SVector{D,T}}()
    points        = Vector{SVector{D,T}}()
    ghost_points  = Vector{SVector{D,T}}()

    # Loop over each row of the file
    for row in csv_file
        # Extract the "Normal" fields into an SVector
        normal, point = LoadBoundaryNormalPoint(Val(D), T, row)

        push!(normals, normal)
        push!(points,  point)
        push!(ghost_points,  point+normal)

    end

    return points, ghost_points, normals
end

function LoadMDBCNormals!(::SimulationMetaData{D,T,S,K,NoMDBC,L}, SimParticles, path) where {D,T,S<:ShiftingMode, K<:KernelOutputMode, L<:LogMode}
    return nothing
end

function LoadMDBCNormals!(::SimulationMetaData{D,T,S,K,SimpleMDBC,L}, SimParticles, path) where {D,T,S<:ShiftingMode, K<:KernelOutputMode, L<:LogMode}
    if isnothing(path)
        return nothing
    end
    _, GhostPoints, GhostNormals = LoadBoundaryNormals(Val(D), T, path)
    for gi ∈ eachindex(GhostPoints)
        SimParticles.GhostPoints[gi]  = GhostPoints[gi]
        SimParticles.GhostNormals[gi] = GhostNormals[gi]
    end
    return nothing
end

function LoadMDBCNormals!(::SimulationMetaData{D,T,S,K,AdvancedMDBC,L}, SimParticles, path) where {D,T,S<:ShiftingMode, K<:KernelOutputMode, L<:LogMode}
    if isnothing(path)
        return nothing
    end
    _, GhostPoints, GhostNormals = LoadBoundaryNormals(Val(D), T, path)
    for gi ∈ eachindex(GhostPoints)
        SimParticles.GhostPoints[gi]  = GhostPoints[gi]
        SimParticles.GhostNormals[gi] = GhostNormals[gi]
    end
    return nothing
end

end
