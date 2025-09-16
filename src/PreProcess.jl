module PreProcess

export LoadBoundaryNormals, AllocateDataStructures, AllocateSupportDataStructures, AllocateThreadedArrays

using CSV
using StaticArrays
using StructArrays

using ..SimulationGeometry
using ..SimulationMetaDataConfiguration

function LoadSpecificCSV(::Val{D}, ::Type{T}, particle_type::ParticleType, particle_group_marker::Int, specific_csv::String) where {D, T}
    csv_file = CSV.File(specific_csv)

    nrows = length(csv_file)

    points       = Vector{SVector{D,T}}(undef, nrows)
    density      = Vector{T}(undef, nrows)
    types        = Vector{ParticleType}(undef, nrows)
    group_marker = Vector{Int}(undef, nrows)
    idp          = Vector{Int}(undef, nrows)

    for (i, row) ∈ enumerate(csv_file)
        P1   = getproperty(row, Symbol("Points:0"))
        P2   = getproperty(row, Symbol("Points:1"))
        P3   = getproperty(row, Symbol("Points:2"))
        Rhop = row.Rhop
        Idp  = row.Idp + 1

        points[i] = if D == 3
            SVector{3,T}(P1, P2, P3)
        else
            SVector{2,T}(P1, P3)
        end

        density[i]      = Rhop
        types[i]        = particle_type
        group_marker[i] = particle_group_marker
        idp[i]          = Idp
    end

    return points, density, types, group_marker, idp
end

"""
    AllocateDataStructures(SimGeometry, SimMetaData)

Load particle data from `SimGeometry` and return a struct array of particles.
Kernel arrays are included or omitted based on the `KernelOutputMode` of
`SimMetaData`. A convenience method without `SimMetaData` defaults to
`NoKernelOutput`.
"""
function AllocateDataStructures(
    SimGeometry::Vector{<:Geometry{Dimensions, FloatType}},
    ::Type{KMode},
    ::Type{BMode},
) where {Dimensions, FloatType, KMode<:KernelOutputMode, BMode<:MDBCMode}
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

    GravityFactor = similar(Density)
    for i ∈ eachindex(GravityFactor)
        fac = 0
        if     Types[i] == Fluid
            fac = -1
        elseif Types[i] == Moving
            fac =  1
        end
        GravityFactor[i] = fac
    end

    MotionLimiter = similar(Density)
    for i ∈ eachindex(MotionLimiter)
        fac = 0
        if   Types[i] == Fluid
            fac =  1
        else Types[i] == Moving
            fac =  0
        end
        MotionLimiter[i] = fac
    end

    BoundaryBool  = UInt8.(.!Bool.(MotionLimiter))

    Acceleration   = zeros(PositionType, NumberOfPoints)
    Velocity       = zeros(PositionType, NumberOfPoints)
    Pressureᵢ      = zeros(PositionUnderlyingType, NumberOfPoints)
    Cells   = fill(zero(CartesianIndex{Dimensions}), NumberOfPoints)
    ChunkID = zeros(Int, NumberOfPoints)

    base_nt = (
        Cells         = Cells,
        ChunkID       = ChunkID,
        Position      = Position,
        Acceleration  = Acceleration,
        Velocity      = Velocity,
        Density       = Density,
        Pressure      = Pressureᵢ,
        GravityFactor = GravityFactor,
        MotionLimiter = MotionLimiter,
        BoundaryBool  = BoundaryBool,
        ID            = Idp,
        Type          = Types,
        GroupMarker   = GroupMarker,
    )

    kernel_nt = kernel_particle_fields(KMode, NumberOfPoints,
                                       PositionType, PositionUnderlyingType)

    mdbc_nt = mdbc_particle_fields(BMode, NumberOfPoints, PositionType)

    SimParticles = StructArray(merge(base_nt, kernel_nt, mdbc_nt))

    sort!(SimParticles, by = p -> p.ID)

    return SimParticles
end

function AllocateDataStructures(
    SimGeometry::Vector{<:Geometry{Dimensions, FloatType}},
    ::Type{KMode},
) where {Dimensions, FloatType, KMode<:KernelOutputMode}
    AllocateDataStructures(SimGeometry, KMode, NoMDBC)
end

AllocateDataStructures(SimGeometry::Vector{<:Geometry{Dimensions, FloatType}}) where {Dimensions, FloatType} =
    AllocateDataStructures(SimGeometry, NoKernelOutput, NoMDBC)

function AllocateDataStructures(
    SimGeometry::Vector{<:Geometry{Dimensions, FloatType}},
    SimMetaData::SimulationMetaData{Dimensions, FloatType, SMode, KMode, BMode, LMode},
) where {Dimensions, FloatType, SMode<:ShiftingMode, KMode<:KernelOutputMode,
         BMode<:MDBCMode, LMode<:LogMode}
    AllocateDataStructures(SimGeometry, KMode, BMode)
end

function kernel_particle_fields(::Type{NoKernelOutput}, n, _, _)
    NamedTuple()
end

function kernel_particle_fields(::Type{StoreKernelOutput}, n, position_type, underlying_type)
    (
        Kernel         = zeros(underlying_type, n),
        KernelGradient = zeros(position_type, n),
    )
end

function mdbc_particle_fields(::Type{NoMDBC}, _, _)
    NamedTuple()
end

function mdbc_particle_fields(::Type{BMode}, n, position_type) where {BMode<:MDBCMode}
    (
        GhostPoints  = zeros(position_type, n),
        GhostNormals = zeros(position_type, n),
    )
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

function allocate_kernel_arrays(::SimulationMetaData{D,T,S,NoKernelOutput,B,L},
                                SimParticles, n_copy) where {D,T,S<:ShiftingMode,
                                                             B<:MDBCMode,
                                                             L<:LogMode}
    return NamedTuple()
end
function allocate_kernel_arrays(::SimulationMetaData{D,T,S,K,B,L},
                                SimParticles, n_copy) where {D,T,S<:ShiftingMode,
                                                             K<:KernelOutputMode,
                                                             B<:MDBCMode,
                                                             L<:LogMode}
    KernelThreaded         = [copy(SimParticles.Kernel) for _ in 1:n_copy]
    KernelGradientThreaded = [copy(SimParticles.KernelGradient) for _ in 1:n_copy]
    return (
        KernelThreaded = KernelThreaded,
        KernelGradientThreaded = KernelGradientThreaded,
    )
end

function allocate_shifting_arrays(::SimulationMetaData{D,T,NoShifting,K,B,L},
                                  ∇Cᵢ, ∇◌rᵢ, n_copy) where {D,T,K<:KernelOutputMode,
                                                            B<:MDBCMode,
                                                            L<:LogMode}
    return NamedTuple()
end
function allocate_shifting_arrays(::SimulationMetaData{D,T,S,K,B,L},
                                  ∇Cᵢ, ∇◌rᵢ, n_copy) where {D,T,S<:ShiftingMode,
                                                            K<:KernelOutputMode,
                                                            B<:MDBCMode,
                                                            L<:LogMode}
    ∇CᵢThreaded  = [copy(∇Cᵢ) for _ in 1:n_copy]
    ∇◌rᵢThreaded = [copy(∇◌rᵢ) for _ in 1:n_copy]
    return (
        ∇CᵢThreaded  = ∇CᵢThreaded,
        ∇◌rᵢThreaded = ∇◌rᵢThreaded,
    )
end

function AllocateThreadedArrays(SimMetaData::SimulationMetaData{D,T,S,K,B,L},
                                SimParticles, dρdtI, ∇Cᵢ, ∇◌rᵢ;
                                n_copy = Base.Threads.nthreads()) where {D,T,S<:ShiftingMode,
                                                                           K<:KernelOutputMode,
                                                                           B<:MDBCMode,
                                                                           L<:LogMode}
    dρdtIThreaded        = [copy(dρdtI) for _ in 1:n_copy]
    AccelerationThreaded = [copy(SimParticles.Acceleration) for _ in 1:n_copy]
    nt = (
        dρdtIThreaded = dρdtIThreaded,
        AccelerationThreaded = AccelerationThreaded,
    )

    nt = merge(nt, allocate_kernel_arrays(SimMetaData, SimParticles, n_copy))
    nt = merge(nt, allocate_shifting_arrays(SimMetaData, ∇Cᵢ, ∇◌rᵢ, n_copy))

    return StructArray(nt)
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
        if D == 3
            normal = SVector{D,T}(getproperty(row, Symbol("Normal:0")), getproperty(row, Symbol("Normal:1")), getproperty(row, Symbol("Normal:2")))
            point  = SVector{D,T}(getproperty(row, Symbol("Points:0")), getproperty(row, Symbol("Points:1")), getproperty(row, Symbol("Points:2")))
        elseif D == 2
            normal = SVector{D,T}(getproperty(row, Symbol("Normal:0")), getproperty(row, Symbol("Normal:2")))
            point  = SVector{D,T}(getproperty(row, Symbol("Points:0")), getproperty(row, Symbol("Points:2")))
        end

        push!(normals, normal)
        push!(points,  point)
        push!(ghost_points,  point+normal)

    end

    return points, ghost_points, normals
end

end
