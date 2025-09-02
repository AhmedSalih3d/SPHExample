module PreProcess

export LoadBoundaryNormals, AllocateDataStructures, AllocateSupportDataStructures, AllocateThreadedArrays

using CSV
using DataFrames
using StaticArrays
using StructArrays

using ..SimulationGeometry

function LoadSpecificCSV(::Val{D}, ::Type{T}, particle_type::ParticleType, particle_group_marker::Int, specific_csv::String) where {D, T}
    DF_SPECIFIC = CSV.read(specific_csv, DataFrame)

    nrows = nrow(DF_SPECIFIC)

    points       = Vector{SVector{D,T}}(undef, nrows)
    density      = Vector{T}(undef, nrows)
    types        = Vector{ParticleType}(undef, nrows)
    group_marker = Vector{Int}(undef, nrows)
    idp          = Vector{Int}(undef, nrows)

    i = 1
    for DF ∈ eachrow(DF_SPECIFIC)
        P1   = DF["Points:0"]
        P2   = DF["Points:1"]
        P3   = DF["Points:2"]
        Rhop = DF["Rhop"]
        Idp  = DF["Idp"] + 1

        points[i] = if D == 3
            SVector{3,T}(P1, P2, P3)
        else
            SVector{2,T}(P1, P3)
        end

        density[i]      = Rhop
        types[i]        = particle_type
        group_marker[i] = particle_group_marker
        idp[i]          = Idp
        i += 1
    end

    return points, density, types, group_marker, idp
end

function AllocateDataStructures(SimGeometry::Vector{<:Geometry{Dimensions, FloatType}}) where {Dimensions, FloatType}
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

    Acceleration    = zeros(PositionType, NumberOfPoints)
    Velocity        = zeros(PositionType, NumberOfPoints)
    Kernel          = zeros(PositionUnderlyingType, NumberOfPoints)
    KernelGradient  = zeros(PositionType, NumberOfPoints)
    GhostPoints     = zeros(PositionType, NumberOfPoints)
    GhostNormals    = zeros(PositionType, NumberOfPoints)

    Pressureᵢ      = zeros(PositionUnderlyingType, NumberOfPoints)
    
    Cells          = fill(zero(CartesianIndex{Dimensions}), NumberOfPoints)
    ChunkID        = zeros(Int, NumberOfPoints)

    SimParticles = StructArray((Cells = Cells, ChunkID = ChunkID, Kernel = Kernel, KernelGradient = KernelGradient, Position=Position, Acceleration=Acceleration, Velocity=Velocity, Density=Density, Pressure=Pressureᵢ, GravityFactor=GravityFactor, MotionLimiter=MotionLimiter, BoundaryBool = BoundaryBool, ID = Idp , Type = Types, GroupMarker = GroupMarker, GhostPoints = GhostPoints, GhostNormals=GhostNormals))

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

function AllocateThreadedArrays(
    SimMetaData,
    SimParticles,
    dρdtI,
    ∇Cᵢ,
    ∇◌rᵢ;
    n_copy = Base.Threads.nthreads(),
)
    n = length(dρdtI)
    dρdtI_threaded = similar(dρdtI, n * n_copy)
    Acceleration_threaded = similar(SimParticles.KernelGradient, n * n_copy)

    nt = (
        dρdtIThreaded = dρdtI_threaded,
        AccelerationThreaded = Acceleration_threaded,
    )

    if SimMetaData.FlagOutputKernelValues
        kernel_len = length(SimParticles.Kernel)
        Kernel_threaded = similar(SimParticles.Kernel, kernel_len * n_copy)
        KernelGradient_threaded =
            similar(SimParticles.KernelGradient, kernel_len * n_copy)
        nt = merge(
            nt,
            (
                KernelThreaded = Kernel_threaded,
                KernelGradientThreaded = KernelGradient_threaded,
            ),
        )
    end

    if SimMetaData.FlagShifting
        len_grad = length(∇Cᵢ)
        ∇Cᵢ_threaded = similar(∇Cᵢ, len_grad * n_copy)
        ∇◌rᵢ_threaded = similar(∇◌rᵢ, len_grad * n_copy)
        nt = merge(
            nt,
            (
                ∇CᵢThreaded = ∇Cᵢ_threaded,
                ∇◌rᵢThreaded = ∇◌rᵢ_threaded,
            ),
        )
    end

    SimThreadedArrays = StructArray(nt)

    return SimThreadedArrays
end

function LoadBoundaryNormals(::Val{D}, ::Type{T}, path_mdbc) where {D, T}
    # Read the CSV file into a DataFrame
    df = CSV.read(path_mdbc, DataFrame)

    normals       = Vector{SVector{D,T}}()
    points        = Vector{SVector{D,T}}()
    ghost_points  = Vector{SVector{D,T}}()

    # Loop over each row of the DataFrame
    
    for df_ in eachrow(df)
        # Extract the "Normal" fields into an SVector
        if D == 3
            normal = SVector{D,T}(df_["Normal:0"], df_["Normal:1"], df_["Normal:2"])
            point  = SVector{D,T}(df_["Points:0"], df_["Points:1"], df_["Points:2"])
        elseif D == 2
            normal = SVector{D,T}(df_["Normal:0"], df_["Normal:2"])
            point  = SVector{D,T}(df_["Points:0"], df_["Points:2"])
        end

        push!(normals, normal)
        push!(points,  point)
        push!(ghost_points,  point+normal)

    end

    return points, ghost_points, normals
end

end

