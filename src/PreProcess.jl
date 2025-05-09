module PreProcess

export LoadBoundaryNormals, AllocateDataStructures, AllocateSupportDataStructures, AllocateThreadedArrays

using CSV
using DataFrames
using StaticArrays
using StructArrays

using ..SimulationGeometry

function LoadSpecificCSV(::Val{D}, ::Type{T}, particle_type::ParticleType, particle_group_marker::Int, specific_csv::String) where {D, T}
    DF_SPECIFIC = CSV.read(specific_csv, DataFrame)

    points       = Vector{SVector{D,T}}()
    density      = Vector{T}()
    types        = Vector{ParticleType}()
    group_marker = Vector{Int}()
    idp          = Vector{Int}()

    for DF ∈ eachrow(DF_SPECIFIC)
        P1   = DF["Points:0"]
        P2   = DF["Points:1"]
        P3   = DF["Points:2"]
        Rhop = DF["Rhop"]
        Idp  = DF["Idp"] + 1

        point = if D == 3
            SVector{3,T}(P1, P2, P3)
        else
            SVector{2,T}(P1, P3)
        end

        push!(points,  point)
        push!(density, Rhop)
        push!(types,   particle_type)
        push!(group_marker, particle_group_marker)
        push!(idp, Idp)
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
    
        # Assuming LoadSpecificCSV is already defined and works with these arguments
        points, density, types, group_marker, idp = LoadSpecificCSV(Val(Dimensions), FloatType, particle_type, particle_group_marker, specific_csv)
    
        # Concatenate the results to the respective arrays
        Position    = vcat(Position    , points)
        Density     = vcat(Density     , density)
        Types       = vcat(Types       , types)
        GroupMarker = vcat(GroupMarker , group_marker)
        Idp         = vcat(Idp         , idp)
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

function AllocateThreadedArrays(SimMetaData, SimParticles, dρdtI, ∇Cᵢ, ∇◌rᵢ   ; n_copy = Base.Threads.nthreads())
    
        
    dρdtIThreaded        = [copy(dρdtI) for _ in 1:n_copy]
    AccelerationThreaded = [copy(SimParticles.KernelGradient) for _ in 1:n_copy]

    nt = (
        dρdtIThreaded = dρdtIThreaded,
        AccelerationThreaded = AccelerationThreaded,
    )

    if SimMetaData.FlagOutputKernelValues
        KernelThreaded         = [copy(SimParticles.Kernel) for _ in 1:n_copy]
        KernelGradientThreaded = [copy(SimParticles.KernelGradient) for _ in 1:n_copy]
        nt = merge(nt, (
            KernelThreaded = KernelThreaded,
            KernelGradientThreaded = KernelGradientThreaded,
        ))
    end

    if SimMetaData.FlagShifting
        ∇CᵢThreaded  = [copy(∇Cᵢ) for _ in 1:n_copy]
        ∇◌rᵢThreaded = [copy(∇◌rᵢ) for _ in 1:n_copy]
        nt = merge(nt, (
            ∇CᵢThreaded  = ∇CᵢThreaded,
            ∇◌rᵢThreaded = ∇◌rᵢThreaded,
        ))
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

