module PreProcess

export LoadBoundaryNormals, AllocateDataStructures, AllocateSupportDataStructures, AllocateThreadedArrays

using CSV
using DataFrames
using StaticArrays
using StructArrays

using ..SimulationGeometry

function LoadSpecificCSV(dims, float_type, particle_type, particle_group_marker, specific_csv)
    DF_SPECIFIC = CSV.read(specific_csv, DataFrame)

    points       = Vector{SVector{dims,float_type}}()
    density      = Vector{float_type}()
    types        = Vector{ParticleType}()
    group_marker = Vector{Int}()

    for DF ∈ eachrow(DF_SPECIFIC)
        P1   = DF["Points:0"]
        P2   = DF["Points:1"]
        P3   = DF["Points:2"]
        Rhop = DF["Rhop"]

        point = dims == 3 ? SVector{dims,float_type}(P1, P2, P3) : SVector{dims,float_type}(P1, P3)
        push!(points,  point)
        push!(density, Rhop)
        push!(types, particle_type)
        push!(group_marker, particle_group_marker)
    end

    return points, density, types, group_marker
end

function AllocateDataStructures(Dimensions,FloatType, SimGeometry)
    Position    = Vector{SVector{Dimensions, FloatType}}()
    Density     = Vector{FloatType}()
    Types       = Vector{ParticleType}()
    GroupMarker = Vector{UInt}()
    
    for geom in SimGeometry
        particle_type         = geom.Type
        particle_group_marker = geom.GroupMarker
        specific_csv          = geom.CSVFile
    
        # Assuming LoadSpecificCSV is already defined and works with these arguments
        points, density, types, group_marker = LoadSpecificCSV(Dimensions, FloatType, particle_type, particle_group_marker, specific_csv)
    
        # Concatenate the results to the respective arrays
        Position    = vcat(Position    , points)
        Density     = vcat(Density     , density)
        Types       = vcat(Types       , types)
        GroupMarker = vcat(GroupMarker , group_marker)
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



    Pressureᵢ      = zeros(PositionUnderlyingType, NumberOfPoints)
    
    Cells          = fill(zero(CartesianIndex{Dimensions}), NumberOfPoints)

    SimParticles = StructArray((Cells = Cells, Kernel = Kernel, KernelGradient = KernelGradient, Position=Position, Acceleration=Acceleration, Velocity=Velocity, Density=Density, Pressure=Pressureᵢ, GravityFactor=GravityFactor, MotionLimiter=MotionLimiter, BoundaryBool = BoundaryBool, ID = collect(1:NumberOfPoints) , Type = Types, GroupMarker = GroupMarker))

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

function LoadBoundaryNormals(dims, float_type, path_mdbc)
    # Read the CSV file into a DataFrame
    df = CSV.read(path_mdbc, DataFrame)

    normals       = Vector{SVector{dims,float_type}}()
    points        = Vector{SVector{dims,float_type}}()
    ghost_points  = Vector{SVector{dims,float_type}}()

    # Loop over each row of the DataFrame
    for i in eachindex(df)
        # Extract the "Normal" fields into an SVector
        if dims == 3
            normal = SVector{dims,float_type}(df[i, "Normal:0"], df[i, "Normal:1"], df[i, "Normal:2"])
            point  = SVector{dims,float_type}(df[i, "Points:0"], df[i, "Points:1"], df[i, "Points:2"])
        elseif dims == 2
            normal = SVector{dims,float_type}(df[i, "Normal:0"], df[i, "Normal:2"])
            point  = SVector{dims,float_type}(df[i, "Points:0"], df[i, "Points:2"])
        end

        push!(normals, normal)
        push!(points,  point)
        push!(ghost_points,  point+normal)

    end

    return points, ghost_points, normals
end

end

