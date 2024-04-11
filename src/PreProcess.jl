module PreProcess

export LoadParticlesFromCSV_StaticArrays, LoadBoundaryNormals, AllocateDataStructures

using CSV
using DataFrames
using StaticArrays
using StructArrays

function LoadSpecificCSV(dims, float_type, particle_type, specific_csv)
    DF_SPECIFIC = CSV.read(specific_csv, DataFrame)

    points      = Vector{SVector{dims,float_type}}()
    density     = Vector{float_type}()
    types       = Vector{Int}()

    for DF ∈ eachrow(DF_SPECIFIC)
        P1   = DF["Points:0"]
        P2   = DF["Points:1"]
        P3   = DF["Points:2"]
        Rhop = DF["Rhop"]

        point = dims == 3 ? SVector{dims,float_type}(P1, P2, P3) : SVector{dims,float_type}(P1, P3)
        push!(points,  point)
        push!(density, Rhop)
        push!(types, particle_type)
    end

    return points, density, types
end

function LoadParticlesFromCSV_StaticArrays(dims, float_type, fluid_csv, boundary_csv)

    FluidParticlesPoints,          FluidParticlesDensity         , FluidParticlesTypes          = LoadSpecificCSV(dims, float_type, 4, fluid_csv)
    FixedBoundaryParticlesPoints,  FixedBoundaryParticlesDensity , FixedBoundaryParticlesTypes  = LoadSpecificCSV(dims, float_type, 0, boundary_csv)
    MovingBoundaryParticlesPoints, MovingBoundaryParticlesDensity, MovingBoundaryParticlesTypes = LoadSpecificCSV(dims, float_type, 1, boundary_csv)

    points  = [FluidParticlesPoints;  FixedBoundaryParticlesPoints]
    density = [FluidParticlesDensity; FixedBoundaryParticlesDensity]
    types   = [FluidParticlesTypes;   FixedBoundaryParticlesTypes]

    return points, density, types
end

function AllocateDataStructures(Dimensions,FloatType, FluidCSV,BoundCSV)
    @inline Position, Density, Types  = LoadParticlesFromCSV_StaticArrays(Dimensions,FloatType, FluidCSV,BoundCSV)

    NumberOfPoints           = length(Position)
    PositionType             = eltype(Position)
    PositionUnderlyingType   = eltype(PositionType)

    GravityFactor = similar(Density)
    for i ∈ eachindex(GravityFactor)
        fac = 0
        if     Types[i] == 4
            fac = -1
        elseif Types[i] == 1
            fac =  1
        end
        GravityFactor[i] = fac
    end

    # GravityFactor = [ zeros(size(density_bound,1)) ; -ones(size(density_fluid,1)) ]
    
    MotionLimiter = similar(Density)
    for i ∈ eachindex(MotionLimiter)
        fac = 0
        if   Types[i] == 4
            fac =  1
        else Types[i] == 1
            fac =  0
        end
        MotionLimiter[i] = fac
    end

    # MotionLimiter = [ zeros(size(density_bound,1)) ;  ones(size(density_fluid,1)) ]

    BoundaryBool  = .!Bool.(MotionLimiter)

    Acceleration    = zeros(PositionType, NumberOfPoints)
    Velocity        = zeros(PositionType, NumberOfPoints)
    Kernel          = zeros(PositionUnderlyingType, NumberOfPoints)
    KernelGradient  = zeros(PositionType, NumberOfPoints)

    dρdtI           = zeros(PositionUnderlyingType, NumberOfPoints)

    Velocityₙ⁺      = zeros(PositionType, NumberOfPoints)
    Positionₙ⁺      = zeros(PositionType, NumberOfPoints)
    ρₙ⁺             = zeros(PositionUnderlyingType, NumberOfPoints)

    Pressureᵢ      = zeros(PositionUnderlyingType, NumberOfPoints)
    
    Cells          = fill(zero(CartesianIndex{Dimensions}), NumberOfPoints)

    if Types == nothing
        SimParticles = StructArray((Cells = Cells, Position=Position, Acceleration=Acceleration, Velocity=Velocity, Density=Density, Pressure=Pressureᵢ, GravityFactor=GravityFactor, MotionLimiter=MotionLimiter, BoundaryBool = BoundaryBool, ID = collect(1:NumberOfPoints)))
    else
        SimParticles = StructArray((Cells = Cells, Position=Position, Acceleration=Acceleration, Velocity=Velocity, Density=Density, Pressure=Pressureᵢ, GravityFactor=GravityFactor, MotionLimiter=MotionLimiter, BoundaryBool = BoundaryBool, ID = collect(1:NumberOfPoints) , Type = Types))
    end

    return SimParticles, dρdtI, Velocityₙ⁺, Positionₙ⁺, ρₙ⁺, Kernel, KernelGradient
end

function LoadBoundaryNormals(dims, float_type, path_mdbc)
    # Read the CSV file into a DataFrame
    df = CSV.read(path_mdbc, DataFrame)

    normals       = Vector{SVector{dims,float_type}}()
    points        = Vector{SVector{dims,float_type}}()
    ghost_points  = Vector{SVector{dims,float_type}}()

    # Loop over each row of the DataFrame
    for i in 1:size(df, 1)
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

