module PreProcess

export LoadParticlesFromCSV_StaticArrays, LoadBoundaryNormals, AllocateDataStructures

using CSV
using DataFrames
using StaticArrays
using StructArrays

function LoadParticlesFromCSV_StaticArrays(dims, float_type, fluid_csv, boundary_csv)
    DF_FLUID = CSV.read(fluid_csv, DataFrame)
    DF_BOUND = CSV.read(boundary_csv, DataFrame)

    P1F = DF_FLUID[!, "Points:0"]
    P2F = DF_FLUID[!, "Points:1"]
    P3F = DF_FLUID[!, "Points:2"] 
    P1B = DF_BOUND[!, "Points:0"]
    P2B = DF_BOUND[!, "Points:1"]
    P3B = DF_BOUND[!, "Points:2"]

    points = Vector{SVector{dims,float_type}}()
    density_fluid = Vector{float_type}()
    density_bound = Vector{float_type}()

    for (P1, P2, P3, DF, density) in [(P1B, P2B, P3B, DF_BOUND, density_bound), (P1F, P2F, P3F, DF_FLUID, density_fluid)]
        for i = 1:length(P1)
            point = dims == 3 ? SVector{dims,float_type}(P1[i], P2[i], P3[i]) : SVector{dims,float_type}(P1[i], P3[i])
            push!(points, point)
            push!(density, DF.Rhop[i])
        end
    end

    return points, density_fluid, density_bound
end

function AllocateDataStructures(Dimensions,FloatType, FluidCSV,BoundCSV)
    @inline Position, density_fluid, density_bound  = LoadParticlesFromCSV_StaticArrays(Dimensions,FloatType, FluidCSV,BoundCSV)

    NumberOfPoints           = length(Position)
    PositionType             = eltype(Position)
    PositionUnderlyingType   = eltype(PositionType)

    Density        = deepcopy([density_bound; density_fluid])

    GravityFactor = [ zeros(size(density_bound,1)) ; -ones(size(density_fluid,1)) ]
    
    MotionLimiter = [ zeros(size(density_bound,1)) ;  ones(size(density_fluid,1)) ]

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

    SimParticles = StructArray((Cells = Cells, Position=Position, Acceleration=Acceleration, Velocity=Velocity, Density=Density, Pressure=Pressureᵢ, GravityFactor=GravityFactor, MotionLimiter=MotionLimiter, BoundaryBool = BoundaryBool, ID = collect(1:NumberOfPoints)))

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

