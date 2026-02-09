using StaticArrays: SVector
using LinearAlgebra: norm
using SPHExample

const PARTICLE_HEADER = "\"Idp\",\"Vel:0\",\"Vel:1\",\"Vel:2\",\"Rhop\",\"Type\",\"Mk\",\"Points:0\",\"Points:1\",\"Points:2\""

@inline function UniformGrid(MinValue, MaxValue, Dx)
    Count = round(Int, (MaxValue - MinValue) / Dx)
    return [MinValue + Dx * i for i in 0:Count]
end

@inline function WriteParticleRow!(Io, Idp, X, Y, Z, Rhop)
    println(Io, join((Idp, 0.0, 0.0, 0.0, Rhop, 0, 0, X, Y, Z), ","))
    return Idp + 1
end

function CollectBoxSurfacePoints(BoxMin::SVector{3,T}, BoxMax::SVector{3,T}, Dx::T) where {T<:AbstractFloat}
    XValues = UniformGrid(BoxMin[1], BoxMax[1], Dx)
    YValues = UniformGrid(BoxMin[2], BoxMax[2], Dx)
    ZValues = UniformGrid(BoxMin[3], BoxMax[3], Dx)
    Tolerance = Dx * T(0.1)

    Points = SVector{3,T}[]
    for Z in ZValues
        for Y in YValues
            for X in XValues
                IsOnSurface = abs(X - BoxMin[1]) <= Tolerance || abs(X - BoxMax[1]) <= Tolerance ||
                              abs(Y - BoxMin[2]) <= Tolerance || abs(Y - BoxMax[2]) <= Tolerance ||
                              abs(Z - BoxMin[3]) <= Tolerance || abs(Z - BoxMax[3]) <= Tolerance
                if IsOnSurface
                    push!(Points, SVector{3,T}(X, Y, Z))
                end
            end
        end
    end
    return Points
end

function CollectBoxSolidPoints(BoxMin::SVector{3,T}, BoxSize::SVector{3,T}, Dx::T) where {T<:AbstractFloat}
    BoxMax = BoxMin + BoxSize
    XValues = UniformGrid(BoxMin[1], BoxMax[1], Dx)
    YValues = UniformGrid(BoxMin[2], BoxMax[2], Dx)
    ZValues = UniformGrid(BoxMin[3], BoxMax[3], Dx)

    Points = SVector{3,T}[]
    for Z in ZValues
        for Y in YValues
            for X in XValues
                push!(Points, SVector{3,T}(X, Y, Z))
            end
        end
    end
    return Points
end

function CollectSphereSurfacePoints(Centre::SVector{3,T}, Radius::T, Dx::T) where {T<:AbstractFloat}
    XValues = UniformGrid(Centre[1] - Radius, Centre[1] + Radius, Dx)
    YValues = UniformGrid(Centre[2] - Radius, Centre[2] + Radius, Dx)
    ZValues = UniformGrid(Centre[3] - Radius, Centre[3] + Radius, Dx)
    Tolerance = Dx * T(0.55)

    Points = SVector{3,T}[]
    for Z in ZValues
        for Y in YValues
            for X in XValues
                Point = SVector{3,T}(X, Y, Z)
                DistanceToCentre = norm(Point - Centre)
                if abs(DistanceToCentre - Radius) <= Tolerance
                    push!(Points, Point)
                end
            end
        end
    end
    return Points
end

function WritePointsCSV(FilePath::String, Points::AbstractVector{SVector{3,T}}, StartIdp::Int, Rho0::T) where {T<:AbstractFloat}
    CurrentIdp = StartIdp
    open(FilePath, "w") do Io
        println(Io, PARTICLE_HEADER)
        for Point in Points
            CurrentIdp = WriteParticleRow!(Io, CurrentIdp, Point[1], Point[2], Point[3], Rho0)
        end
    end
    return CurrentIdp
end

function GenerateCaseForcesGeometryCSVs(
    Dx,
    InputFolder;
    ForceRegenerate::Bool=false,
    Rho0=1000.0,
)
    BaseName = "CaseForces3D_Dp$(Dx)"
    ContainerCSV = joinpath(InputFolder, "$(BaseName)_ContainerBound.csv")
    SphereCSV = joinpath(InputFolder, "$(BaseName)_MiddleSphereBound.csv")
    FluidLeftCSV = joinpath(InputFolder, "$(BaseName)_FluidLeft.csv")
    FluidRightCSV = joinpath(InputFolder, "$(BaseName)_FluidRight.csv")

    if !ForceRegenerate && isfile(ContainerCSV) && isfile(SphereCSV) && isfile(FluidLeftCSV) && isfile(FluidRightCSV)
        return ContainerCSV, SphereCSV, FluidLeftCSV, FluidRightCSV
    end

    mkpath(InputFolder)

    ContainerMin = SVector{3,Float64}(-0.5, -0.5, -0.5)
    ContainerMax = SVector{3,Float64}(0.5, 0.5, 0.5)
    SphereCentre = SVector{3,Float64}(0.0, 0.0, 0.0)
    SphereRadius = 0.1
    FluidSize = SVector{3,Float64}(0.3213, 0.964, 0.3213)
    FluidLeftMin = SVector{3,Float64}(-0.482, -0.482, -0.482)
    FluidRightMin = SVector{3,Float64}(0.1607, -0.482, -0.482)

    ContainerPoints = CollectBoxSurfacePoints(ContainerMin, ContainerMax, Dx)
    SpherePoints = CollectSphereSurfacePoints(SphereCentre, SphereRadius, Dx)
    FluidLeftPoints = CollectBoxSolidPoints(FluidLeftMin, FluidSize, Dx)
    FluidRightPoints = CollectBoxSolidPoints(FluidRightMin, FluidSize, Dx)

    NextIdp = 0
    NextIdp = WritePointsCSV(ContainerCSV, ContainerPoints, NextIdp, Rho0)
    NextIdp = WritePointsCSV(SphereCSV, SpherePoints, NextIdp, Rho0)
    NextIdp = WritePointsCSV(FluidLeftCSV, FluidLeftPoints, NextIdp, Rho0)
    NextIdp = WritePointsCSV(FluidRightCSV, FluidRightPoints, NextIdp, Rho0)
    @assert NextIdp > 0 "No particles were generated for CaseForces geometry."

    return ContainerCSV, SphereCSV, FluidLeftCSV, FluidRightCSV
end

@inline function CaseForcesLinearAccZ(Time)
    if Time <= 1
        return 15.0 * Time
    elseif Time <= 3
        return 15.0 * (3.0 - Time) / 2.0
    else
        return 0.0
    end
end

@inline function CaseForcesAngularAccY(Time)
    if Time < 3
        return 0.0
    else
        return 20.0 * sin(pi * (Time - 3.0))
    end
end

function WriteFallbackCaseForcesAccelerationCSV(FilePath::String, SimulationTime; SampleDt=0.1)
    open(FilePath, "w") do Io
        println(Io, "#Time;LinearAccX;LinearAccY;LinearAccZ;AngularAccX;AngularAccY;AngularAccZ")
        for Time in 0.0:SampleDt:SimulationTime
            TimeValue = round(Time; digits=8)
            LinearAccZ = CaseForcesLinearAccZ(TimeValue)
            AngularAccY = CaseForcesAngularAccY(TimeValue)
            println(Io, "$(TimeValue);0.0;0.0;$(LinearAccZ);0.0;$(AngularAccY);0.0")
        end
    end
    return nothing
end

function EnsureCaseForcesAccelerationFiles(
    InputFolder::String;
    ForceRegenerate::Bool=false,
    SimulationTime=5.0,
    DualSPHysicsCaseFolder::String="E:/DualSPHysics_v5.4/examples/main/04_ExternalForces",
)
    ForcingFolder = joinpath(InputFolder, "forcing")
    AccFile0 = joinpath(ForcingFolder, "CaseForcesData_0.csv")
    AccFile1 = joinpath(ForcingFolder, "CaseForcesData_1.csv")

    if !ForceRegenerate && isfile(AccFile0) && isfile(AccFile1)
        return AccFile0, AccFile1
    end

    mkpath(ForcingFolder)

    SourceFile0 = joinpath(DualSPHysicsCaseFolder, "CaseForcesData_0.csv")
    SourceFile1 = joinpath(DualSPHysicsCaseFolder, "CaseForcesData_1.csv")
    if isfile(SourceFile0) && isfile(SourceFile1)
        cp(SourceFile0, AccFile0; force=true)
        cp(SourceFile1, AccFile1; force=true)
        return AccFile0, AccFile1
    end

    WriteFallbackCaseForcesAccelerationCSV(AccFile0, SimulationTime)
    cp(AccFile0, AccFile1; force=true)
    return AccFile0, AccFile1
end

let
    Dimensions = 3
    FloatType = Float64

    Dx = 0.02
    Rho0 = 1000.0
    SoundSpeed = 70.8712
    ArtificialAlpha = 0.1

    SimulationName = "CaseForces3DAcc"
    SaveLocation = "./results/$(SimulationName)"
    InputFolder = "./input/caseforces_3d"

    SimulationTime = 5.0
    OutputTime = 0.02

    ContainerCSV, SphereCSV, FluidLeftCSV, FluidRightCSV = GenerateCaseForcesGeometryCSVs(
        Dx,
        InputFolder;
        ForceRegenerate=false,
        Rho0=Rho0,
    )
    AccelerationFile0, AccelerationFile1 = EnsureCaseForcesAccelerationFiles(
        InputFolder;
        ForceRegenerate=false,
        SimulationTime=SimulationTime,
    )

    SimConstants = SimulationConstants{FloatType}(
        dx = Dx,
        ρ₀ = Rho0,
        m₀ = Rho0 * Dx^3,
        c₀ = SoundSpeed,
        g = 9.81,
        δᵩ = 0.1,
        CFL = 0.25,
    )

    SimMetaData = SimulationMetaData{Dimensions,FloatType,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(
        SimulationName = SimulationName,
        SaveLocation = SaveLocation,
        SimulationTime = SimulationTime,
        OutputTimes = OutputTime,
        VisualizeInParaview = true,
        ExportSingleVTKHDF = true,
        ExportGridCells = true,
        OpenLogFile = true,
    )

    if !isdir(SimMetaData.SaveLocation)
        mkpath(SimMetaData.SaveLocation)
    end

    ContainerBoundary = Geometry{Dimensions, FloatType}(
        CSVFile = ContainerCSV,
        GroupMarker = 1,
        Type = Fixed,
        Motion = nothing,
    )

    MiddleSphereBoundary = Geometry{Dimensions, FloatType}(
        CSVFile = SphereCSV,
        GroupMarker = 2,
        Type = Fixed,
        Motion = nothing,
    )

    FluidLeft = Geometry{Dimensions, FloatType}(
        CSVFile = FluidLeftCSV,
        GroupMarker = 3,
        Type = Fluid,
        Motion = nothing,
    )

    FluidRight = Geometry{Dimensions, FloatType}(
        CSVFile = FluidRightCSV,
        GroupMarker = 4,
        Type = Fluid,
        Motion = nothing,
    )

    SimulationGeometry = [ContainerBoundary, MiddleSphereBoundary, FluidLeft, FluidRight]
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaData)
    SimLogger = SimulationLogger(SimMetaData.SaveLocation)
    SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); h = 0.8 * sqrt(3 * Dx^2))

    GroupModelCount = maximum(GeometryDef.GroupMarker for GeometryDef in SimulationGeometry)
    FluidAccelerationFilesByGroup = Dict(
        3 => AccelerationFile0,
        4 => AccelerationFile1,
    )
    AccelerationCentreByGroup = Dict(
        3 => SVector{3,FloatType}(0.0, 0.0, 0.0),
        4 => SVector{3,FloatType}(0.0, 0.0, 0.0),
    )
    GlobalGravityEnabledByGroup = Dict(
        3 => true,
        4 => true,
    )

    FluidAccelerationModel = LoadFluidAccelerationInputByGroupCSV(
        FluidAccelerationFilesByGroup,
        GroupModelCount,
        Val(Dimensions),
        FloatType;
        AccelerationCentreByGroup = AccelerationCentreByGroup,
        GlobalGravityEnabledByGroup = GlobalGravityEnabledByGroup,
    )

    CleanUpSimulationFolder(SimMetaData.SaveLocation)

    RunSimulation(
        SimGeometry = SimulationGeometry,
        SimMetaData = SimMetaData,
        SimConstants = SimConstants,
        SimLogger = SimLogger,
        SimParticles = SimParticles,
        SimKernel = SimKernel,
        SimViscosity = ArtificialViscosity(α = ArtificialAlpha),
        SimDensityDiffusion = LinearDensityDiffusion(),
        SimTimeStepping = SingleNeighborTimeStepping(),
        FluidAccelerationModel = FluidAccelerationModel,
    )
end
