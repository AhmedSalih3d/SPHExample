using StaticArrays: SVector
using SPHExample

const PARTICLE_HEADER = "\"Idp\",\"Vel:0\",\"Vel:1\",\"Vel:2\",\"Rhop\",\"Type\",\"Mk\",\"Points:0\",\"Points:1\",\"Points:2\""
const GHOST_HEADER = "\"Idp\",\"Mk\",\"Normal:0\",\"Normal:1\",\"Normal:2\",\"NormalSize\",\"Points:0\",\"Points:1\",\"Points:2\""

@inline function UniformGrid(MinValue, MaxValue, Dx)
    Count = round(Int, (MaxValue - MinValue) / Dx)
    return [MinValue + Dx * i for i in 0:Count]
end

@inline function WriteParticleRow!(Io, Idp, TypeValue, Mk, X, Z, Rho)
    println(Io, join((Idp, 0.0, 0.0, 0.0, Rho, TypeValue, Mk, X, 0.0, Z), ","))
    return nothing
end

@inline function WriteGhostRow!(Io, Idp, Mk, NormalX, NormalZ, X, Z)
    NormalSize = hypot(NormalX, NormalZ)
    println(Io, join((Idp, Mk, NormalX, 0.0, NormalZ, NormalSize, X, 0.0, Z), ","))
    return nothing
end

@inline function WallNormalTowardFluid(X, Z, TankMinX, TankMaxX, TankMinZ, TankMaxZ, Dx, GhostInsetFactor)
    # Place ghost target one spacing inside the physical tank.
    GhostInset = GhostInsetFactor * Dx
    TargetX = clamp(X, TankMinX + GhostInset, TankMaxX - GhostInset)
    TargetZ = clamp(Z, TankMinZ + GhostInset, TankMaxZ - GhostInset)
    return TargetX - X, TargetZ - Z
end

function GenerateSloshingTankCSVs(
    Dx,
    InputFolder;
    ForceRegenerate=false,
    BoundaryLayers=3,
    TankMinX=-0.45,
    TankMaxX=0.45,
    TankMinZ=0.0,
    TankMaxZ=0.508,
    WaterLevel=0.093,
    Rho0=1000.0,
    Gravity=9.81,
    SoundSpeed=28.198718,
    GhostInsetFactor=1.0,
)
    @assert BoundaryLayers >= 1 "BoundaryLayers must be at least 1."
    @assert GhostInsetFactor > 0 "GhostInsetFactor must be positive."

    BaseName = "SloshingTank_Dp$(Dx)_L$(BoundaryLayers)"
    BoundFile = joinpath(InputFolder, "$(BaseName)_Bound.csv")
    FluidFile = joinpath(InputFolder, "$(BaseName)_Fluid.csv")
    GhostFile = joinpath(InputFolder, "$(BaseName)_GhostNodes.csv")

    if !ForceRegenerate && isfile(BoundFile) && isfile(FluidFile) && isfile(GhostFile)
        return BoundFile, FluidFile, GhostFile
    end

    mkpath(InputFolder)

    OuterOffset = (BoundaryLayers - 1) * Dx
    XValues = UniformGrid(TankMinX - OuterOffset, TankMaxX + OuterOffset, Dx)
    ZValues = UniformGrid(TankMinZ - OuterOffset, TankMaxZ + OuterOffset, Dx)

    BoundaryPoints = SVector{2,Float64}[]
    for Z in ZValues
        for X in XValues
            IsInterior = (X > TankMinX) && (X < TankMaxX) && (Z > TankMinZ) && (Z < TankMaxZ)
            if !IsInterior
                push!(BoundaryPoints, SVector{2,Float64}(X, Z))
            end
        end
    end

    open(BoundFile, "w") do Io
        println(Io, PARTICLE_HEADER)
        for (i, Point) in enumerate(BoundaryPoints)
            WriteParticleRow!(Io, i - 1, 0, 10, Point[1], Point[2], Rho0)
        end
    end

    open(GhostFile, "w") do Io
        println(Io, GHOST_HEADER)
        for (i, Point) in enumerate(BoundaryPoints)
            X = Point[1]
            Z = Point[2]
            Nx, Nz = WallNormalTowardFluid(X, Z, TankMinX, TankMaxX, TankMinZ, TankMaxZ, Dx, GhostInsetFactor)
            WriteGhostRow!(Io, i - 1, 10, Nx, Nz, X, Z)
        end
    end

    FluidTop = floor((WaterLevel - TankMinZ) / Dx) * Dx
    FluidXValues = collect(range(TankMinX + Dx, TankMaxX - Dx; step=Dx))
    FluidZValues = collect(range(TankMinZ + Dx, FluidTop; step=Dx))

    CbInv = inv((SoundSpeed^2 * Rho0) / 7.0)
    FluidId = length(BoundaryPoints)

    open(FluidFile, "w") do Io
        println(Io, PARTICLE_HEADER)
        for X in FluidXValues
            for Z in FluidZValues
                HydrostaticPressure = Rho0 * Gravity * max(WaterLevel - Z, 0.0)
                Rhop = Rho0 + InverseHydrostaticEquationOfState(Rho0, HydrostaticPressure, CbInv)
                WriteParticleRow!(Io, FluidId, 3, 1, X, Z, Rhop)
                FluidId += 1
            end
        end
    end

    return BoundFile, FluidFile, GhostFile
end

function EnsureSloshingMotionFile(
    InputFolder::String,
)
    MotionFile = joinpath(InputFolder, "CaseSloshingMotionData.dat")

    if !isfile(MotionFile)
        error("Missing local sloshing motion file: $(MotionFile).")
    end

    return MotionFile
end

function LoadSloshingMotionAngles(MotionFile::String, ::Type{T}) where {T<:AbstractFloat}
    Times = T[]
    Angles = T[]

    for line in eachline(MotionFile)
        text = strip(line)
        if isempty(text) || startswith(text, "#")
            continue
        end

        fields = split(text)
        if length(fields) < 2
            continue
        end

        push!(Times, parse(T, fields[1]))
        push!(Angles, deg2rad(parse(T, fields[2])))
    end

    @assert !isempty(Times) "No motion samples were loaded from $(MotionFile)."
    return Times, Angles
end

let
    Dimensions = 2
    FloatType = Float64

    Dx = 0.002
    # In this codebase, h = k*dx and support radius H = k*h = k^2*dx.
    # Keep k modest for stability in this case.
    KernelScale = 2.0
    # Full wall support requires (BoundaryLayers - 1) * dx >= H.
    BoundaryLayers = max(3, ceil(Int, KernelScale^2) + 1)
    SoundSpeed = 28.198718
    ArtificialAlpha = 0.05
    GhostInsetFactor = 1.0

    InputFolder = "./input/sloshing_tank_2d_layers"

    BoundCSV, FluidCSV, GhostCSV = GenerateSloshingTankCSVs(
        Dx,
        InputFolder;
        ForceRegenerate = true,
        BoundaryLayers = BoundaryLayers,
        SoundSpeed = SoundSpeed,
        GhostInsetFactor = GhostInsetFactor,
    )

    SimulationName = "SloshingTank2DRotationLayers$(BoundaryLayers)"

    SimConstants = SimulationConstants{FloatType}(
        dx = Dx,
        ρ₀ = 1000.0,
        c₀ = SoundSpeed,
        g = 9.81,
        δᵩ = 0.1,
        CFL = 0.20,
    )

    SimMetaData = SimulationMetaData{Dimensions,FloatType,NoShifting,NoKernelOutput,SimpleMDBC,StoreLog}(
        SimulationName = SimulationName,
        SaveLocation = "W:/Simulations/$(SimulationName)",
        SimulationTime = 8.35,
        OutputTimes = 0.01,
        VisualizeInParaview = true,
        ExportSingleVTKHDF = true,
        ExportGridCells = true,
        OpenLogFile = true,
    )

    if !isdir(SimMetaData.SaveLocation)
        mkpath(SimMetaData.SaveLocation)
    end

    MotionFile = EnsureSloshingMotionFile(InputFolder)
    MotionTimes, MotionAngles = LoadSloshingMotionAngles(MotionFile, FloatType)

    TankBoundary = Geometry{Dimensions, FloatType}(
        CSVFile = BoundCSV,
        GroupMarker = 1,
        Type = Moving,
        Motion = nothing,
    )

    Water = Geometry{Dimensions, FloatType}(
        CSVFile = FluidCSV,
        GroupMarker = 2,
        Type = Fluid,
        Motion = nothing,
    )

    SimulationGeometry = [TankBoundary, Water]
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaData)
    SimLogger = SimulationLogger(SimMetaData.SaveLocation)
    SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); h = 1.3 * Dx, k = KernelScale)

    BoundaryIndices = findall(i -> SimParticles.Type[i] == Moving, eachindex(SimParticles.Type))
    BoundaryKeys = [(Int(SimParticles.GroupMarker[i]), SimParticles.ID[i]) for i in BoundaryIndices]
    InitialBoundaryPositions = copy(SimParticles.Position[BoundaryIndices])
    Pivot = SVector{2,FloatType}(0.0, 0.0)

    RigidMotionModel = RigidRotationMotionSeries(
        MotionTimes,
        MotionAngles,
        BoundaryKeys,
        InitialBoundaryPositions,
        Pivot,
        FollowGhostNormals = true,
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
        ParticleNormalsPath = GhostCSV,
        RigidMotionModel = RigidMotionModel,
    )
end
