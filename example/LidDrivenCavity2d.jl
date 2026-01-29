using Printf
using StaticArrays: SVector
using SPHExample

function WriteParticleRow!(Io, PointId, Idp, Marker, X, Y, Density, TypeValue, Vx, Vy)
    PointsMagnitude = sqrt(X^2 + Y^2)
    VelocityMagnitude = sqrt(Vx^2 + Vy^2)
    println(Io, join((PointId, Idp, Marker, X, 0.0, Y, PointsMagnitude, Density, TypeValue, Vx, 0.0, Vy, VelocityMagnitude), ","))
    return nothing
end

function WriteGhostRow!(Io, Idp, Marker, Nx, Ny, X, Y)
    NormalSize = sqrt(Nx^2 + Ny^2)
    println(Io, join((Idp, Marker, Nx, 0.0, Ny, NormalSize, X, 0.0, Y), ","))
    return nothing
end

function GenerateLidDrivenCavityCSVs(
    Resolution,
    Dx,
    InputFolder;
    Density=1000.0,
    LidVelocity=0.0,
    LidType=Moving,
    ForceRegenerate=false
)
    BaseName = "LidDrivenCavity_N$(Resolution)"
    FluidFile = joinpath(InputFolder, "$(BaseName)_Fluid.csv")
    FixedFile = joinpath(InputFolder, "$(BaseName)_Fixed.csv")
    LidFile = joinpath(InputFolder, "$(BaseName)_Lid.csv")

    if !ForceRegenerate && isfile(FluidFile) && isfile(FixedFile) && isfile(LidFile)
        return FluidFile, FixedFile, LidFile
    end

    mkpath(InputFolder)

    Header = "\"Point ID\", \"Idp\", \"Mk\", \"Points:0\", \"Points:1\", \"Points:2\", \"Points Magnitude\", \"Rhop\", \"Type\", \"Vel:0\", \"Vel:1\", \"Vel:2\", \"Vel Magnitude\""
    IdpCounter = 0

    open(FixedFile, "w") do Io
        println(Io, Header)
        PointId = 0
        for X in range(-2 * Dx, length=Resolution + 5, step=Dx)
            WriteParticleRow!(Io, PointId, IdpCounter, 1, X, 0.0, Density, Int(Fixed), 0.0, 0.0)
            PointId += 1
            IdpCounter += 1
        end
        for X in range(-2 * Dx, length=Resolution + 5, step=Dx)
            WriteParticleRow!(Io, PointId, IdpCounter, 1, X, -Dx, Density, Int(Fixed), 0.0, 0.0)
            PointId += 1
            IdpCounter += 1
        end
        for X in range(-2 * Dx, length=Resolution + 5, step=Dx)
            WriteParticleRow!(Io, PointId, IdpCounter, 1, X, -2 * Dx, Density, Int(Fixed), 0.0, 0.0)
            PointId += 1
            IdpCounter += 1
        end
        for Y in range(Dx, length=Resolution - 3, step=Dx)
            WriteParticleRow!(Io, PointId, IdpCounter, 1, 0.0, Y, Density, Int(Fixed), 0.0, 0.0)
            PointId += 1
            IdpCounter += 1
            WriteParticleRow!(Io, PointId, IdpCounter, 1, 1.0, Y, Density, Int(Fixed), 0.0, 0.0)
            PointId += 1
            IdpCounter += 1
        end
        for Y in range(Dx, length=Resolution - 3, step=Dx)
            WriteParticleRow!(Io, PointId, IdpCounter, 1, -Dx, Y, Density, Int(Fixed), 0.0, 0.0)
            PointId += 1
            IdpCounter += 1
            WriteParticleRow!(Io, PointId, IdpCounter, 1, 1.0 + Dx, Y, Density, Int(Fixed), 0.0, 0.0)
            PointId += 1
            IdpCounter += 1
        end
        for Y in range(Dx, length=Resolution - 3, step=Dx)
            WriteParticleRow!(Io, PointId, IdpCounter, 1, -2 * Dx, Y, Density, Int(Fixed), 0.0, 0.0)
            PointId += 1
            IdpCounter += 1
            WriteParticleRow!(Io, PointId, IdpCounter, 1, 1.0 + 2 * Dx, Y, Density, Int(Fixed), 0.0, 0.0)
            PointId += 1
            IdpCounter += 1
        end
    end

    open(LidFile, "w") do Io
        println(Io, Header)
        PointId = 0
        for Y in (1.0, 1.0 - Dx, 1.0 - 2 * Dx)
            for X in range(-2 * Dx, length=Resolution + 5, step=Dx)
                WriteParticleRow!(Io, PointId, IdpCounter, 3, X, Y, Density, Int(LidType), LidVelocity, 0.0)
                PointId += 1
                IdpCounter += 1
            end
        end
    end

    open(FluidFile, "w") do Io
        println(Io, Header)
        PointId = 0
        for Y in range(Dx, length=Resolution - 3, step=Dx)
            for X in range(Dx, length=Resolution - 1, step=Dx)
                WriteParticleRow!(Io, PointId, IdpCounter, 2, X, Y, Density, Int(Fluid), 0.0, 0.0)
                PointId += 1
                IdpCounter += 1
            end
        end
    end

    return FluidFile, FixedFile, LidFile
end

function GenerateLidDrivenCavityGhostNodes(Resolution, Dx, InputFolder; ForceRegenerate=false)
    BaseName = "LidDrivenCavity_N$(Resolution)"
    GhostFile = joinpath(InputFolder, "$(BaseName)_GhostNodes.csv")

    if !ForceRegenerate && isfile(GhostFile)
        return GhostFile
    end

    mkpath(InputFolder)

    Header = "\"Idp\",\"Mk\",\"Normal:0\",\"Normal:1\",\"Normal:2\",\"NormalSize\",\"Points:0\",\"Points:1\",\"Points:2\""
    IdpCounter = 0
    NormalScale = Dx

    open(GhostFile, "w") do Io
        println(Io, Header)
        for X in range(-2 * Dx, length=Resolution + 5, step=Dx)
            WriteGhostRow!(Io, IdpCounter, 1, 0.0, NormalScale, X, 0.0)
            IdpCounter += 1
        end
        for X in range(-2 * Dx, length=Resolution + 5, step=Dx)
            WriteGhostRow!(Io, IdpCounter, 1, 0.0, NormalScale, X, -Dx)
            IdpCounter += 1
        end
        for X in range(-2 * Dx, length=Resolution + 5, step=Dx)
            WriteGhostRow!(Io, IdpCounter, 1, 0.0, NormalScale, X, -2 * Dx)
            IdpCounter += 1
        end
        for Y in range(Dx, length=Resolution - 3, step=Dx)
            WriteGhostRow!(Io, IdpCounter, 1, NormalScale, 0.0, 0.0, Y)
            IdpCounter += 1
            WriteGhostRow!(Io, IdpCounter, 1, -NormalScale, 0.0, 1.0, Y)
            IdpCounter += 1
        end
        for Y in range(Dx, length=Resolution - 3, step=Dx)
            WriteGhostRow!(Io, IdpCounter, 1, NormalScale, 0.0, -Dx, Y)
            IdpCounter += 1
            WriteGhostRow!(Io, IdpCounter, 1, -NormalScale, 0.0, 1.0 + Dx, Y)
            IdpCounter += 1
        end
        for Y in range(Dx, length=Resolution - 3, step=Dx)
            WriteGhostRow!(Io, IdpCounter, 1, NormalScale, 0.0, -2 * Dx, Y)
            IdpCounter += 1
            WriteGhostRow!(Io, IdpCounter, 1, -NormalScale, 0.0, 1.0 + 2 * Dx, Y)
            IdpCounter += 1
        end
        for Y in (1.0, 1.0 - Dx, 1.0 - 2 * Dx)
            for X in range(-2 * Dx, length=Resolution + 5, step=Dx)
                WriteGhostRow!(Io, IdpCounter, 3, 0.0, -NormalScale, X, Y)
                IdpCounter += 1
            end
        end
    end

    return GhostFile
end

let
    Dimensions = 2
    FloatType = Float64

    Resolution = 200
    Reynolds = 1000.0
    LidVelocity = 1.0
    LidType = FixedMoving
    LidMovesPosition = LidType == Moving
    RegenerateCSVs = true
    DomainLength = 1.0
    KernelScale = 1.2 * sqrt(2)

    Dx = DomainLength / Resolution
    KinematicViscosity = LidVelocity * DomainLength / Reynolds

    SimulationTime = 100.5
    OutputInterval = 0.1

    InputFolder = "./input/lid_driven_cavity"
    FluidCSV, FixedCSV, LidCSV = GenerateLidDrivenCavityCSVs(
        Resolution,
        Dx,
        InputFolder;
        LidVelocity = LidVelocity,
        LidType = LidType,
        ForceRegenerate = RegenerateCSVs
    )
    GhostCSV = GenerateLidDrivenCavityGhostNodes(Resolution, Dx, InputFolder; ForceRegenerate = RegenerateCSVs)

    SimConstants = SimulationConstants{FloatType}(
        dx = Dx,
        ν₀ = KinematicViscosity,
        g = 0.0,
        c₀ = 10LidVelocity,
        α = 0.0001,
        CFL = 0.2,
        δᵩ = 0.1,
        A = 0.01
    )

    SimMetaData = SimulationMetaData{Dimensions, FloatType, PlanarShifting, NoKernelOutput, SimpleMDBC, StoreLog}(
        SimulationName = "LidDrivenCavityRe$(Int(Reynolds))",
        SaveLocation = "output/LidDrivenCavityRe$(Int(Reynolds))_N$(Resolution)",
        SimulationTime = SimulationTime,
        OutputTimes = OutputInterval,
        VisualizeInParaview = true,
        ExportSingleVTKHDF = true,
        OpenLogFile = true
    )

    if !isdir(SimMetaData.SaveLocation)
        mkdir(SimMetaData.SaveLocation)
    end

    FixedWalls = Geometry{Dimensions, FloatType}(
        CSVFile = FixedCSV,
        GroupMarker = 1,
        Type = Fixed,
        Motion = nothing
    )

    FluidDomain = Geometry{Dimensions, FloatType}(
        CSVFile = FluidCSV,
        GroupMarker = 2,
        Type = Fluid,
        Motion = nothing
    )

    MovingLid = Geometry{Dimensions, FloatType}(
        CSVFile = LidCSV,
        GroupMarker = 3,
        Type = LidType,
        Motion = (LidType == Moving || LidType == FixedMoving) ? MotionDetails{Dimensions, FloatType}(
            Velocity = LidVelocity,
            StartTime = 0.0,
            Duration = SimulationTime,
            Direction = SVector{Dimensions, FloatType}(1.0, 0.0),
            MovePosition = LidMovesPosition
        ) : nothing
    )

    SimulationGeometry = [FixedWalls, FluidDomain, MovingLid]
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaData)

    SimLogger = SimulationLogger(SimMetaData.SaveLocation)
    SimKernel = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); dx = SimConstants.dx, k = KernelScale)

    CleanUpSimulationFolder(SimMetaData.SaveLocation)

    SimViscosity = ArtificialViscosity()

    RunSimulation(
        SimGeometry = SimulationGeometry,
        SimMetaData = SimMetaData,
        SimConstants = SimConstants,
        SimLogger = SimLogger,
        SimParticles = SimParticles,
        SimKernel = SimKernel,
        SimViscosity = SimViscosity,
        SimDensityDiffusion = ZeroGravityLinearDensityDiffusion(),
        SimTimeStepping = SingleNeighborTimeStepping(),
        ParticleNormalsPath = GhostCSV
    )
end
