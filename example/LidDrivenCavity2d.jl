using Printf
using StaticArrays: SVector
using SPHExample

function WriteParticleRow!(Io, PointId, Idp, Marker, X, Y, Density, TypeValue, Vx, Vy)
    PointsMagnitude = sqrt(X^2 + Y^2)
    VelocityMagnitude = sqrt(Vx^2 + Vy^2)
    println(Io, join((PointId, Idp, Marker, X, 0.0, Y, PointsMagnitude, Density, TypeValue, Vx, 0.0, Vy, VelocityMagnitude), ","))
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

    open(FluidFile, "w") do Io
        println(Io, Header)
        PointId = 0
        for Y in range(Dx / 2, length=Resolution, step=Dx)
            for X in range(Dx / 2, length=Resolution, step=Dx)
                WriteParticleRow!(Io, PointId, IdpCounter, 2, X, Y, Density, Int(Fluid), 0.0, 0.0)
                PointId += 1
                IdpCounter += 1
            end
        end
    end

    open(FixedFile, "w") do Io
        println(Io, Header)
        PointId = 0
        for X in range(0.0, length=Resolution + 1, step=Dx)
            WriteParticleRow!(Io, PointId, IdpCounter, 1, X, 0.0, Density, Int(Fixed), 0.0, 0.0)
            PointId += 1
            IdpCounter += 1
        end
        for Y in range(Dx, length=Resolution - 1, step=Dx)
            WriteParticleRow!(Io, PointId, IdpCounter, 1, 0.0, Y, Density, Int(Fixed), 0.0, 0.0)
            PointId += 1
            IdpCounter += 1
            WriteParticleRow!(Io, PointId, IdpCounter, 1, 1.0, Y, Density, Int(Fixed), 0.0, 0.0)
            PointId += 1
            IdpCounter += 1
        end
    end

    open(LidFile, "w") do Io
        println(Io, Header)
        PointId = 0
        for X in range(0.0, length=Resolution + 1, step=Dx)
            WriteParticleRow!(Io, PointId, IdpCounter, 3, X, 1.0, Density, Int(LidType), LidVelocity, 0.0)
            PointId += 1
            IdpCounter += 1
        end
    end

    return FluidFile, FixedFile, LidFile
end

let
    Dimensions = 2
    FloatType = Float64

    Resolution = 50
    Reynolds = 1000.0
    LidVelocity = 1.0
    LidType = FixedMoving
    LidMovesPosition = LidType == Moving
    RegenerateCSVs = true
    DomainLength = 1.0
    KernelScale = 2.0

    Dx = DomainLength / Resolution
    KinematicViscosity = LidVelocity * DomainLength / Reynolds

    SimulationTime = 5.0
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

    SimConstants = SimulationConstants{FloatType}(
        dx = Dx,
        ν₀ = KinematicViscosity,
        g = 0.0,
        c₀ = max(20 * LidVelocity, 20.0),
        α = 0.01,
        CFL = 0.2,
        δᵩ = 0.1
    )

    SimMetaData = SimulationMetaData{Dimensions, FloatType, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(
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

    SimViscosity = Reynolds >= 5000 ? LaminarSPS() : Laminar()

    RunSimulation(
        SimGeometry = SimulationGeometry,
        SimMetaData = SimMetaData,
        SimConstants = SimConstants,
        SimLogger = SimLogger,
        SimParticles = SimParticles,
        SimKernel = SimKernel,
        SimViscosity = SimViscosity,
        SimDensityDiffusion = LinearDensityDiffusion(),
        SimTimeStepping = SingleNeighborTimeStepping()
    )
end
