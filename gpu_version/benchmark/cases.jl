# Shared case definitions for the CPU/GPU benchmarks.
#
# Every case mirrors one of the scripts in `example/` (same constants, kernel,
# viscosity and density diffusion model) but runs only for a short physical
# time so that a few hundred time steps are taken. `SimulationTime` is chosen
# per case so that the number of steps is comparable.
#
# The file is included by `benchmark_cpu.jl` and `benchmark_gpu.jl`, which
# bring `SPHExample` or `SPHExampleGPU` into scope beforehand. Both packages
# export the same names, so the definitions below work for either.

using StaticArrays

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
inputpath(args...) = joinpath(REPO_ROOT, "input", args...)

"""
A benchmark case: a name, the dimensionality and a constructor
`(FloatType, SaveLocation) -> NamedTuple` returning the keyword arguments
that `RunSimulation` needs (except the logger and the particles).
"""
struct BenchCase
    name::String
    dims::Int
    build::Function
end

function still_wedge_mdbc(::Type{T}, save; dx = 0.02) where {T}
    D = 2
    consts = SimulationConstants{T}(dx=dx, c₀=42.48576250492629, δᵩ=0.1, CFL=0.5)
    geom = [
        Geometry{D,T}(CSVFile=inputpath("still_wedge_mdbc", "StillWedge_Dp$(dx)_Bound.csv"),
                      GroupMarker=1, Type=Fixed),
        Geometry{D,T}(CSVFile=inputpath("still_wedge_mdbc", "StillWedge_Dp$(dx)_Fluid.csv"),
                      GroupMarker=2, Type=Fluid),
    ]
    meta = SimulationMetaData{D,T}(SimulationName="StillWedgeMDBC", SaveLocation=save,
        SimulationTime=0.2, OutputTimes=0.05, VisualizeInParaview=false, OpenLogFile=false,
        ExportSingleVTKHDF=true, ExportGridCells=false, FlagLog=true, FlagMDBCSimple=true)
    return (SimGeometry=geom, SimMetaData=meta, SimConstants=consts,
            SimKernel=SPHKernelInstance{D,T}(WendlandC2(); dx=consts.dx),
            SimViscosity=ArtificialViscosity(), SimDensityDiffusion=LinearDensityDiffusion(),
            ParticleNormalsPath=inputpath("still_wedge_mdbc", "StillWedge_Dp$(dx)_GhostNodes_Correct.csv"))
end

function still_wedge_dbc(::Type{T}, save; dx = 0.01) where {T}
    D = 2
    consts = SimulationConstants{T}(dx=dx, c₀=43.4, δᵩ=0.1, CFL=0.2)
    geom = [
        Geometry{D,T}(CSVFile=inputpath("still_wedge", "StillWedge_Dp$(dx)_Bound.csv"),
                      GroupMarker=1, Type=Fixed),
        Geometry{D,T}(CSVFile=inputpath("still_wedge", "StillWedge_Dp$(dx)_Fluid.csv"),
                      GroupMarker=2, Type=Fluid),
    ]
    meta = SimulationMetaData{D,T}(SimulationName="StillWedgeDBC", SaveLocation=save,
        SimulationTime=0.05, OutputTimes=0.025, VisualizeInParaview=false, OpenLogFile=false,
        ExportSingleVTKHDF=true, ExportGridCells=false, FlagLog=true)
    return (SimGeometry=geom, SimMetaData=meta, SimConstants=consts,
            SimKernel=SPHKernelInstance{D,T}(WendlandC2(); dx=consts.dx),
            SimViscosity=ArtificialViscosity(), SimDensityDiffusion=LinearDensityDiffusion(),
            ParticleNormalsPath=nothing)
end

function dambreak_2d_mdbc(::Type{T}, save) where {T}
    D = 2
    consts = SimulationConstants{T}(dx=0.01, c₀=88.14487860902641, δᵩ=0.1, CFL=0.5, α=0.01)
    geom = [
        Geometry{D,T}(CSVFile=inputpath("dam_break_2d", "DamBreak2d_Dp0.02_MDBC_Bound_ThreeLayers.csv"),
                      GroupMarker=1, Type=Fixed),
        Geometry{D,T}(CSVFile=inputpath("dam_break_2d", "DamBreak2d_Dp0.02_MDBC_Fluid_ThreeLayers.csv"),
                      GroupMarker=2, Type=Fluid),
    ]
    meta = SimulationMetaData{D,T}(SimulationName="DamBreak2DMDBC", SaveLocation=save,
        SimulationTime=0.05, OutputTimes=0.025, VisualizeInParaview=false, OpenLogFile=false,
        ExportSingleVTKHDF=true, ExportGridCells=false, FlagLog=true, FlagMDBCSimple=true)
    return (SimGeometry=geom, SimMetaData=meta, SimConstants=consts,
            SimKernel=SPHKernelInstance{D,T}(WendlandC2(); dx=consts.dx),
            SimViscosity=ArtificialViscosity(), SimDensityDiffusion=LinearDensityDiffusion(),
            ParticleNormalsPath=inputpath("dam_break_2d", "DamBreak2d_Dp0.02_MDBC_GhostNodes_ThreeLayers.csv"))
end

function moving_square_2d(::Type{T}, save; dx = 0.04) where {T}
    D = 2
    consts = SimulationConstants{T}(dx=dx, c₀=28, δᵩ=0.1, g=0, Cb=112000, α=1e-6, CFL=0.2)
    geom = [
        Geometry{D,T}(CSVFile=inputpath("moving_square_2d", "MovingSquare_Dp$(dx)_Fixed.csv"),
                      GroupMarker=1, Type=Fixed),
        Geometry{D,T}(CSVFile=inputpath("moving_square_2d", "MovingSquare_Dp$(dx)_Fluid.csv"),
                      GroupMarker=2, Type=Fluid),
        Geometry{D,T}(CSVFile=inputpath("moving_square_2d", "MovingSquare_Dp$(dx)_Square.csv"),
                      GroupMarker=3, Type=Moving,
                      Motion=MotionDetails{D,T}(Velocity=2.8, StartTime=0.0, Duration=3.0,
                                                 Direction=SVector{D,T}(1.0, 0.0))),
    ]
    meta = SimulationMetaData{D,T}(SimulationName="MovingSquare2D", SaveLocation=save,
        SimulationTime=0.05, OutputTimes=0.025, VisualizeInParaview=false, OpenLogFile=false,
        ExportSingleVTKHDF=true, ExportGridCells=false, FlagLog=true, FlagShifting=true)
    return (SimGeometry=geom, SimMetaData=meta, SimConstants=consts,
            SimKernel=SPHKernelInstance{D,T}(WendlandC2(); dx=consts.dx, k=T(sqrt(2))),
            SimViscosity=LaminarSPS(), SimDensityDiffusion=ZeroGravityLinearDensityDiffusion(),
            ParticleNormalsPath=nothing)
end

function dambreak_3d(::Type{T}, save; dx = 0.02) where {T}
    D = 3
    consts = SimulationConstants{T}(dx=dx, c₀=33.14, α=0.1, m₀=1000*dx^3, CFL=0.2)
    geom = [
        Geometry{D,T}(CSVFile=inputpath("dam_break_3d", "DamBreak3d_Dp$(dx)_Bound.csv"),
                      GroupMarker=1, Type=Fixed),
        Geometry{D,T}(CSVFile=inputpath("dam_break_3d", "DamBreak3d_Dp$(dx)_Fluid.csv"),
                      GroupMarker=2, Type=Fluid),
    ]
    meta = SimulationMetaData{D,T}(SimulationName="DamBreak3D", SaveLocation=save,
        SimulationTime=0.02, OutputTimes=0.01, VisualizeInParaview=false, OpenLogFile=false,
        ExportSingleVTKHDF=true, ExportGridCells=false, FlagLog=true)
    return (SimGeometry=geom, SimMetaData=meta, SimConstants=consts,
            SimKernel=SPHKernelInstance{D,T}(WendlandC2(); h=T(sqrt(3*dx^2))),
            SimViscosity=ArtificialViscosity(), SimDensityDiffusion=LinearDensityDiffusion(),
            ParticleNormalsPath=nothing)
end

function duckling_3d_mdbc(::Type{T}, save; dx = 0.01) where {T}
    D = 3
    consts = SimulationConstants{T}(dx=dx, c₀=23.43842998154953, δᵩ=0.1, CFL=0.2, α=0.02,
                                    m₀=1000*dx^3)
    geom = [
        Geometry{D,T}(CSVFile=inputpath("case_duckling_mdbc", "CaseDuckling_Dp$(dx)_Bound_MDBC.csv"),
                      GroupMarker=1, Type=Fixed),
        Geometry{D,T}(CSVFile=inputpath("case_duckling_mdbc", "CaseDuckling_Dp$(dx)_Fluid_MDBC.csv"),
                      GroupMarker=2, Type=Fluid),
    ]
    meta = SimulationMetaData{D,T}(SimulationName="Duckling3DMDBC", SaveLocation=save,
        SimulationTime=0.03, OutputTimes=0.015, VisualizeInParaview=false, OpenLogFile=false,
        ExportSingleVTKHDF=true, ExportGridCells=false, FlagLog=true, FlagMDBCSimple=true)
    return (SimGeometry=geom, SimMetaData=meta, SimConstants=consts,
            SimKernel=SPHKernelInstance{D,T}(WendlandC2(); dx=consts.dx, k=T(1.5)),
            SimViscosity=ArtificialViscosity(), SimDensityDiffusion=LinearDensityDiffusion(),
            ParticleNormalsPath=inputpath("case_duckling_mdbc", "CaseDuckling_Dp$(dx)_GhostNodes.csv"))
end

const BENCH_CASES = [
    BenchCase("StillWedge2D_MDBC_dp0.02",     2, (T, s) -> still_wedge_mdbc(T, s)),
    BenchCase("DamBreak2D_MDBC_dp0.01",       2, (T, s) -> dambreak_2d_mdbc(T, s)),
    BenchCase("StillWedge2D_DBC_dp0.01",      2, (T, s) -> still_wedge_dbc(T, s)),
    BenchCase("MovingSquare2D_dp0.04",        2, (T, s) -> moving_square_2d(T, s; dx=0.04)),
    BenchCase("MovingSquare2D_dp0.02",        2, (T, s) -> moving_square_2d(T, s; dx=0.02)),
    BenchCase("DamBreak3D_dp0.02",            3, (T, s) -> dambreak_3d(T, s; dx=0.02)),
    BenchCase("Duckling3D_MDBC_dp0.01",       3, (T, s) -> duckling_3d_mdbc(T, s; dx=0.01)),
    BenchCase("DamBreak3D_dp0.0085",          3, (T, s) -> dambreak_3d(T, s; dx=0.0085)),
    BenchCase("Duckling3D_MDBC_dp0.005",      3, (T, s) -> duckling_3d_mdbc(T, s; dx=0.005)),
]

"""
    select_cases(args)

Return the subset of `BENCH_CASES` whose names contain any of the substrings in
`args`. With no arguments every case is returned.
"""
function select_cases(args)
    isempty(args) && return BENCH_CASES
    filter(c -> any(occursin(a, c.name) for a in args), BENCH_CASES)
end
