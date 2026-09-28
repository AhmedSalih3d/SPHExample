# Shared case definitions for the CPU/GPU benchmarks.
#
# Every case mirrors one of the scripts in `example/` (same constants, kernel,
# viscosity and density diffusion model) but runs only for a short physical
# time so that a few hundred time steps are taken. `SimulationTime` is chosen
# per case so that the number of steps is comparable.
#
# The file is included by `benchmark_cpu.jl` and `benchmark_gpu.jl`, which
# bring `SPHExample` or `SPHExampleGPU` into scope beforehand. Both packages
# export the same names, so the definitions below work for either: the
# shifting, kernel output, mDBC and log modes are the type parameters of
# `SimulationMetaData`, and the time stepping scheme is passed to
# `RunSimulation` as `SimTimeStepping`.

using StaticArrays
using TimerOutputs

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
inputpath(args...) = joinpath(REPO_ROOT, "input", args...)

# Scheme used by every case. The symplectic scheme (two neighbour loops per
# step) is the one the GPU kernels are validated against.
const BENCH_TIME_STEPPING = SymplecticTimeStepping()

"""
    loop_time(hg)

Seconds spent in the time stepping loop. The CPU package times every step
under "00 Simulation Step", the GPU package every output interval under
"00 SimulationLoop".
"""
function loop_time(hg::TimerOutput)
    for key in ("00 SimulationLoop", "00 Simulation Step")
        haskey(hg.inner_timers, key) && return TimerOutputs.time(hg[key]) / 1e9
    end
    error("no time stepping loop timer found; top level timers: $(collect(keys(hg.inner_timers)))")
end

"""
A benchmark case: a name, the dimensionality and a constructor
`(FloatType, SaveLocation) -> NamedTuple` returning the keyword arguments
that `RunSimulation` needs (except the logger and the particles). Allocate
the particles with `AllocateDataStructures(kw.SimGeometry, kw.SimMetaData)`.
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
        SPHGeometry{D,T}(CSVFile=inputpath("still_wedge_mdbc", "StillWedge_Dp$(dx)_Bound.csv"),
                      GroupMarker=1, Type=Fixed),
        SPHGeometry{D,T}(CSVFile=inputpath("still_wedge_mdbc", "StillWedge_Dp$(dx)_Fluid.csv"),
                      GroupMarker=2, Type=Fluid),
    ]
    meta = SimulationMetaData{D,T,NoShifting,NoKernelOutput,SimpleMDBC,StoreLog}(SimulationName="StillWedgeMDBC", SaveLocation=save,
        SimulationTime=0.2, OutputTimes=0.05, VisualizeInParaview=false, OpenLogFile=false,
        ExportSingleVTKHDF=true, ExportGridCells=false)
    return (SimGeometry=geom, SimMetaData=meta, SimConstants=consts,
            SimKernel=SPHKernelInstance{D,T}(WendlandC2(); dx=consts.dx),
            SimViscosity=ArtificialViscosity(), SimDensityDiffusion=LinearDensityDiffusion(),
            SimTimeStepping=BENCH_TIME_STEPPING,
            ParticleNormalsPath=inputpath("still_wedge_mdbc", "StillWedge_Dp$(dx)_GhostNodes_Correct.csv"))
end

function still_wedge_dbc(::Type{T}, save; dx = 0.01) where {T}
    D = 2
    consts = SimulationConstants{T}(dx=dx, c₀=43.4, δᵩ=0.1, CFL=0.2)
    geom = [
        SPHGeometry{D,T}(CSVFile=inputpath("still_wedge", "StillWedge_Dp$(dx)_Bound.csv"),
                      GroupMarker=1, Type=Fixed),
        SPHGeometry{D,T}(CSVFile=inputpath("still_wedge", "StillWedge_Dp$(dx)_Fluid.csv"),
                      GroupMarker=2, Type=Fluid),
    ]
    meta = SimulationMetaData{D,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(SimulationName="StillWedgeDBC", SaveLocation=save,
        SimulationTime=0.05, OutputTimes=0.025, VisualizeInParaview=false, OpenLogFile=false,
        ExportSingleVTKHDF=true, ExportGridCells=false)
    return (SimGeometry=geom, SimMetaData=meta, SimConstants=consts,
            SimKernel=SPHKernelInstance{D,T}(WendlandC2(); dx=consts.dx),
            SimViscosity=ArtificialViscosity(), SimDensityDiffusion=LinearDensityDiffusion(),
            SimTimeStepping=BENCH_TIME_STEPPING,
            ParticleNormalsPath=nothing)
end

function dambreak_2d_mdbc(::Type{T}, save) where {T}
    D = 2
    consts = SimulationConstants{T}(dx=0.01, c₀=88.14487860902641, δᵩ=0.1, CFL=0.5, α=0.01)
    geom = [
        SPHGeometry{D,T}(CSVFile=inputpath("dam_break_2d", "DamBreak2d_Dp0.02_MDBC_Bound_ThreeLayers.csv"),
                      GroupMarker=1, Type=Fixed),
        SPHGeometry{D,T}(CSVFile=inputpath("dam_break_2d", "DamBreak2d_Dp0.02_MDBC_Fluid_ThreeLayers.csv"),
                      GroupMarker=2, Type=Fluid),
    ]
    meta = SimulationMetaData{D,T,NoShifting,NoKernelOutput,SimpleMDBC,StoreLog}(SimulationName="DamBreak2DMDBC", SaveLocation=save,
        SimulationTime=0.05, OutputTimes=0.025, VisualizeInParaview=false, OpenLogFile=false,
        ExportSingleVTKHDF=true, ExportGridCells=false)
    return (SimGeometry=geom, SimMetaData=meta, SimConstants=consts,
            SimKernel=SPHKernelInstance{D,T}(WendlandC2(); dx=consts.dx),
            SimViscosity=ArtificialViscosity(), SimDensityDiffusion=LinearDensityDiffusion(),
            SimTimeStepping=BENCH_TIME_STEPPING,
            ParticleNormalsPath=inputpath("dam_break_2d", "DamBreak2d_Dp0.02_MDBC_GhostNodes_ThreeLayers.csv"))
end

function moving_square_2d(::Type{T}, save; dx = 0.04) where {T}
    D = 2
    consts = SimulationConstants{T}(dx=dx, c₀=28, δᵩ=0.1, g=0, Cb=112000, α=1e-6, CFL=0.2)
    geom = [
        SPHGeometry{D,T}(CSVFile=inputpath("moving_square_2d", "MovingSquare_Dp$(dx)_Fixed.csv"),
                      GroupMarker=1, Type=Fixed),
        SPHGeometry{D,T}(CSVFile=inputpath("moving_square_2d", "MovingSquare_Dp$(dx)_Fluid.csv"),
                      GroupMarker=2, Type=Fluid),
        SPHGeometry{D,T}(CSVFile=inputpath("moving_square_2d", "MovingSquare_Dp$(dx)_Square.csv"),
                      GroupMarker=3, Type=Moving,
                      Motion=MotionDetails{D,T}(Velocity=2.8, StartTime=0.0, Duration=3.0,
                                                 Direction=SVector{D,T}(1.0, 0.0))),
    ]
    meta = SimulationMetaData{D,T,PlanarShifting,NoKernelOutput,NoMDBC,StoreLog}(SimulationName="MovingSquare2D", SaveLocation=save,
        SimulationTime=0.05, OutputTimes=0.025, VisualizeInParaview=false, OpenLogFile=false,
        ExportSingleVTKHDF=true, ExportGridCells=false)
    return (SimGeometry=geom, SimMetaData=meta, SimConstants=consts,
            SimKernel=SPHKernelInstance{D,T}(WendlandC2(); dx=consts.dx, k=T(sqrt(2))),
            SimViscosity=LaminarSPS(), SimDensityDiffusion=ZeroGravityLinearDensityDiffusion(),
            SimTimeStepping=BENCH_TIME_STEPPING,
            ParticleNormalsPath=nothing)
end

function dambreak_3d(::Type{T}, save; dx = 0.02) where {T}
    D = 3
    consts = SimulationConstants{T}(dx=dx, c₀=33.14, α=0.1, m₀=1000*dx^3, CFL=0.2)
    geom = [
        SPHGeometry{D,T}(CSVFile=inputpath("dam_break_3d", "DamBreak3d_Dp$(dx)_Bound.csv"),
                      GroupMarker=1, Type=Fixed),
        SPHGeometry{D,T}(CSVFile=inputpath("dam_break_3d", "DamBreak3d_Dp$(dx)_Fluid.csv"),
                      GroupMarker=2, Type=Fluid),
    ]
    meta = SimulationMetaData{D,T,NoShifting,NoKernelOutput,NoMDBC,StoreLog}(SimulationName="DamBreak3D", SaveLocation=save,
        SimulationTime=0.02, OutputTimes=0.01, VisualizeInParaview=false, OpenLogFile=false,
        ExportSingleVTKHDF=true, ExportGridCells=false)
    return (SimGeometry=geom, SimMetaData=meta, SimConstants=consts,
            SimKernel=SPHKernelInstance{D,T}(WendlandC2(); h=T(sqrt(3*dx^2))),
            SimViscosity=ArtificialViscosity(), SimDensityDiffusion=LinearDensityDiffusion(),
            SimTimeStepping=BENCH_TIME_STEPPING,
            ParticleNormalsPath=nothing)
end

function duckling_3d_mdbc(::Type{T}, save; dx = 0.01) where {T}
    D = 3
    consts = SimulationConstants{T}(dx=dx, c₀=23.43842998154953, δᵩ=0.1, CFL=0.2, α=0.02,
                                    m₀=1000*dx^3)
    geom = [
        SPHGeometry{D,T}(CSVFile=inputpath("case_duckling_mdbc", "CaseDuckling_Dp$(dx)_Bound_MDBC.csv"),
                      GroupMarker=1, Type=Fixed),
        SPHGeometry{D,T}(CSVFile=inputpath("case_duckling_mdbc", "CaseDuckling_Dp$(dx)_Fluid_MDBC.csv"),
                      GroupMarker=2, Type=Fluid),
    ]
    meta = SimulationMetaData{D,T,NoShifting,NoKernelOutput,SimpleMDBC,StoreLog}(SimulationName="Duckling3DMDBC", SaveLocation=save,
        SimulationTime=0.03, OutputTimes=0.015, VisualizeInParaview=false, OpenLogFile=false,
        ExportSingleVTKHDF=true, ExportGridCells=false)
    return (SimGeometry=geom, SimMetaData=meta, SimConstants=consts,
            SimKernel=SPHKernelInstance{D,T}(WendlandC2(); dx=consts.dx, k=T(1.5)),
            SimViscosity=ArtificialViscosity(), SimDensityDiffusion=LinearDensityDiffusion(),
            SimTimeStepping=BENCH_TIME_STEPPING,
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
