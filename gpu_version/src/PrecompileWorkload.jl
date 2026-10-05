# Precompilation workload (PrecompileTools.jl).
#
# Runs a few tiny simulations while the package is precompiled, so that the
# host code (CSV input, logging, VTKHDF output, the time loop) and, on a
# machine with a working GPU, the CUDA kernels are compiled once and stored in
# the package image instead of being compiled at the start of every Julia
# session. The cases mirror the configurations of the scripts in `example/`.
#
# Only the combinations of types exercised here are cached. Other float types,
# kernels or viscosity / density diffusion models still work, but are compiled
# on first use. Two preferences control the workload (set them with
# `Preferences.set_preferences!(SPHExampleGPU, ...)` and restart Julia):
#   - `precompile_float_types`: float types to precompile, default `["Float32"]`
#   - `precompile_gpu_lanes`:   lanes per particle compiled for the gather
#                               kernels, default `[1]`; automatic selection
#                               is also compiled
#   - `precompile_double_position`: also run every case with
#                               `GPUDoublePosition = true` (the cell relative
#                               kernel variants), default `false`
# `precompile_workload = false` (PrecompileTools) disables the workload.

using PrecompileTools: @setup_workload, @compile_workload
using Preferences: @load_preference
using StaticArrays: SVector

const _PRECOMPILE_FLOAT_TYPES = let
    known = Dict("Float32" => Float32, "Float64" => Float64)
    names = @load_preference("precompile_float_types", ["Float32"])
    Type[known[n] for n in names if haskey(known, n)]
end

const _PRECOMPILE_GPU_LANES = let
    lanes = @load_preference("precompile_gpu_lanes", [1])
    Int[k for k in lanes if ispow2(k) && 1 <= k <= 32]
end

const _PRECOMPILE_DOUBLE_POSITION = @load_preference("precompile_double_position", false)::Bool

# Write a small box of particles in the CSV layout of the files in `input/`: a
# three layer floor, a fluid block and a block that can be moved. In 2D the
# particles lie in the x-z plane like the DualSPHysics exports. Returns the
# paths of the particle files and of the mDBC ghost node file of the floor.
function _write_precompile_geometry(dir::String, D::Int, dx::Float64)
    ny    = D == 3 ? 6 : 1
    jfl   = D == 3 ? (1:4) : (0:0)
    bound = [(i * dx, j * dx, -l * dx) for l in 0:2 for j in 0:(ny - 1) for i in 0:11]
    fluid = [(i * dx, j * dx, k * dx) for k in 1:4 for j in jfl for i in 2:5]
    solid = [(i * dx, j * dx, k * dx) for k in 1:2 for j in jfl for i in 8:9]

    function write_particles(name, points, first_id)
        path = joinpath(dir, "$(name)_$(D)d.csv")
        open(path, "w") do io
            println(io, "\"Idp\",\"Rhop\",\"Points:0\",\"Points:1\",\"Points:2\"")
            for (n, p) in enumerate(points)
                println(io, first_id + n - 1, ",1000.0,", p[1], ",", p[2], ",", p[3])
            end
        end
        return path
    end

    paths = (bound = write_particles("bound", bound, 0),
             fluid = write_particles("fluid", fluid, length(bound)),
             solid = write_particles("solid", solid, length(bound) + length(fluid)),
             ghost = joinpath(dir, "ghost_$(D)d.csv"))

    # Ghost nodes are the floor particles mirrored at the wall (z = dx / 2).
    open(paths.ghost, "w") do io
        println(io, "\"Idp\",\"Normal:0\",\"Normal:1\",\"Normal:2\",\"Points:0\",\"Points:1\",\"Points:2\"")
        for (n, p) in enumerate(bound)
            println(io, n - 1, ",0.0,0.0,", dx - 2 * p[3], ",", p[1], ",", p[2], ",", p[3])
        end
    end
    return paths
end

# Keyword arguments of `RunSimulation` (without logger and particles) for the
# configurations of the example scripts, each with its own save location.
function _precompile_cases(dir::String, ::Type{T}) where {T}
    dx = 0.02
    g2 = _write_precompile_geometry(dir, 2, dx)
    g3 = _write_precompile_geometry(dir, 3, dx)
    save(name) = mkpath(joinpath(dir, "$(name)_$(T)"))
    # One short output interval exercises stepping and output without
    # simulating physical time that adds no further specializations.
    time = (SimulationTime = 1e-4, VisualizeInParaview = false, OpenLogFile = false)

    # example/Dambreak2dMDBC.jl, StillWedgeMDBC.jl
    c2mdbc = SimulationConstants{T}(dx = dx, c₀ = 88.14487860902641, δᵩ = 0.1, CFL = 0.5, α = 0.01)
    mdbc_2d = (
        SimGeometry  = [SPHGeometry{2, T}(CSVFile = g2.bound, GroupMarker = 1, Type = Fixed),
                        SPHGeometry{2, T}(CSVFile = g2.fluid, GroupMarker = 2, Type = Fluid)],
        SimMetaData  = SimulationMetaData{2, T, NoShifting, NoKernelOutput, SimpleMDBC, StoreLog}(;
                           SimulationName = "MDBC2D", SaveLocation = save("MDBC2D"),
                           OutputTimes = [time.SimulationTime], ExportGridCells = true, time...),
        SimConstants = c2mdbc,
        SimKernel    = SPHKernelInstance{2, T}(WendlandC2(); dx = c2mdbc.dx),
        SimViscosity = ArtificialViscosity(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        ParticleNormalsPath = g2.ghost,
    )

    # example/MovingSquare2d.jl
    c2move = SimulationConstants{T}(dx = dx, c₀ = 28, δᵩ = 0.1, g = 0, Cb = 112000, α = 1e-6, CFL = 0.2)
    moving_2d = (
        SimGeometry  = [SPHGeometry{2, T}(CSVFile = g2.bound, GroupMarker = 1, Type = Fixed),
                        SPHGeometry{2, T}(CSVFile = g2.fluid, GroupMarker = 2, Type = Fluid),
                        SPHGeometry{2, T}(CSVFile = g2.solid, GroupMarker = 3, Type = Moving,
                                       Motion = MotionDetails{2, T}(Velocity = 2.8, StartTime = 0.0,
                                                                    Duration = 3.0,
                                                                    Direction = SVector{2, T}(1, 0)))],
        SimMetaData  = SimulationMetaData{2, T, PlanarShifting, NoKernelOutput, NoMDBC, StoreLog}(;
                           SimulationName = "Moving2D", SaveLocation = save("Moving2D"),
                           OutputTimes = time.SimulationTime, time...),
        SimConstants = c2move,
        SimKernel    = SPHKernelInstance{2, T}(WendlandC2(); dx = c2move.dx, k = T(sqrt(2))),
        SimViscosity = LaminarSPS(),
        SimDensityDiffusion = ZeroGravityLinearDensityDiffusion(),
        ParticleNormalsPath = nothing,
    )

    # example/DucklingMDBC.jl
    c3mdbc = SimulationConstants{T}(dx = dx, c₀ = 23.43842998154953, δᵩ = 0.1, CFL = 0.2, α = 0.02,
                                    m₀ = 1000 * dx^3)
    mdbc_3d = (
        SimGeometry  = [SPHGeometry{3, T}(CSVFile = g3.bound, GroupMarker = 1, Type = Fixed),
                        SPHGeometry{3, T}(CSVFile = g3.fluid, GroupMarker = 2, Type = Fluid)],
        SimMetaData  = SimulationMetaData{3, T, NoShifting, NoKernelOutput, SimpleMDBC, StoreLog}(;
                           SimulationName = "MDBC3D", SaveLocation = save("MDBC3D"),
                           OutputTimes = time.SimulationTime, time...),
        SimConstants = c3mdbc,
        SimKernel    = SPHKernelInstance{3, T}(WendlandC2(); dx = c3mdbc.dx, k = T(1.5)),
        SimViscosity = ArtificialViscosity(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        ParticleNormalsPath = g3.ghost,
    )

    # example/Dambreak3d.jl
    c3dbc = SimulationConstants{T}(dx = dx, c₀ = 33.14, α = 0.1, m₀ = 1000 * dx^3, CFL = 0.2)
    dbc_3d = (
        SimGeometry  = [SPHGeometry{3, T}(CSVFile = g3.bound, GroupMarker = 1, Type = Fixed),
                        SPHGeometry{3, T}(CSVFile = g3.fluid, GroupMarker = 2, Type = Fluid)],
        SimMetaData  = SimulationMetaData{3, T, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(;
                           SimulationName = "DBC3D", SaveLocation = save("DBC3D"),
                           OutputTimes = time.SimulationTime, ExportGridCells = true,
                           GPUCellSubdivision = 2, time...),
        SimConstants = c3dbc,
        SimKernel    = SPHKernelInstance{3, T}(WendlandC2(); h = T(sqrt(3 * dx^2))),
        SimViscosity = ArtificialViscosity(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        ParticleNormalsPath = nothing,
    )

    return (mdbc_2d, moving_2d, mdbc_3d, dbc_3d)
end

# Complete run on the GPU, as done by the example scripts.
function _precompile_run(case, lanes::Int, double_position::Bool)
    meta = case.SimMetaData
    meta.GPULanesPerParticle = lanes
    meta.GPUDoublePosition   = double_position
    CleanUpSimulationFolder(meta.SaveLocation)
    particles = AllocateDataStructures(case.SimGeometry, meta)
    logger    = SimulationLogger(meta.SaveLocation)
    RunSimulation(; case..., SimLogger = logger, SimParticles = particles,
                    SimTimeStepping = SymplecticTimeStepping())
    return nothing
end

# Host side of a run (input, logging and output) for machines without a GPU.
function _precompile_host(case)
    meta = case.SimMetaData
    (; SimGeometry, SimConstants, SimKernel, SimViscosity, SimDensityDiffusion) = case
    CleanUpSimulationFolder(meta.SaveLocation)
    particles = AllocateDataStructures(SimGeometry, meta)
    D = length(eltype(particles.Position))
    if case.ParticleNormalsPath !== nothing
        LoadBoundaryNormals(Val(D), eltype(particles.Density), case.ParticleNormalsPath)
    end
    logger = SimulationLogger(meta.SaveLocation)
    resolve_output_variables!(meta)
    InitializeLogger(logger, SimConstants, meta, SimKernel, SimViscosity, SimDensityDiffusion,
                     SimGeometry, particles)
    LogStep(logger, meta, meta.HourGlass)
    output = SetupVTKOutput(meta, particles, SimKernel, D)
    output.save_particles(1)
    output.save_particles(2, meta.TotalTime)
    output.close_files()
    LogFinal(logger, meta.HourGlass)
    close(logger.LoggerIo)
    return nothing
end

@setup_workload begin
    dir = mktempdir()
    gpu = CUDA.functional()
    # Lanes per particle are a compile time parameter of the gather kernels and
    # the automatic choice (0) depends on the particle count and the device, so
    # every lane count of the preference is compiled.
    lane_counts = gpu ? [_PRECOMPILE_GPU_LANES; 0] : [0]
    position_modes = _PRECOMPILE_DOUBLE_POSITION ? (false, true) : (false,)
    @compile_workload begin
        redirect_stdout(devnull) do
            for T in _PRECOMPILE_FLOAT_TYPES, (n, lanes) in enumerate(lane_counts), dpos in position_modes
                # Every run gets fresh meta data and its own save locations.
                for case in _precompile_cases(mkpath(joinpath(dir, "$(T)_$(n)_$(dpos)")), T)
                    if gpu
                        try
                            _precompile_run(case, lanes, dpos)
                        catch err
                            # Never make the package unloadable because of the
                            # GPU (e.g. out of memory); fall back to the host code.
                            @warn "SPHExampleGPU: GPU precompile workload failed, only the host " *
                                  "code is precompiled" exception = (err, catch_backtrace())
                            gpu = false
                        end
                    end
                    gpu || _precompile_host(case)
                end
            end
        end
    end
end
