# Precompilation workload (PrecompileTools.jl).
#
# Runs a few tiny simulations while the package is precompiled, so that the
# host code (CSV input, polygon/particle generation, logging, VTKHDF output,
# measurements, the time loop) and, on a machine with a working GPU, the CUDA
# kernels are compiled once and stored in the package image instead of being
# compiled at the start of every Julia session. The cases mirror the
# configurations of the scripts in `example/`.
#
# Only the combinations of types exercised here are cached. Other float types,
# kernels or viscosity / density diffusion models still work, but are compiled
# on first use. Two preferences control the workload (set them with
# `Preferences.set_preferences!(SPHExampleGPU, ...)` and restart Julia):
#   - `precompile_float_types`: float types to precompile, default `["Float32"]`
#   - `precompile_gpu_lanes`:   lanes per particle compiled for the gather
#                               kernels, default `[1, 4]`; automatic selection
#                               is also compiled
#   - `precompile_double_position`: also run every case with
#                               `GPUDoublePosition = true` (the cell relative
#                               kernel variants), default `false`; the cases
#                               whose examples use double positions always cache
#                               this mode
# `precompile_workload = false` (PrecompileTools) disables the workload.

using PrecompileTools: @setup_workload, @compile_workload
using Preferences: @load_preference
using StaticArrays: SVector
using StructArrays: StructArray
using Meshes: Multi, SutherlandHodgmanClipping, clip
using Logging: ConsoleLogger, with_logger

const _PRECOMPILE_FLOAT_TYPES = let
    known = Dict("Float32" => Float32, "Float64" => Float64)
    names = @load_preference("precompile_float_types", ["Float32"])
    Type[known[n] for n in names if haskey(known, n)]
end

const _PRECOMPILE_GPU_LANES = let
    lanes = @load_preference("precompile_gpu_lanes", [1, 4])
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

# Precompile the polygon drawing, lattice/conforming particle sampling,
# hydrostatic density, CSV/VTKHDF geometry exports, and host StructArray
# allocations used by the example and generator scripts.
function _precompile_geometry_tools(dir::String, ::Type{T}) where {T}
    dx = 0.2
    tank = polyline([(0, 1.0), (0, 0), (1.0, 0), (1.0, 1.0)]; thickness = dx, side = :right)
    lid = line((-dx, 1.0), (1.0 + dx, 1.0); thickness = dx, side = :left)
    box = square((0.2, 0.2), dx; angle = π / 4, centered = true)
    cyl = circle((0.6, 0.6), dx; segments = 16)
    tri = triangle((0.2, 0.0), (0.6, 0.0), (0.4, dx))
    baffle = polyline(arc((0.5, 0.8), dx, π, 2π); thickness = dx / 2)
    water_outline = rectangle((0, 0), 1.0, 0.8)
    water = polygon(water_outline; holes = [box])
    closed_tank = outline(water_outline; thickness = dx, side = :outward)
    shrunk = offset_polygon(water_outline, -0.01)

    clipping = SutherlandHodgmanClipping()
    for obs in (cyl, tri)
        clipped = clip(obs, water_outline, clipping)
        clipped !== nothing && polygon(water_outline; holes = [offset_polygon(clipped, -1e-6)])
    end

    # Exercise the distinct region-vector element types used across the examples.
    sample_particles([ParticleRegion("Bound", tank, Fixed), ParticleRegion("Fluid", water, Fluid)], T(dx))
    sample_particles([ParticleRegion("Bound", tank, Fixed), ParticleRegion("Fluid", water, Fluid)], dx)
    sample_particles([ParticleRegion("Bound", Multi([tank, box]), Fixed), ParticleRegion("Fluid", water, Fluid)], T(dx))
    sample_particles([ParticleRegion("Bound", Multi([tank, box]), Fixed), ParticleRegion("Fluid", water, Fluid)], dx)
    sample_particles([ParticleRegion("Cyl", cyl, Floating; sampling = :conforming),
                      ParticleRegion("Bound", tank, Fixed),
                      ParticleRegion("Fluid", water, Fluid)], T(dx))
    sample_particles([ParticleRegion("Cyl", cyl, Floating; sampling = :conforming),
                      ParticleRegion("Bound", tank, Fixed),
                      ParticleRegion("Fluid", water, Fluid)], dx)

    regions_2d = [
        ParticleRegion("Cyl", cyl, Floating; sampling = :conforming),
        ParticleRegion("Body", box, Moving; sampling = :conforming),
        ParticleRegion("Bound", Multi([tank, tri, baffle]), Fixed),
        ParticleRegion("Lid", lid, Moving),
        ParticleRegion("Fluid", water, Fluid),
    ]
    sampled_2d = sample_particles(regions_2d, dx)
    sample_particles(regions_2d, T(dx))
    sample_particles([
        ParticleRegion("Curved", Multi([cyl, baffle]), Fixed; sampling = :conforming),
        ParticleRegion("Body", box, Moving; sampling = :conforming),
        ParticleRegion("Bound", Multi([tank, tri]), Fixed),
        ParticleRegion("Fluid", water, Fluid),
    ], dx)

    c_f64 = SimulationConstants{Float64}(; dx)
    c_t   = SimulationConstants{T}(; dx)
    fluid_pos = sampled_2d[5].positions
    maximum(last, fluid_pos)
    hd_64 = hydrostatic_density(fluid_pos, c_f64)
    hd_t  = hydrostatic_density(fluid_pos, c_t; water_level = 0.8)
    hydrostatic_density(fluid_pos, c_t)
    EquationOfStateGamma7.(hd_64, c_f64.c₀, c_f64.ρ₀)

    write_particle_csv(joinpath(dir, "geom_particles.csv"), fluid_pos; density = hd_64, first_id = 0)
    SavePolygonVTKHDF(joinpath(dir, "geom2d.vtkhdf"), (; tank, water, closed_tank, curved = Multi([cyl, baffle])))
    SavePolygonMotionSequence(
        joinpath(dir, "motion2d_t.vtkhdf"),
        (; tank = closed_tank, square = box);
        motions = (; square = MotionDetails{2, T}(Velocity = one(T), StartTime = zero(T),
                                                  Duration = one(T), Direction = SVector{2, T}(1, 0))),
        times = 0.0:0.5:1.0,
    )
    SavePolygonMotionSequence(
        joinpath(dir, "motion2d_f64.vtkhdf"),
        (; tank = closed_tank, square = box);
        motions = (; square = MotionDetails{2, Float64}(Velocity = 1.0, StartTime = 0.0,
                                                        Duration = 1.0, Direction = SVector{2, Float64}(1, 0))),
        times = 0.0:0.5:1.0,
    )

    p_wall   = prism(closed_tank, -dx, 0.6)
    p_floor  = prism(shrunk, -dx, 0.0)
    p_pillar = prism(cyl, 0.0, 0.6)
    p_wedge  = prism(tri, 0.0, 0.2)
    p_water  = prism(water_outline, 0.0, 0.6)
    sampled_3d = sample_particles([
        ParticleRegion("Bound", (p_wall, p_floor, p_pillar, p_wedge), Fixed),
        ParticleRegion("Fluid", p_water, Fluid),
    ], dx)
    hd_3d = hydrostatic_density(sampled_3d[2].positions, c_f64)
    write_particle_csv(joinpath(dir, "geom3d_particles.csv"), sampled_3d[2].positions; density = hd_3d, first_id = 0)
    SavePolygonVTKHDF(joinpath(dir, "geom3d.vtkhdf"),
                      (; boundary = (p_wall, p_floor, p_pillar, p_wedge), water = p_water))
    pos3d = to_3d(fluid_pos)
    append!(SVector{3, Float64}[], pos3d)
    append!(SVector{3, Float64}[], sampled_3d[2].positions)
    append!(Float64[], hd_64)
    append!(Int8[], fill(Int8(1), length(pos3d)))
    append!(Int[], fill(1, length(pos3d)))
    SaveVTKHDF(joinpath(dir, "particles.vtkhdf"), pos3d,
               ["Density", "Pressure", "Type", "GroupMarker"],
               hd_64, hd_64, fill(Int8(1), length(pos3d)), fill(1, length(pos3d)))

    pos2 = sampled_2d[3].positions
    pos3 = sampled_3d[1].positions
    g2_a = SPHGeometry{2, T}(pos2; Density = c_t.ρ₀, GroupMarker = 1, Type = Fixed)
    g2_b = SPHGeometry{2, T}(fluid_pos; Density = hd_t, GroupMarker = 2, Type = Fluid)
    g2_c = SPHGeometry{2, T}(Particles = particle_struct_array(pos2, c_t.ρ₀;
                                 GhostPoints = pos2, GhostNormals = pos2),
                             GroupMarker = 1, Type = Fixed)
    g2_d = SPHGeometry{2, T}(Particles = particle_struct_array(pos2, c_t.ρ₀),
                             GroupMarker = 2, Type = Moving,
                             Motion = MotionDetails{2, T}(Velocity = one(T), StartTime = zero(T),
                                                          Duration = one(T), Direction = SVector{2, T}(1, 0),
                                                          MoveParticles = false))
    g2_e = SPHGeometry{2, T}(Particles = StructArray((Position = fluid_pos, Density = hd_t)),
                             GroupMarker = 3, Type = Floating,
                             Floating = FloatingDetails{T}(RelativeWeight = T(1.2), PauseTime = one(T)))
    g3_a = SPHGeometry{3, T}(pos3; Density = c_t.ρ₀, GroupMarker = 1, Type = Fixed)

    for dpos in (false, true), cells in (false, true), ghosts in (false, true)
        TP = dpos ? Float64 : T
        AllocateDataStructures([g2_a, g2_b]; position_type = TP, cells = cells,
                               kernel_output = false, ghost_data = ghosts, compact_metadata = true)
        AllocateDataStructures([g2_c, g2_d, g2_e]; position_type = TP, cells = cells,
                               kernel_output = false, ghost_data = ghosts, compact_metadata = true)
        AllocateDataStructures([g3_a]; position_type = TP, cells = cells,
                               kernel_output = false, ghost_data = ghosts, compact_metadata = true)
    end
    return nothing
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

    # example/StillWedgeMiddleSquareMDBC.jl (2D SimpleMDBC with CSV/normals path)
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

    # example/StillWedgeMDBC.jl (2D DBC with Float64 positions, both reference
    # and radial interactions, with and without cached interaction fields)
    dbc_2d = merge(mdbc_2d, (
        SimMetaData = SimulationMetaData{2, T, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(;
            SimulationName = "DBC2D", SaveLocation = save("DBC2D"),
            OutputTimes = time.SimulationTime, ExportGridCells = true,
            GPUDoublePosition = true, GPUOutputQueueBytes = 8 * 1024^2, time...),
        ParticleNormalsPath = nothing,
    ))
    dbc_2d_radial = merge(mdbc_2d, (
        SimMetaData = SimulationMetaData{2, T, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(;
            SimulationName = "DBC2DRadial", SaveLocation = save("DBC2DRadial"),
            OutputTimes = time.SimulationTime, ExportGridCells = true,
            GPUDoublePosition = true, GPURadialInteractions = true,
            GPUOutputQueueBytes = 8 * 1024^2, time...),
        ParticleNormalsPath = g2.ghost,
    ))
    dbc_2d_radial_cached = merge(mdbc_2d, (
        SimMetaData = SimulationMetaData{2, T, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(;
            SimulationName = "DBC2DRadialCached", SaveLocation = save("DBC2DRadialCached"),
            OutputTimes = time.SimulationTime, ExportGridCells = true,
            GPUDoublePosition = true, GPURadialInteractions = true,
            GPUCacheInteractionFields = true, GPUOutputQueueBytes = 8 * 1024^2, time...),
        ParticleNormalsPath = g2.ghost,
    ))

    # example/Dambreak2dMDBC.jl (2D DBC with ZeroViscosity and in-memory ghost fields)
    bound_pos = mdbc_2d.SimGeometry[1].Particles.Position
    fluid_pos = mdbc_2d.SimGeometry[2].Particles.Position
    dbc_2d_zero_visc = (
        SimGeometry  = [SPHGeometry{2, T}(Particles = particle_struct_array(bound_pos, c2mdbc.ρ₀;
                                              GhostPoints = bound_pos, GhostNormals = bound_pos),
                                          GroupMarker = 1, Type = Fixed),
                        SPHGeometry{2, T}(Particles = StructArray((Position = fluid_pos,
                                              Density = fill(c2mdbc.ρ₀, length(fluid_pos)))),
                                          GroupMarker = 2, Type = Fluid)],
        SimMetaData  = SimulationMetaData{2, T, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(;
                           SimulationName = "DBC2DZeroVisc", SaveLocation = save("DBC2DZeroVisc"),
                           OutputTimes = [time.SimulationTime], ExportGridCells = true, time...),
        SimConstants = c2mdbc,
        SimKernel    = SPHKernelInstance{2, T}(WendlandC2(); dx = c2mdbc.dx),
        SimViscosity = ZeroViscosity(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        ParticleNormalsPath = nothing,
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

    # example/LidDrivenCavity2d.jl (PlanarShifting + SimpleMDBC + Laminar + GPUBoundaryForces = false)
    solid_pos = moving_2d.SimGeometry[3].Particles.Position
    cavity_2d = (
        SimGeometry  = [SPHGeometry{2, T}(Particles = particle_struct_array(bound_pos, c2move.ρ₀;
                                              GhostPoints = bound_pos, GhostNormals = bound_pos),
                                          GroupMarker = 1, Type = Fixed),
                        SPHGeometry{2, T}(Particles = particle_struct_array(solid_pos, c2move.ρ₀;
                                              GhostPoints = solid_pos, GhostNormals = solid_pos),
                                          GroupMarker = 2, Type = Moving,
                                          Motion = MotionDetails{2, T}(Velocity = one(T), StartTime = zero(T),
                                                                       Duration = one(T),
                                                                       Direction = SVector{2, T}(1, 0),
                                                                       MoveParticles = false)),
                        SPHGeometry{2, T}(Particles = particle_struct_array(fluid_pos, c2move.ρ₀),
                                          GroupMarker = 3, Type = Fluid)],
        SimMetaData  = SimulationMetaData{2, T, PlanarShifting, NoKernelOutput, SimpleMDBC, StoreLog}(;
                           SimulationName = "Cavity2D", SaveLocation = save("Cavity2D"),
                           OutputTimes = time.SimulationTime, GPUBoundaryForces = false, time...),
        SimConstants = c2move,
        SimKernel    = SPHKernelInstance{2, T}(WendlandC2(); dx = c2move.dx, k = T(sqrt(2))),
        SimViscosity = Laminar(),
        SimDensityDiffusion = ZeroGravityLinearDensityDiffusion(),
        ParticleNormalsPath = nothing,
    )

    # example/MultipleFloatingCylinders2d.jl & example/FloatingCylinder2d.jl
    floating_geom = [
        SPHGeometry{2, T}(CSVFile = g2.bound, GroupMarker = 1, Type = Fixed),
        SPHGeometry{2, T}(CSVFile = g2.fluid, GroupMarker = 2, Type = Fluid),
        SPHGeometry{2, T}(CSVFile = g2.solid, GroupMarker = 3, Type = Floating,
                          Floating = FloatingDetails{T}(RelativeWeight = 1.2, PauseTime = 1.0)),
    ]
    floating_2d_symplectic = (
        SimGeometry  = floating_geom,
        SimMetaData  = SimulationMetaData{2, T, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(;
                           SimulationName = "Floating2DSymp", SaveLocation = save("Floating2DSymp"),
                           OutputTimes = time.SimulationTime, ExportGridCells = true,
                           GPUDoublePosition = true, time...),
        SimConstants = c2mdbc,
        SimKernel    = SPHKernelInstance{2, T}(WendlandC2(); dx = c2mdbc.dx),
        SimViscosity = LaminarSPS(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        SimTimeStepping = SymplecticTimeStepping(),
        ParticleNormalsPath = nothing,
    )
    floating_2d_single = (
        SimGeometry  = floating_geom,
        SimMetaData  = SimulationMetaData{2, T, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(;
                           SimulationName = "Floating2DSingle", SaveLocation = save("Floating2DSingle"),
                           OutputTimes = time.SimulationTime, ExportGridCells = true,
                           GPUDoublePosition = true, GPUCacheInteractionFields = true, time...),
        SimConstants = c2mdbc,
        SimKernel    = SPHKernelInstance{2, T}(WendlandC2(); dx = c2mdbc.dx),
        SimViscosity = LaminarSPS(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        SimTimeStepping = SingleNeighborTimeStepping(),
        ParticleNormalsPath = nothing,
    )

    # example/DucklingMDBC.jl & example/DucklingMeasurements.jl
    c3mdbc = SimulationConstants{T}(dx = dx, c₀ = 23.43842998154953, δᵩ = 0.1, CFL = 0.2, α = 0.02,
                                    m₀ = 1000 * dx^3)
    mdbc_3d = (
        SimGeometry  = [SPHGeometry{3, T}(CSVFile = g3.bound, GroupMarker = 1, Type = Fixed),
                        SPHGeometry{3, T}(CSVFile = g3.fluid, GroupMarker = 2, Type = Fluid)],
        SimMetaData  = SimulationMetaData{3, T, NoShifting, NoKernelOutput, SimpleMDBC, StoreLog}(;
                           SimulationName = "MDBC3D", SaveLocation = save("MDBC3D"),
                           OutputTimes = time.SimulationTime, ExportGridCells = true, time...),
        SimConstants = c3mdbc,
        SimKernel    = SPHKernelInstance{3, T}(WendlandC2(); dx = c3mdbc.dx, k = T(1.5)),
        SimViscosity = ArtificialViscosity(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        ParticleNormalsPath = g3.ghost,
    )
    measurements_3d = MeasurementConfig(
        pressure_probes = [MeasurementProbe("p1", (2 * dx, 2 * dx, dx))],
        velocity_probes = [MeasurementProbe("v1", (2 * dx, 2 * dx, dx))],
        water_column_probes = [WaterColumnProbe("c1", (2 * dx, 2 * dx, 0.0); radius = 2 * dx)],
        free_surface = FreeSurfaceDomain((0.0, 0.0, 0.0), (4 * dx, 4 * dx, 0.0), 2 * dx),
        vertical_axis = 3,
        sample_every = 1,
    )
    mdbc_3d_meas = (
        SimGeometry  = mdbc_3d.SimGeometry,
        SimMetaData  = SimulationMetaData{3, T, NoShifting, NoKernelOutput, SimpleMDBC, StoreLog}(;
                           SimulationName = "MDBC3DMeas", SaveLocation = save("MDBC3DMeas"),
                           OutputTimes = time.SimulationTime, ExportGridCells = false, time...),
        SimConstants = c3mdbc,
        SimKernel    = mdbc_3d.SimKernel,
        SimViscosity = ArtificialViscosity(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        ParticleNormalsPath = g3.ghost,
        SimMeasurements = measurements_3d,
    )

    # example/Dambreak3d.jl (3D DBC with GPUDoublePosition = true and GPUCellSubdivision = 2,
    # plus single-precision positions)
    c3dbc = SimulationConstants{T}(dx = dx, c₀ = 33.14, α = 0.1, m₀ = 1000 * dx^3, CFL = 0.2)
    dbc_3d = (
        SimGeometry  = [SPHGeometry{3, T}(CSVFile = g3.bound, GroupMarker = 1, Type = Fixed),
                        SPHGeometry{3, T}(CSVFile = g3.fluid, GroupMarker = 2, Type = Fluid)],
        SimMetaData  = SimulationMetaData{3, T, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(;
                           SimulationName = "DBC3D", SaveLocation = save("DBC3D"),
                           OutputTimes = time.SimulationTime, ExportGridCells = true,
                           GPUCellSubdivision = 2, GPUDoublePosition = true, time...),
        SimConstants = c3dbc,
        SimKernel    = SPHKernelInstance{3, T}(WendlandC2(); h = T(sqrt(3 * dx^2))),
        SimViscosity = ArtificialViscosity(),
        SimDensityDiffusion = LinearDensityDiffusion(),
        ParticleNormalsPath = nothing,
    )
    dbc_3d_single_pos = merge(dbc_3d, (
        SimMetaData = SimulationMetaData{3, T, NoShifting, NoKernelOutput, NoMDBC, StoreLog}(;
            SimulationName = "DBC3DSingle", SaveLocation = save("DBC3DSingle"),
            OutputTimes = time.SimulationTime, ExportGridCells = true,
            GPUCellSubdivision = 2, GPUDoublePosition = false, time...),
    ))

    return (mdbc_2d, dbc_2d, dbc_2d_radial, dbc_2d_radial_cached, dbc_2d_zero_visc,
            moving_2d, cavity_2d, floating_2d_symplectic, floating_2d_single,
            mdbc_3d, mdbc_3d_meas, dbc_3d, dbc_3d_single_pos)
end

# Complete run on the GPU, as done by the example scripts.
function _precompile_run(case, lanes::Int, double_position::Bool)
    meta = case.SimMetaData
    meta.GPULanesPerParticle = lanes
    meta.GPUDoublePosition   = double_position || meta.GPUDoublePosition
    CleanUpSimulationFolder(meta.SaveLocation)
    particles = AllocateDataStructures(case.SimGeometry, meta)
    logger    = SimulationLogger(meta.SaveLocation)
    time_stepping = get(case, :SimTimeStepping, SymplecticTimeStepping())
    measurements  = get(case, :SimMeasurements, nothing)
    if measurements === nothing
        RunSimulation(
            SimGeometry = case.SimGeometry,
            SimMetaData = meta,
            SimConstants = case.SimConstants,
            SimKernel = case.SimKernel,
            SimLogger = logger,
            SimParticles = particles,
            SimViscosity = case.SimViscosity,
            SimDensityDiffusion = case.SimDensityDiffusion,
            SimTimeStepping = time_stepping,
            ParticleNormalsPath = case.ParticleNormalsPath,
        )
    else
        RunSimulation(
            SimGeometry = case.SimGeometry,
            SimMetaData = meta,
            SimConstants = case.SimConstants,
            SimKernel = case.SimKernel,
            SimLogger = logger,
            SimParticles = particles,
            SimViscosity = case.SimViscosity,
            SimDensityDiffusion = case.SimDensityDiffusion,
            SimTimeStepping = time_stepping,
            ParticleNormalsPath = case.ParticleNormalsPath,
            SimMeasurements = measurements,
        )
    end
    # Also compile the multi-block `commit_step = Val(false)` variant of
    # `launch_final_step!` (tiny precompile geometries have n <= 256 so
    # `red.nblocks == 1` only exercises `Val(true)`, whereas examples have n > 256).
    D = typeof(meta).parameters[1]
    T = typeof(meta).parameters[2]
    SMode = typeof(meta).parameters[3]
    BMode = typeof(meta).parameters[5]
    gpu = upload_particles(particles, meta)
    red = ReductionWorkspace{SVector{3, T}}(length(particles))
    cl  = CellListWorkspace{D, position_float_type(meta)}(length(particles);
              reach = meta.GPUCellSubdivision, max_cells = meta.GPUMaxCells,
              fixed_grid = meta.GPUFixedGrid, expansion = meta.GPUGridExpansion,
              deterministic = meta.GPUDeterministicSort)
    sup = GPUSupportArrays{D, T}(gpu, meta)
    state = StepState{T}(; time = meta.TotalTime, iteration = meta.Iteration,
                           dx = one(T) + case.SimKernel.h)
    pos_half, vel_half = SPHCellList.predictor_arrays(sup, gpu)
    launch_final_step!(
        gpu.Position, gpu.Velocity, gpu.Acceleration, gpu.Density,
        sup.dρdtI, sup.ρₙ⁺, pos_half, vel_half, gpu.Type,
        sup.∇Cᵢ, sup.∇◌rᵢ, state, case.SimKernel, case.SimConstants, red,
        Val(SMode === PlanarShifting);
        boundary_mode = BMode(),
        ghost_points = gpu.GhostPoints, ghost_normals = gpu.GhostNormals,
        boundary_slots = gpu.BoundarySlot,
        pos_cells = sup.PosCells, CellID = gpu.CellID, grid = cl.grid_dev,
        commit_step = false, inverse_density = sup.InvDensity,
    )
    launch_commit!(state)
    CUDA.synchronize()
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
    measurements = get(case, :SimMeasurements, nothing)
    output = SetupVTKOutput(meta, particles, SimKernel, D;
                            measurements = measurements, fluid_type = Fluid,
                            constants = SimConstants)
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
            with_logger(ConsoleLogger(devnull)) do
                @warn "precompile"
                @info "precompile" output_dir = dir particles = 1 relative_weights = [1.0]
            end
            if gpu
                for D in (2, 3), E in (Float32, Float64)
                    CUDA.zeros(SVector{D, E}, 1)
                    CUDA.zeros(E, 1)
                end
                CUDA.zeros(Int32, 1)
                CUDA.ones(Bool, 1)
            end
            for T in _PRECOMPILE_FLOAT_TYPES
                _precompile_geometry_tools(mkpath(joinpath(dir, "geom_$(T)")), T)
                for (n, lanes) in enumerate(lane_counts), dpos in position_modes
                    # Every run gets fresh meta data and its own save locations.
                    for case in _precompile_cases(mkpath(joinpath(dir, "$(T)_$(n)_$(dpos)")), T)
                        # Multi-lane gather kernels are only chosen for small 2D cases.
                        lanes > 1 && !(lanes in _PRECOMPILE_GPU_LANES && length(_PRECOMPILE_GPU_LANES) > 2) &&
                            typeof(case.SimMetaData).parameters[1] == 3 && continue
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
end
