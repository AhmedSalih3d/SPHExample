# DualSPHysics 5.4 examples/main/17_WaveRunup (CaseWaveRunup), drawn and run
# with SPHExampleGPU. The measured CIEMito piston signal generates waves that
# run up a slope covered by 98 fixed armour blocks. This is the reference's
# three-dimensional, transversely periodic case: x/y/z remain x/y/z.
# Including or running this file starts the simulation. Requires an NVIDIA GPU.
#
#     julia --project=gpu_version -t 2,0 gpu_version/example/WaveRunup3d.jl
using SPHExampleGPU
using StaticArrays: SVector
using LinearAlgebra: cross, dot, norm

let
    Dimensions = 3
    FloatType  = Float32

    dx                = 0.008
    still_water_level = 0.25
    piston_face       = 0.0
    piston_top        = 0.55
    bed_start         = -0.5
    bed_end           = 7.0
    simulation_time   = 20.0
    data_dir          = joinpath(@__DIR__, "data", "wave_runup")

    # GenCase clips the 0.37 m wide case to 46 rows at y = 0:0.008:0.36.
    # Periodic faces are half a spacing outside those rows: the period is
    # 0.368 m, rather than the nominal geometry width of 0.37 m.
    transverse_rows = 0:(floor(Int, 0.37 / dx) - 1)
    y_min = -dx / 2
    y_max = (last(transverse_rows) + 0.5) * dx
    flume_width = y_max - y_min

    # The supplied binary STLs consist of one convex wedge (8 triangles) and
    # 98 separate convex blocks (12 triangles each). Keep their actual faces,
    # including each block's orientation. The reference rotates by -90 degrees
    # about z, then translates by (5.95, 0.37, 0) m.
    function read_solids(path, triangles_per_solid)
        open(path, "r") do io
            skip(io, 80)
            count = Int(ltoh(read(io, UInt32)))
            filesize(path) == 84 + 50count && count % triangles_per_solid == 0 ||
                throw(ArgumentError("unexpected CaseWaveRunup STL layout: $path"))
            solids = PolygonSurface[]
            for _ in 1:(count ÷ triangles_per_solid)
                points = SVector{3, Float64}[]
                for _ in 1:triangles_per_solid
                    skip(io, 12)  # stored normal; recompute from the vertices
                    for _ in 1:3
                        x, y, z = ntuple(_ -> Float64(reinterpret(Float32, ltoh(read(io, UInt32)))), 3)
                        push!(points, SVector(5.95 + y, 0.37 - x, z))
                    end
                    skip(io, 2)
                end
                push!(solids, PolygonSurface(points, [(i, i + 1, i + 2) for i in 1:3:length(points)]))
            end
            return solids
        end
    end
    slope  = only(read_solids(joinpath(data_dir, "Slope.stl"), 8))
    blocks = read_solids(joinpath(data_dir, "Blocks_3D_scaled.stl"), 12)

    # Rasterize each convex solid only inside its own bounding box. All groups
    # use the same origin-anchored lattice. Integer indices make shared faces
    # and overlapping blocks own each particle once, and remove water inside
    # the armour without a million-particle-by-98-block search.
    function solid_indices(solids)
        occupied = Set{NTuple{3, Int}}()
        for solid in solids
            center = sum(solid.points) / length(solid.points)
            planes = map(solid.faces) do face
                a, b, c = solid.points[face]
                normal = cross(b - a, c - a)
                normal /= norm(normal)
                dot(normal, center - a) > 0 && (normal = -normal)
                (normal, dot(normal, a))
            end
            lower = reduce((a, b) -> min.(a, b), solid.points)
            upper = reduce((a, b) -> max.(a, b), solid.points)
            ranges = ntuple(d -> ceil(Int, (lower[d] - 1e-7) / dx):floor(Int, (upper[d] + 1e-7) / dx), 3)
            for iz in ranges[3], iy in intersect(ranges[2], transverse_rows), ix in ranges[1]
                point = SVector(ix * dx, iy * dx, iz * dx)
                all(((normal, distance),) -> dot(normal, point) <= distance + 1e-7, planes) &&
                    push!(occupied, (ix, iy, iz))
            end
        end
        return occupied
    end
    slope_indices = solid_indices([slope])
    block_indices = setdiff(solid_indices(blocks), slope_indices)
    boundary_indices = union(slope_indices, block_indices)
    positions(indices) = [SVector(ix * dx, iy * dx, iz * dx) for (ix, iy, iz) in sort!(collect(indices))]
    lattice_index(point) = Tuple(round.(Int, point ./ dx))

    # Six bed and piston layers cover the reference's 2h = 5.20dx support.
    # The piston slides one spacing above the bed. The fill surface is snapped
    # down before excluding it, as GenCase's fillbox does: water particles reach
    # z = 0.24 m and their sampled column is 0.232 m high at this resolution.
    thickness  = 5dx
    tank       = prism(rectangle((bed_start, y_min), bed_end - bed_start, flume_width), -thickness, 0.0)
    piston     = prism(rectangle((piston_face - thickness, y_min), thickness, flume_width), dx, floor(piston_top / dx) * dx)
    fill_top   = floor(still_water_level / dx) * dx
    slope_end  = maximum(first, slope.points)
    water_box  = prism(rectangle((piston_face, y_min), slope_end - piston_face, flume_width), 0.0, fill_top)

    # Draw the initial water surface up to the shoreline, rather than exporting
    # the rectangular fill box over the dry part of the beach.
    slope_toe = minimum(first, slope.points)
    slope_bottom, slope_top = extrema(last, slope.points)
    shoreline = slope_toe + (fill_top - slope_bottom) * (slope_end - slope_toe) / (slope_top - slope_bottom)
    water_section = [(piston_face, 0.0), (slope_toe, 0.0), (shoreline, fill_top), (piston_face, fill_top)]
    water = PolygonSurface([SVector(x, y, z) for y in (y_min, y_max) for (x, z) in water_section],
        [[1, 2, 3, 4], [8, 7, 6, 5], [[i, i + 4, mod1(i + 1, 4) + 4, mod1(i + 1, 4)] for i in 1:4]...])

    regions = [
        ParticleRegion("Tank",   tank,   Fixed),
        ParticleRegion("Piston", piston, Moving),
        ParticleRegion("Water",  water_box, Fluid),
    ]
    TankParticles, PistonParticles, WaterParticles = sample_particles(regions, dx)
    filter!(point -> lattice_index(point) ∉ boundary_indices, WaterParticles.positions)
    SlopeParticles = positions(slope_indices)
    BlockParticles = positions(block_indices)

    water_bottom, water_level = extrema(last, WaterParticles.positions)
    SimConstants = SimulationConstants{FloatType}(Dimensions;
        dx = dx,
        c₀ = 20 * sqrt(9.81 * (water_level - water_bottom)),
    )
    density(points) = hydrostatic_density(points, SimConstants; water_level)

    Tank = SPHGeometry{Dimensions, FloatType}(TankParticles.positions;
        Density     = density(TankParticles.positions),
        GroupMarker = 1,
        Type        = Fixed,
    )

    Slope = SPHGeometry{Dimensions, FloatType}(SlopeParticles;
        Density     = density(SlopeParticles),
        GroupMarker = 2,
        Type        = Fixed,
    )

    Blocks = SPHGeometry{Dimensions, FloatType}(BlockParticles;
        Density     = density(BlockParticles),
        GroupMarker = 3,
        Type        = Fixed,
    )

    Piston = SPHGeometry{Dimensions, FloatType}(PistonParticles.positions;
        Density     = density(PistonParticles.positions),
        GroupMarker = 4,
        Type        = Moving,
    )

    Water = SPHGeometry{Dimensions, FloatType}(WaterParticles.positions;
        Density     = density(WaterParticles.positions),
        GroupMarker = 5,
        Type        = Fluid,
    )

    SimulationGeometry = [Tank, Slope, Blocks, Piston, Water]

    # The measured displacement is piecewise linear, not a regular-wave fit.
    # The first two rows are identical (t = 0, x = 0); remove that duplicate.
    # Motion ends at 15 s as in mvpredef; the piston then stays at x = 0.
    signal = unique([Tuple(parse.(Float64, split(line))) for line in eachline(joinpath(data_dir, "Mov_piston.dat"))])
    filter!(row -> first(row) <= 15.0, signal)
    PistonMotion = PrescribedMotion(Piston.GroupMarker;
        translation = SampledTrajectory(first.(signal), [(row[2], 0.0, 0.0) for row in signal]),
        center      = (0.0, 0.0, 0.0),
    )

    # The four experimental gauges measure elevation about the still surface;
    # subtract the initial water-column height for comparison. The beach's
    # binned free surface also shows the advancing and retreating swash front.
    # It is a highest-particle diagnostic, distinct from the reference's seven
    # kernel-mass-threshold gauges along inclined lines through the armour.
    WaveGauges = MeasurementConfig(
        water_column_probes = [WaterColumnProbe("WG$i", (x, 0.18, 0.0); radius = dx)
                               for (i, x) in enumerate((3.10, 3.20, 3.34, 3.63))],
        free_surface = FreeSurfaceDomain((5.72, y_min, 0.0), (6.81, y_max, 0.62), dx),
    )

    SimMetaData = SimulationMetaData{
        Dimensions,
        FloatType,
        NoShifting,
        NoKernelOutput,
        NoMDBC,
        StoreLog,
    }(
        SimulationName      = "WaveRunup3D",
        SaveLocation        = joinpath(@__DIR__, "..", "output", "wave_runup_3d"),
        SimulationTime      = simulation_time,
        OutputTimes         = 0.02,
        VisualizeInParaview = true,
        ExportSingleVTKHDF  = true,
        ExportGridCells     = true,
        OpenLogFile         = true,
        Periodicity         = PeriodicBoundary{Dimensions}(2; min = y_min, max = y_max),
        # Resolve slow particle motion several metres from the origin.
        GPUDoublePosition   = true,
    )

    # h = 1.5 √3 dx, the reference's coefh = 1.5 in 3D. Match its artificial
    # viscosity (zero wall-viscosity factor) and Fourtakas inner diffusion.
    SimKernel           = SPHKernelInstance{Dimensions, FloatType}(WendlandC2(); h = FloatType(1.5 * sqrt(3) * dx))
    SimViscosity        = ArtificialViscosity(boundary_factor = 0)
    SimDensityDiffusion = FourtakasDensityDiffusion()

    mkpath(SimMetaData.SaveLocation)

    SimLogger    = SimulationLogger(SimMetaData.SaveLocation)
    SimParticles = AllocateDataStructures(SimulationGeometry, SimMetaData)

    CleanUpSimulationFolder(SimMetaData.SaveLocation)
    SavePolygonVTKHDF(joinpath(SimMetaData.SaveLocation, "WaveRunup3D_Geometry.vtkhdf"), (; tank, slope, blocks, piston, water))

    RunSimulation(
        SimGeometry         = SimulationGeometry,
        SimMetaData         = SimMetaData,
        SimConstants        = SimConstants,
        SimLogger           = SimLogger,
        SimParticles        = SimParticles,
        SimKernel           = SimKernel,
        SimViscosity        = SimViscosity,
        SimDensityDiffusion = SimDensityDiffusion,
        SimTimeStepping     = SymplecticTimeStepping(),
        SimMotion           = PistonMotion,
        SimMeasurements     = WaveGauges,
    )
end
