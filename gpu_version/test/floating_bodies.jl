# GPU tests of the floating rigid bodies (`Type = Floating`, see `GPUFloating`).
# Included by runtests.jl inside the "SPHExampleGPU" test set.
using CSV

"""
    run_floating_case(; FloatType, relative_weight, pause, water, simtime, double)

Run a small 2D case on the GPU: a conforming disc of radius 0.1 m centred at
`(0.5, 0.4)` as a floating body in a closed 1 m tank, with water up to 0.8 m
when `water` is set. Returns the final particles (sorted by ID) and the rows of
the floating body log, and the particles as loaded (sorted by ID).
"""
function run_floating_case(; FloatType = Float64, relative_weight, pause = 0.0, water = true,
                           simtime, double = false)
    T  = FloatType
    dx = 0.02
    dir = mktempdir()
    generator_constants = SimulationConstants{Float64}(; dx, c₀ = 20.0)
    regions = [ParticleRegion("Body", circle((0.5, 0.4), 0.1), Floating; sampling = :conforming),
               ParticleRegion("Tank", outline(rectangle((0, 0), 1, 1); thickness = 3dx), Fixed)]
    water && push!(regions, ParticleRegion("Fluid", rectangle((0, 0), 1, 0.8), Fluid))
    sampled = sample_particles(regions, dx)
    level = water ? maximum(last, sampled[end].positions) : 0.0
    next_id = 0
    for region in sampled
        density = hydrostatic_density(region.positions, generator_constants; water_level = level)
        next_id = write_particle_csv(joinpath(dir, region.name * ".csv"), region.positions;
                                     density, first_id = next_id)
    end

    geometry = [SPHGeometry{2, T}(CSVFile = joinpath(dir, "Tank.csv"), GroupMarker = 1, Type = Fixed),
                SPHGeometry{2, T}(CSVFile = joinpath(dir, "Body.csv"), GroupMarker = 3, Type = Floating,
                                  Floating = FloatingDetails{T}(RelativeWeight = relative_weight,
                                                                PauseTime = pause))]
    water && push!(geometry, SPHGeometry{2, T}(CSVFile = joinpath(dir, "Fluid.csv"), GroupMarker = 2,
                                               Type = Fluid))
    meta = SimulationMetaData{2, T, NoShifting, NoKernelOutput, NoMDBC, NoLog}(
        SimulationName = "Floating", SaveLocation = dir, SimulationTime = simtime,
        OutputTimes = simtime / 5, GPUDoublePosition = double)
    particles = AllocateDataStructures(geometry, meta)
    initial   = deepcopy(particles)
    RunSimulation(SimGeometry = geometry, SimMetaData = meta,
                  SimConstants = SimulationConstants{T}(; dx = T(dx), c₀ = T(20)),
                  SimKernel = SPHKernelInstance{2, T}(WendlandC2(); dx = T(dx), k = T(sqrt(2))),
                  SimLogger = SimulationLogger(dir; to_console = false), SimParticles = particles,
                  SimViscosity = Laminar(), SimDensityDiffusion = LinearDensityDiffusion(),
                  SimTimeStepping = SymplecticTimeStepping())
    rows = CSV.File(joinpath(dir, "Floating_Floating.csv"))
    return particles[sortperm(particles.ID)], rows, initial[sortperm(initial.ID)]
end

body_distances(p) = (x = p.Position[p.Type .== Floating];
                     [norm(x[i] - x[j]) for i in eachindex(x) for j in (i + 1):lastindex(x)])

@testset "floating bodies" begin
    @testset "free fall in air is exact and rigid" begin
        g, pause = 9.81, 0.02
        p, rows, initial = run_floating_case(; relative_weight = 1.2, pause, water = false, simtime = 0.1)
        y0 = rows[1][Symbol("Center:1")]
        @test y0 ≈ 0.4 atol = 1e-12
        held = [r for r in rows if r.Time <= pause]
        @test all(r -> r[Symbol("Center:1")] == y0 && r[Symbol("Velocity:1")] == 0, held)

        last_row = rows[end]
        v = last_row[Symbol("Velocity:1")]
        released = last_row.Time + v / g           # the start of the first step after the pause
        @test pause <= released < pause + 1e-3
        # the symplectic update is exact for a constant acceleration
        @test last_row[Symbol("Center:1")] ≈ y0 - v^2 / (2g) rtol = 1e-10
        @test last_row[Symbol("Center:0")] == rows[1][Symbol("Center:0")]
        @test last_row[Symbol("Velocity:0")] == 0 && last_row.Angle == 0
        @test maximum(abs.(body_distances(p) .- body_distances(initial))) < 1e-12
        body = p.Type .== Floating
        @test all(x -> x ≈ SVector(0.0, v), p.Velocity[body])
        @test sum(p.Position[body]) / count(body) ≈
              SVector(last_row[Symbol("Center:0")], last_row[Symbol("Center:1")]) atol = 1e-12
    end

    @testset "buoyancy: a neutral body stays, a heavy one sinks (Float32, double positions)" begin
        sink(rows) = rows[1][Symbol("Center:1")] - rows[end][Symbol("Center:1")]
        _, neutral, _ = run_floating_case(FloatType = Float32, relative_weight = 1.0, pause = 0.1,
                                          simtime = 0.3, double = true)
        heavy_particles, heavy, _ = run_floating_case(FloatType = Float32, relative_weight = 2.0,
                                                      pause = 0.1, simtime = 0.3, double = true)
        # a cylinder of twice the density of water sinks with about g/3 (added mass)
        @test 0.03 < sink(heavy) < 0.1
        @test abs(sink(neutral)) < 0.2 * sink(heavy)
        @test all(isfinite, heavy_particles.Density)
        @test abs(heavy[end].Angle) < 0.05
    end

    @testset "floating bodies need the symplectic scheme and the details" begin
        @test_throws ErrorException FloatingArrays(
            [SPHGeometry{2, Float64}(CSVFile = "", GroupMarker = 1, Type = Floating)],
            StructArray((Position = [SVector(0.0, 0.0)], GroupMarker = UInt[1], Type = [Floating])),
            SimulationConstants{Float64}())
    end
end
