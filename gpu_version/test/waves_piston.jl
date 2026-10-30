module WavesPistonTests

using Test, SPHExampleGPU, CUDA, StaticArrays, CSV, HDF5
include(joinpath(@__DIR__, "..", "example", "WavesPiston", "WavesPistonSetup.jl"))
const DZ = SPHExampleGPU.DampingZones

@testset "planar absorption follows completed-step velocity reduction" begin
    zone = DampingZone((11, 0), (15.5, 0))
    velocity = SVector(2.0f0, -3.0f0)
    for x in (10.0, 11.0, 12.0, 13.25, 15.5, 16.5, 16.51)
        factor = 11 < x <= 16.5 ? max(0, 1 - 0.1 * 10 * min((x - 11) / 4.5, 1)^2) : 1.0
        @test DZ.damped_velocity((zone,), SVector(x, 0.), velocity, Fluid, 0.1) ≈ velocity * factor
        for kind in (Fixed, Moving, Floating, Frozen)
            @test DZ.damped_velocity((zone,), SVector(x, 0.), velocity, kind, 0.1) == velocity
        end
    end
    @test iszero(DZ.damped_velocity((zone,), SVector(16., 0.), velocity, Fluid, 10.))
    @test DZ.damped_velocity((), SVector(16., 0.), velocity, Fluid, 10.) == velocity
    directional = DampingZone((0, 0, 0), (0, 0, 2); factors = (1, 0, 2))
    @test DZ.damped_velocity((directional,), SVector(0., 0., 1.), SVector(1., 1., 1.), Fluid, 0.1) ≈ SVector(0.75, 1, 0.5)
    @test_throws ArgumentError DampingZone((0, 0), (0, 0))
    @test_throws ArgumentError DampingZone((0, 0), (1, 0); rate = -1)
    @test_throws ArgumentError DampingZone((0, 0), (1, 0); factors = (1, NaN))
    @test_throws ArgumentError DZ.prepare_damping(directional, Val(2))
    @test ArtificialViscosity().boundary_factor == 1
    @test_throws ArgumentError ArtificialViscosity(boundary_factor = -1)
end

@testset "piston cases and bundled second-order reference" begin
    regular = collect(CSV.File(joinpath(PISTON_WAVE_DATA, "RegularTheory.csv")))
    @test maximum(abs(row.displacement - first(regular_piston_state(row.time))) for row in regular) < 1e-6
    irregular = collect(CSV.File(joinpath(PISTON_WAVE_DATA, "IrregularMotion.csv")))
    @test length(irregular) == 3001
    @test first(irregular).time == 0 && last(irregular).time == 30
    @test all(diff([row.time for row in irregular]) .> 0)
    @test maximum(row -> abs(row.displacement), irregular) > 0.05
    for termination in (:beach, :damping), waves in (:regular, :irregular)
        case = waves_piston_case(; termination, waves, visualize = false)
        @test getproperty.(case.geometry, :Type) == [Fixed, Moving, Fluid]
        @test all(!isempty(p.positions) for p in case.sampled)
        @test all(p[1] > 0 && 0 < p[2] < 0.66 for p in case.sampled[3].positions)
        @test all(p[1] <= 9 || p[2] > (p[1] - 9) / 10 for p in case.sampled[3].positions) || termination == :damping
        @test (case.damping === nothing) == (termination == :beach)
        @test case.constants.m₀ ≈ 1000 * 0.02^2
        @test case.metadata.GPUCellSubdivision == 1
        @test case.scheme isa DualSPHysicsSymplectic
        @test first(trajectory_state(case.motion.translation, 30.0))[1] ≈
            (waves == :regular ? first(regular_piston_state(30.0)) : last(irregular).displacement)
    end
    @test_throws ArgumentError waves_piston_case(termination = :unknown)
    @test_throws ArgumentError waves_piston_case(waves = :irregular, duration = 30.1)
    @test_throws ArgumentError waves_piston_case(duration = 30.1)
end

if CUDA.functional()
    function pair_acceleration(kind, factor, reference)
        T = Float32
        geometry = [SPHGeometry{2,T}([SVector(0., 0.)]; Density = 1000, GroupMarker = 1, Type = kind),
                    SPHGeometry{2,T}([SVector(0.02, 0.)]; Density = 1000, GroupMarker = 2, Type = Fluid)]
        metadata = SimulationMetaData{2,T}(SimulationName = "WallViscosity", SaveLocation = tempdir(), GPUBoundaryForces = false)
        particles = AllocateDataStructures(geometry, metadata)
        particles.Velocity[2] = SVector(-1.0f0, 0.0f0)
        gpu = upload_particles(particles, metadata)
        support = GPUSupportArrays{2,T}(gpu, metadata)
        constants = SimulationConstants{T}(2; dx = 0.02, c₀ = 20, g = 0)
        kernel = SPHKernelInstance{2,T}(WendlandC2(); dx = 0.02f0)
        cells = CellListWorkspace{2,T}(2)
        SPHExampleGPU.SPHCellList.rebuild_cell_list!(gpu, cells, kernel.H⁻¹; support, constants, kernel)
        SPHExampleGPU.GPUKernels.launch_interactions!(support.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient,
            support.∇Cᵢ, support.∇◌rᵢ, gpu.Position, gpu.Density, gpu.Velocity, gpu.Type,
            cells.CellStart, gpu.CellID, cells.grid_dev, HostStep(0.001f0, 0.0f0),
            ZeroDensityDiffusion(), ArtificialViscosity(boundary_factor = factor), kernel, constants, Val(false), Val(false);
            lanes = Val(1), boundary_forces = Val(false),
            viscous_dt = reference ? CUDA.zeros(T, 2) : nothing)
        index = findfirst(==(2), Array(gpu.GroupMarker))
        return Array(gpu.Acceleration)[index]
    end

    @testset "wall viscosity factor preserves fluid pairs in both interaction paths" begin
        for reference in (false, true), kind in (Fixed, Moving, Fluid)
            full = pair_acceleration(kind, 1, reference)
            zero_wall = pair_acceleration(kind, 0, reference)
            @test full[1] > 0
            @test kind == Fluid ? zero_wall == full : iszero(zero_wall)
        end
    end

    function run_damping_integration(scheme, graph)
        mktempdir() do dir
            geometry = [SPHGeometry{2,Float32}([SVector(13.25, 0.), SVector(0., 0.)]; Density = 1000, GroupMarker = 1, Type = Fluid)]
            metadata = SimulationMetaData{2,Float32}(SimulationName = "Damping", SaveLocation = dir,
                SimulationTime = 0.02, OutputTimes = 0.02, GPUUseGraph = graph,
                GPUClipTimeStepToOutput = true, GPUDoublePosition = true,
                VisualizeInParaview = false, OpenLogFile = false, ExportGridCells = false)
            particles = AllocateDataStructures(geometry, metadata)
            fill!(particles.Velocity, SVector(1.0f0, 0.0f0))
            logger = SimulationLogger(dir)
            result = try
                RunSimulation(SimGeometry = geometry, SimMetaData = metadata, SimParticles = particles,
                    SimConstants = SimulationConstants{Float32}(2; dx = 0.02, c₀ = 20, g = 0),
                    SimKernel = SPHKernelInstance{2,Float32}(WendlandC2(); dx = 0.02f0),
                    SimViscosity = ZeroViscosity(), SimDensityDiffusion = ZeroDensityDiffusion(),
                    SimTimeStepping = scheme, SimLogger = logger,
                    SimDamping = DampingZone((11., 0.), (15.5, 0.)))
            finally
                isopen(logger.LoggerIo) && close(logger.LoggerIo)
            end
            order = sortperm(particles.ID)
            @test result.final_state.complete
            @test particles.Velocity[order[1]][1] ≈ exp(-2.5 * 0.02) rtol = 0.002
            @test particles.Velocity[order[2]] == SVector(1.0f0, 0.0f0)
            graph && @test result.configuration.execution.graph_batches > 0
            return particles.Position[order], particles.Velocity[order]
        end
    end

    @testset "accepted-step damping in both integrators and CUDA graphs" begin
        for scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            @test run_damping_integration(scheme, false) == run_damping_integration(scheme, true)
        end
    end

    @testset "device damping preserves particle kinds and supports graph replay" begin
        for D in (2, 3), T in (Float32, Float64)
            origin, finish = ntuple(_ -> 0., D), ntuple(i -> i == 1 ? 2. : 0., D)
            zone = DampingZone(origin, finish)
            positions = [SVector{D,Float64}(ntuple(d -> d == 1 ? x : 0., D)) for x in (0., 1., 2., 3., 3.1, 1., 1., 1., 1.)]
            types = [Fluid, Fluid, Fluid, Fluid, Fluid, Fixed, Moving, Floating, Frozen]
            velocities = fill(ones(SVector{D,T}), length(types))
            x, v, kinds = CuArray(positions), CuArray(velocities), CuArray(types)
            step = HostStep(T(0.01), T(0))
            DZ.launch_damping!((zone,), x, v, kinds, step)
            graph = CUDA.capture() do
                DZ.launch_damping!((zone,), x, v, kinds, step)
            end
            CUDA.launch(CUDA.instantiate(graph))
            actual = Array(v)
            expected = [DZ.damped_velocity((zone,), p, DZ.damped_velocity((zone,), p, vel, kind, T(0.01)), kind, T(0.01))
                        for (p, vel, kind) in zip(positions, velocities, types)]
            @test actual ≈ expected
        end
    end

    @testset "all four piston variants run and export" begin
        for termination in (:beach, :damping), waves in (:regular, :irregular)
            mktempdir() do dir
                output = run_waves_piston(; termination, waves, dx = 0.04, duration = 0.01,
                    output_interval = 0.01, visualize = false, save_location = dir)
                @test all(isfinite, output.particles.Density)
                @test all(p -> all(isfinite, p), output.particles.Position)
                @test output.case.metadata.TotalTime >= 0.01
                @test isfile(joinpath(dir, output.case.metadata.SimulationName * "_Theory.csv"))
                @test isfile(output.comparison)
                comparison = collect(CSV.File(output.comparison))
                @test first(comparison).eta_WG6 == 0
                h5open(joinpath(dir, output.case.metadata.SimulationName * ".vtkhdf"), "r") do file
                    velocity = reshape(read(file["VTKHDF/VelocityProbes/PointData/Velocity"]), 3, :)
                    @test last(comparison).vy_SPH == velocity[2, end]
                end
                @test any(endswith(".vtkhdf"), readdir(dir))
            end
        end
    end
end

end
