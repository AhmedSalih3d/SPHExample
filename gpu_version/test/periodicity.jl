module PeriodicityTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra, HDF5

const PB = SPHExampleGPU.PeriodicBoundaries
const GK = SPHExampleGPU.GPUKernels
const CL = SPHExampleGPU.SPHCellList
const GRID = SPHExampleGPU.GPUCellGrid

include(joinpath(@__DIR__, "..", "example", "Periodicity2d.jl"))

@testset "DualSPHysics Periodicity geometry" begin
    case = periodicity_2d_geometry()
    @test length.(getproperty.(case.geometry, :Particles)) == [5002, 18806]
    @test case.periodicity.lower ≈ -0.011
    @test case.periodicity.upper ≈ 0.801
    @test case.periodicity.translation ≈ SVector(-0.812, 0.3)
    @test isempty(intersect(Set(case.sampled[1].positions), Set(case.sampled[2].positions)))
    @test case.constants.α == 0.2f0
    @test case.constants.c₀ ≈ 15.5347f0 rtol = 1e-4
end

@testset "periodic configuration and crossing direction" begin
    for D in (2, 3), axis in 1:D, T in (Float32, Float64)
        shift = SVector{D, T}(ntuple(d -> d == mod1(axis + 1, D) ? T(0.3) : zero(T), D))
        p = PeriodicBoundary{D, T}(axis; min = -0.1, max = 0.9, shift)
        x = SVector{D, T}(ntuple(d -> d == axis ? T(0.25) : zero(T), D))
        @test PB.wrap_periodic_position(x, p) == x
        for n in (-3, -1, 1, 4)
            @test PB.wrap_periodic_position(x - n * p.translation, p) ≈ x atol = 10eps(T)
        end
        upper = setindex(x, p.upper, axis)
        @test PB.wrap_periodic_position(upper, p) ≈ upper + p.translation
        lower = setindex(x, p.lower, axis)
        @test PB.wrap_periodic_position(lower, p) == lower
        @test p.lower <= PB.wrap_periodic_position(setindex(x, prevfloat(p.lower), axis), p)[axis] < p.upper
        meta = SimulationMetaData{D, T}(SimulationName = "Periodic", SaveLocation = tempdir(), Periodicity = p)
        @test meta.Periodicity.axis == axis
        @test meta.Periodicity.translation ≈ p.translation
    end
    @test_throws ArgumentError PeriodicBoundary{2}(0; min = 0, max = 1)
    @test_throws ArgumentError PeriodicBoundary{2}(1; min = 1, max = 0)
    @test_throws ArgumentError PeriodicBoundary{2}(1; min = 0, max = Inf)
    @test_throws ArgumentError PeriodicBoundary{2}(1; min = 0, max = 1, shift = (1, 0))
    @test_throws ArgumentError PeriodicBoundary{3}(1; min = 0, max = 1, shift = (0, NaN, 0))
end

function pair_fixture(D, T, TP, R, axis)
    transverse = mod1(axis + 1, D)
    point(x, y) = SVector{D, TP}(ntuple(d -> d == axis ? TP(x) : d == transverse ? TP(y) : TP(0.2), D))
    p = PeriodicBoundary{D, TP}(axis; min = -0.01, max = 0.967,
        shift = ntuple(d -> d == transverse ? 0.3 : 0.0, D))
    points = [point(0.005, 0.4), point(0.95, 0.1), point(0.94, 0.12),
              point(0.02, 0.425), point(0.5, 0.2), point(0.52, 0.21)]
    velocity = [SVector{D, T}(ntuple(d -> T(0.01 * (i - 2d)), D)) for i in eachindex(points)]
    geometry = [SPHGeometry{D, T}(Particles = particle_struct_array(points, T(1000);
        Density = T.(1000 .+ 2 .* eachindex(points)), Velocity = velocity), GroupMarker = 1, Type = Fluid)]
    meta = SimulationMetaData{D, T}(SimulationName = "PeriodicPairs", SaveLocation = tempdir(),
        GPUDoublePosition = TP !== T)
    host = AllocateDataStructures(geometry, meta)
    gpu = upload_particles(host, meta)
    constants = SimulationConstants{T}(D; dx = T(0.02), c₀ = T(20), g = zero(T))
    kernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx)
    cl = CellListWorkspace{D, TP}(length(gpu); reach = R, periodicity = p)
    sup = GPUSupportArrays{D, T}(gpu, meta)
    CL.rebuild_cell_list!(gpu, cl, kernel.H⁻¹; support = sup, kernel)
    return (; gpu, cl, sup, kernel, constants, p)
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "periodic image forces versus all-pairs oracle" begin
        for D in (2, 3), (T, TP) in ((Float32, Float32), (Float32, Float64), (Float64, Float64)),
            R in (1, 2), axis in 1:D
            f = pair_fixture(D, T, TP, R, axis)
            (; gpu, cl, sup, kernel, constants, p) = f
            points, velocity, density = Array(gpu.Position), Array(gpu.Velocity), Array(gpu.Density)
            n = length(points)
            rate = CUDA.zeros(T, n)
            acceleration = similar(gpu.Acceleration)
            weight = CUDA.zeros(T, n)
            gradient = CUDA.zeros(SVector{D, T}, n)
            expected_rate = zeros(T, n)
            expected_acc = zeros(SVector{D, T}, n)
            expected_weight = zeros(T, n)
            for i in 1:n, j in 1:n, image in -1:1
                i == j && continue
                x = SVector{D, T}((points[i] - points[j]) + image * p.translation)
                dot(x, x) <= kernel.H² || continue
                q = norm(x) * kernel.h⁻¹
                grad = ∇Wᵢⱼ(kernel, q, x)
                expected_weight[i] += Wᵢⱼ(kernel, q)
                expected_rate[i] += density[i] * (constants.m₀ / density[j]) * dot(velocity[i] - velocity[j], grad)
                pi = EquationOfStateGamma7(density[i], constants.c₀, constants.ρ₀)
                pj = EquationOfStateGamma7(density[j], constants.c₀, constants.ρ₀)
                expected_acc[i] -= constants.m₀ * (pi + pj) * inv(density[i]) * inv(density[j]) * grad
            end
            @test all(>(zero(T)), expected_weight)
            for lanes in (1, 4)
                GK.launch_interactions!(rate, acceleration, weight, gradient, sup.∇Cᵢ, sup.∇◌rᵢ,
                    gpu.Position, gpu.Density, gpu.Velocity, gpu.Type, cl.CellStart, gpu.CellID,
                    cl.grid_dev, HostStep(T(1e-4), zero(T)), ZeroDensityDiffusion(), ZeroViscosity(),
                    kernel, constants, Val(true), Val(false); lanes = Val(lanes),
                    pos_cells = sup.PosCells, periodicity = p)
                @test Array(rate) ≈ expected_rate rtol = 3e-5 atol = 3e-5
                @test Array(acceleration) ≈ expected_acc rtol = 3e-5 atol = 3e-5
                @test Array(weight) ≈ expected_weight rtol = 3e-5
            end
        end
    end

    @testset "wrap before fixed-grid freezing, preserve particle identity" begin
        for T in (Float32, Float64), compact in (false, true)
            p = PeriodicBoundary{2}(1; min = 0, max = 1, shift = (0, 0.3))
            geometry = [SPHGeometry{2, T}([SVector{2, T}(0.99, 0.1), SVector{2, T}(0.01, 0.4)];
                Density = 1000, GroupMarker = 1, Type = Fluid)]
            meta = SimulationMetaData{2, T}(SimulationName = "Wrap", SaveLocation = tempdir(), GPUCompactBuffers = compact)
            host = AllocateDataStructures(geometry, meta)
            gpu = upload_particles(host, meta)
            cl = CellListWorkspace{2, T}(2; fixed_grid = true, expansion = ((false, false), (true, true)), periodicity = p)
            CL.rebuild_cell_list!(gpu, cl, T(10))
            positions = Array(gpu.Position)
            positions[1] += SVector{2, T}(3, -0.9)
            positions[2] -= SVector{2, T}(2, -0.6)
            copyto!(gpu.Position, positions)
            state = StepState{T}()
            GRID.prepare_cell_grid!(cl, gpu.Position, T(10); step = state,
                freeze_state = CL.grid_freeze_state(gpu), periodic_state = CL.periodic_particle_state(gpu))
            readback!(state)
            @test state.ih[SPHExampleGPU.GPUStepState.I_STOP] == SPHExampleGPU.GPUStepState.STOP_REBUILD
            @test GRID.consume_cell_grid!(cl, state.ih)
            CL.rebuild_cell_list!(gpu, cl, T(10); prepared = true)
            @test sort(Array(gpu.ID)) == [1, 2]
            @test all(==(Fluid), Array(gpu.Type))
            @test all(x -> 0 <= x[1] < 1, Array(gpu.Position))
            @test cl.frozen_total == 0
        end
        @test_throws ArgumentError begin
            cl = CellListWorkspace{2, Float32}(1; periodicity = PeriodicBoundary{2}(1; min = 0, max = 0.1))
            GRID.prepare_cell_grid!(cl, CuArray([SVector(0.0f0, 0.0f0)]), 10.0f0)
        end
    end

    @testset "mDBC interpolation across a shifted seam" begin
        for D in (2, 3), T in (Float32, Float64), updated in (false, true), double_position in (false, true)
            TP = double_position ? Float64 : T
            point(x, y) = SVector{D, TP}(ntuple(d -> d == 1 ? TP(x) : d == D ? TP(y) : TP(0.2), D))
            wall, ghost = point(0.005, 0.35), point(0.005, 0.4)
            fluid = [point(x, y) for x in (0.94, 0.95), y in (0.09, 0.1, 0.11)] |> vec
            append!(fluid, [point(0.02, 0.4), point(0.025, 0.42)])
            p = PeriodicBoundary{D, TP}(1; min = -0.01, max = 0.967,
                shift = ntuple(d -> d == D ? 0.3 : 0.0, D))
            mode = updated ? UpdatedMDBC : SimpleMDBC
            function corrected(periodic)
                points = periodic ? fluid : [x[1] > 0.5 ? x + p.translation : x for x in fluid]
                geometry = [SPHGeometry{D, T}(Particles = particle_struct_array([wall], T(1000);
                    GhostPoints = [ghost], GhostNormals = [ghost - wall]), GroupMarker = 1, Type = Fixed),
                    SPHGeometry{D, T}(Particles = particle_struct_array(points, T(1003);
                        Velocity = fill(SVector{D, T}(ntuple(d -> T(0.1d), D)), length(points))), GroupMarker = 2, Type = Fluid)]
                meta = SimulationMetaData{D, T, NoShifting, NoKernelOutput, mode}(
                    SimulationName = "PeriodicMDBC", SaveLocation = tempdir(), GPUDoublePosition = double_position)
                gpu = upload_particles(AllocateDataStructures(geometry, meta), meta)
                constants = SimulationConstants{T}(D; dx = T(0.02), c₀ = T(20), g = zero(T))
                kernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx)
                cl = CellListWorkspace{D, TP}(length(gpu); periodicity = periodic ? p : nothing)
                sup = GPUSupportArrays{D, T}(gpu, meta)
                CL.rebuild_cell_list!(gpu, cl, kernel.H⁻¹; support = sup, kernel)
                step = HostStep(T(1e-4), zero(T))
                if updated
                    GK.launch_updated_mdbc!(gpu.Density, gpu.Position, gpu.Velocity, gpu.GhostPoints,
                        gpu.GhostNormals, gpu.GhostIndex, gpu.Type, cl.CellStart, cl.grid_dev, step,
                        kernel, constants, sup.boundary_data; compact_ghosts = Val(true), pos_cells = sup.PosCells,
                        periodicity = cl.periodicity)
                else
                    GK.launch_mdbc!(gpu.Density, gpu.Position, gpu.GhostPoints, gpu.GhostIndex, gpu.Type,
                        cl.CellStart, cl.grid_dev, step, kernel, constants; compact_ghosts = Val(true),
                        pos_cells = sup.PosCells, periodicity = cl.periodicity)
                end
                i = only(findall(==(Fixed), Array(gpu.Type)))
                return (; density = Array(gpu.Density)[i],
                    velocity = updated ? only(Array(sup.boundary_data.velocity)) : nothing)
            end
            periodic, unfolded = corrected(true), corrected(false)
            @test periodic.density ≈ unfolded.density rtol = 2e-5
            @test periodic.density ≈ T(1003) rtol = 2e-5
            updated && @test periodic.velocity ≈ unfolded.velocity rtol = 2e-5
        end
    end

    @testset "rigid wrapping preserves body state and torque arms" begin
        F = SPHExampleGPU.GPUFloating
        for D in (2, 3), T in (Float32, Float64), hinge in (false, true)
            TP = Float64
            p = PeriodicBoundary{D}(1; min = 0, max = 1, shift = ntuple(d -> d == D ? 0.3 : 0.0, D))
            center = SVector{D, TP}(ntuple(d -> d == 1 ? 0.97 : 0.3, D))
            points = [center + SVector{D, TP}(ntuple(d -> ((i >> (d - 1)) & 1) == 0 ? -0.025 : 0.025, D)) for i in 0:(2^D - 1)]
            constraints = hinge ? FloatingConstraints{D, T}(; HingePoint = Tuple(center),
                (D == 3 ? (; HingeAxis = (0, 0, 1)) : (;))...) : FloatingConstraints{D, T}()
            geometry = [SPHGeometry{D, T}(points; Density = 1000, GroupMarker = 1, Type = Floating,
                Floating = FloatingDetails{T}(RelativeWeight = 1.2, Constraints = constraints))]
            meta = SimulationMetaData{D, T}(SimulationName = "PeriodicRigid", SaveLocation = tempdir(), GPUDoublePosition = true)
            host = AllocateDataStructures(geometry, meta)
            gpu = upload_particles(host, meta)
            constants = SimulationConstants{T}(D; dx = T(0.01), c₀ = T(20), g = zero(T))
            f = FloatingArrays(geometry, host, constants; position_type = TP)
            cl = CellListWorkspace{D, TP}(length(gpu); periodicity = p)
            CL.rebuild_cell_list!(gpu, cl, T(10); floating = f)
            delta = SVector{D, TP}(ntuple(d -> d == 1 ? 0.07 : 0.0, D))
            copyto!(gpu.Position, Array(gpu.Position) .+ Ref(delta))
            for centers in (f.center, f.center_n, f.center_half)
                copyto!(centers, Array(centers) .+ Ref(delta))
            end
            acceleration = [SVector{D, T}(ntuple(d -> T(i + d), D)) for i in 1:length(gpu)]
            copyto!(gpu.Acceleration, acceleration)
            step = HostStep(T(0.001), zero(T))
            F.launch_floating_forces!(f, gpu.Acceleration, gpu.Position, gpu.Type, gpu.GroupMarker, f.center, constants.m₀, step)
            load = Array(f.force)
            before = Array(gpu.Position)
            before_center = only(Array(f.center))
            state = StepState{T}()
            GRID.prepare_cell_grid!(cl, gpu.Position, T(10); step = state, periodic_state = CL.periodic_particle_state(gpu, f))
            readback!(state)
            @test GRID.consume_cell_grid!(cl, state.ih)
            @test Array(gpu.Position) ≈ before .+ Ref(p.translation) atol = 1e-12
            for centers in (f.center, f.center_n, f.center_half)
                @test only(Array(centers)) ≈ before_center + p.translation atol = 1e-12
            end
            @test Array(gpu.Position) .- Ref(only(Array(f.center))) ≈ before .- Ref(before_center) atol = 1e-12
            hinge && @test only(Array(f.constraints)).point ≈ constraints.point + p.translation
            fill!(f.force, zero(T))
            F.launch_floating_forces!(f, gpu.Acceleration, gpu.Position, gpu.Type, gpu.GroupMarker, f.center, constants.m₀, step)
            @test Array(f.force) ≈ load rtol = 2e-6 atol = 1e-10
            @test all(==(Floating), Array(gpu.Type))
            @test all(iszero, Array(f.frozen))
        end
    end

    @testset "fixed walls stay put; moving ghosts follow seam translations" begin
        for compact in (false, true)
            T = Float64
            ghost = SVector(0.98, 0.2)
            geometry = [SPHGeometry{2, T}([SVector(0.5, 0.1)]; Density = 1000, GroupMarker = 1, Type = Fixed),
                SPHGeometry{2, T}(Particles = particle_struct_array([SVector(0.98, 0.15)], 1000.;
                    GhostPoints = [ghost], GhostNormals = [SVector(0., 0.05)]), GroupMarker = 2, Type = Moving)]
            meta = SimulationMetaData{2, T, NoShifting, NoKernelOutput, SimpleMDBC}(
                SimulationName = "PeriodicMoving", SaveLocation = tempdir(), GPUCompactBuffers = compact)
            gpu = upload_particles(AllocateDataStructures(geometry, meta), meta)
            p = PeriodicBoundary{2}(1; min = 0, max = 1, shift = (0, 0.3))
            cl = CellListWorkspace{2, T}(length(gpu); periodicity = p)
            positions = Array(gpu.Position)
            positions[2] += SVector(0.04, 0.)
            copyto!(gpu.Position, positions)
            ghosts = Array(gpu.GhostPoints)
            ghosts[compact ? 1 : 2] = ghost + SVector(0.04, 0.)
            copyto!(gpu.GhostPoints, ghosts)
            CL.rebuild_cell_list!(gpu, cl, 10.)
            order = sortperm(Array(gpu.ID))
            @test Array(gpu.Position)[order] ≈ [SVector(0.5, 0.1), SVector(0.02, 0.45)]
            @test any(x -> isapprox(x, SVector(0.02, 0.5)), Array(gpu.GhostPoints))
        end
    end

    @testset "accepted crossings with graphs, stepping and output" begin
        for scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping()),
            (T, double_position, compact, cache) in ((Float32, true, true, true), (Float64, false, false, false))
            results = map((false, true)) do graph
                mktempdir() do save
                    p = PeriodicBoundary{2}(1; min = 0, max = 1, shift = (0, 0.3))
                    geometry = [SPHGeometry{2, T}(Particles = particle_struct_array(
                        [SVector(0.95, 0.1), SVector(0.02, 0.8)], T(1000);
                        Velocity = [SVector{2, T}(10, 0), SVector{2, T}(-10, 0)]), GroupMarker = 1, Type = Fluid)]
                    meta = SimulationMetaData{2, T}(SimulationName = "PeriodicAdvection", SaveLocation = save,
                        SimulationTime = 0.12, OutputTimes = 0.04, GPUClipTimeStepToOutput = true,
                        VisualizeInParaview = false, OpenLogFile = false, Periodicity = p,
                        GPUFixedGrid = true, GPUGridExpansion = ((false, false), (true, true)),
                        GPUUseGraph = graph, GPUMaxStepsPerSync = 8, GPUDoublePosition = double_position,
                        GPUCompactBuffers = compact, GPUCacheInteractionFields = cache, ExportGridCells = true)
                    particles = AllocateDataStructures(geometry, meta)
                    logger = SimulationLogger(save)
                    constants = SimulationConstants{T}(2; dx = T(0.01), c₀ = T(20), g = zero(T))
                    result = try
                        redirect_stdout(devnull) do
                            RunSimulation(SimGeometry = geometry, SimMetaData = meta, SimParticles = particles,
                                SimConstants = constants, SimKernel = SPHKernelInstance{2, T}(WendlandC2(); dx = constants.dx),
                                SimLogger = logger, SimTimeStepping = scheme,
                                SimViscosity = ZeroViscosity(), SimDensityDiffusion = ZeroDensityDiffusion())
                        end
                    finally
                        isopen(logger.LoggerIo) && close(logger.LoggerIo)
                    end
                    order = sortperm(particles.ID)
                    @test result.final_state.complete
                    @test all(==(Fluid), particles.Type)
                    @test particles.ID[order] == [1, 2]
                    @test particles.Position[order] ≈ [SVector(0.15, 0.7), SVector(0.82, 0.2)] atol = 2e-5
                    @test particles.Density == fill(T(1000), 2)
                    @test result.configuration.execution.graph_capture_failures == 0
                    @test result.configuration.effective.periodicity == p
                    graph && @test result.configuration.execution.graph_batches > 0
                    h5open(joinpath(save, "PeriodicAdvection.vtkhdf"), "r") do io
                        root = io["VTKHDF"]
                        ids = read(root["PointData/ID"])
                        positions = read(root["Points"])
                        @test all(x -> 0 <= x < 1, positions[1, :])
                        for offset in read(root["Steps/PointOffsets"])
                            @test sort(ids[(offset + 1):(offset + 2)]) == [1, 2]
                        end
                    end
                    return (; position = particles.Position[order], velocity = particles.Velocity[order], density = particles.Density[order])
                end
            end
            @test results[1] == results[2]
        end
    end

    @testset "floating-body integration across the seam" begin
        for D in (2, 3), scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            mktempdir() do save
                T = Float32
                center = SVector{D, Float64}(ntuple(d -> d == 1 ? 0.97 : 0.3, D))
                points = [center + SVector{D, Float64}(ntuple(d -> ((i >> (d - 1)) & 1) == 0 ? -0.025 : 0.025, D)) for i in 0:(2^D - 1)]
                constraints = FloatingConstraints{D, T}(PrescribedVelocity = ntuple(d -> d == 1 ? 10 : 0, D),
                    PrescribedOmega = D == 2 ? (0.4,) : (0., 0., 0.4))
                geometry = [SPHGeometry{D, T}(points; Density = 1000, GroupMarker = 1, Type = Floating,
                    Floating = FloatingDetails{T}(RelativeWeight = 1.2, Constraints = constraints))]
                p = PeriodicBoundary{D}(1; min = 0, max = 1, shift = ntuple(d -> d == D ? 0.3 : 0., D))
                meta = SimulationMetaData{D, T}(SimulationName = "PeriodicFloating", SaveLocation = save,
                    SimulationTime = 0.02, OutputTimes = 0.02, GPUClipTimeStepToOutput = true,
                    VisualizeInParaview = false, OpenLogFile = false, Periodicity = p,
                    GPUUseGraph = true, GPUMaxStepsPerSync = 8, GPUDoublePosition = true)
                particles = AllocateDataStructures(geometry, meta)
                logger = SimulationLogger(save)
                constants = SimulationConstants{T}(D; dx = T(0.01), c₀ = T(20), g = zero(T))
                result = try
                    redirect_stdout(devnull) do
                        RunSimulation(SimGeometry = geometry, SimMetaData = meta, SimParticles = particles,
                            SimConstants = constants, SimKernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx),
                            SimLogger = logger, SimTimeStepping = scheme,
                            SimViscosity = ZeroViscosity(), SimDensityDiffusion = ZeroDensityDiffusion())
                    end
                finally
                    isopen(logger.LoggerIo) && close(logger.LoggerIo)
                end
                expected = center + SVector{D, Float64}(ntuple(d -> d == 1 ? 0.2 : 0., D)) + p.translation
                @test only(result.final_state.floating.center) ≈ expected atol = 1e-6
                @test result.diagnostics.frozen_particles == 0
                @test result.configuration.execution.graph_capture_failures == 0
                @test result.configuration.execution.graph_batches > 0
                order = sortperm(particles.ID)
                for i in eachindex(points), j in eachindex(points)
                    @test norm(particles.Position[order[i]] - particles.Position[order[j]]) ≈ norm(points[i] - points[j]) atol = 2e-6
                end
            end
        end
    end

    @testset "Periodicity case short GPU run" begin
        mktempdir() do save
            result = redirect_stdout(devnull) do
                run_periodicity_2d(; dx = 0.01, simulation_time = 0.02, output_interval = 0.01,
                    visualize = false, open_log_file = false, save_location = save)
            end
            @test result.final_state.complete
            @test all(isfinite, result.final_state.particles.Density)
            @test all(x -> all(isfinite, x), result.final_state.particles.Position)
            @test result.diagnostics.frozen_particles == 0
        end
    end
end

end
