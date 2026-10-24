module FluidFreeFallTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra

const DRIVER = SPHExampleGPU.SPHCellList

function fluid_free_fall(D, T; double_position, compact, graph, scheme)
    constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(20), m₀ = T(1000 * 0.02^D))
    kernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx)
    # Equal densities and velocities make every interaction derivative zero.
    positions = [SVector{D, T}(ntuple(d -> T(0.5) + T((i >> (d - 1)) & 1) * constants.dx, D))
                 for i in reverse(0:(2^D - 1))]
    geometry = [SPHGeometry{D, T}(positions; Density = constants.ρ₀, Type = Fluid, GroupMarker = 1)]
    metadata = SimulationMetaData{D, T}(
        SimulationName = "FluidFreeFall", SaveLocation = tempdir(), SimulationTime = T(0.1),
        OutputTimes = T(0.025), GPUDoublePosition = double_position,
        GPUCompactBuffers = compact, GPUUseGraph = graph, GPUMaxStepsPerSync = 32,
        VisualizeInParaview = false, OpenLogFile = false)
    metadata.TimeSteppingMode = scheme
    particles = AllocateDataStructures(geometry, metadata)
    gpu = upload_particles(particles, metadata)
    support = GPUSupportArrays{D, T}(gpu, metadata)
    cells = CellListWorkspace{D, position_float_type(metadata)}(length(gpu))
    reduction = ReductionWorkspace{SVector{3, T}}(length(gpu))
    motion = MotionArrays(geometry, particles)
    state = StepState{T}(dx = one(T) + kernel.h)
    gravity = SVector{D, T}(ntuple(d -> d == D ? -constants.g : zero(T), D))
    tolerance = 100eps(T)

    for frame in 1:4
        metadata.OutputIterationCounter = frame
        DRIVER.SimulationLoop(ZeroDensityDiffusion(), ZeroViscosity(), kernel, metadata,
            constants, gpu, cells, support, reduction, motion, state)
        # Compare with the accepted time because output deadlines can be overshot.
        time = metadata.TotalTime
        order = sortperm(Array(gpu.ID))
        _, velocity_half = DRIVER.predictor_arrays(support, gpu)
        expected_position = particles.Position .+ Ref(gravity * (time^2 / 2))
        @test all(isapprox.(Array(gpu.Position)[order], expected_position; atol = tolerance, rtol = tolerance))
        @test all(v -> isapprox(v, gravity * time; atol = tolerance, rtol = tolerance), Array(gpu.Velocity))
        @test all(v -> isapprox(v, gravity * (time - metadata.CurrentTimeStep / 2);
            atol = tolerance, rtol = tolerance), Array(velocity_half))
        @test all(==(gravity), Array(gpu.Acceleration))
        @test all(==(constants.ρ₀), Array(gpu.Density))
    end
    @test metadata.Iteration > 10
    @test cells.nrebuilds > 1
    @test graph == !isempty(state.graphs)
end

if CUDA.functional()
    @testset "fluid-only free fall applies gravity once" begin
        for (D, T, double_position, compact) in (
                (2, Float32, false, true), (2, Float32, true, true), (2, Float64, false, false),
                (3, Float32, false, true), (3, Float32, true, false), (3, Float64, false, true)),
            graph in (false, true), scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
            @testset "$D dimensions, $T, double=$double_position, compact=$compact, graph=$graph, $(nameof(typeof(scheme)))" begin
                fluid_free_fall(D, T; double_position, compact, graph, scheme)
            end
        end
    end
else
    @testset "fluid free fall requires CUDA" begin
        @test_skip CUDA.functional()
    end
end

end
