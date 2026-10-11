module RadialInteractionTests

using Test, SPHExampleGPU, CUDA, StaticArrays, LinearAlgebra, Random, HDF5

const GK = SPHExampleGPU.GPUKernels
const SK = SPHExampleGPU.SPHKernels
const DD = SPHExampleGPU.SPHDensityDiffusionModels
const VM = SPHExampleGPU.SPHViscosityModels
const CL = SPHExampleGPU.SPHCellList

struct OrderedViscosity <: SPHViscosity end
struct OrderedDiffusion <: SPHDensityDiffusion end

@inline function VM.compute_viscosity(::OrderedViscosity, kernel, constants, particles,
                                     x, v, gradient, d², ρᵢ, ρⱼ, invρᵢ, invρⱼ, i, j)
    T = eltype(x)
    value = T(i) * x + T(j) * v + particles.Velocity[i] * (ρᵢ * invρⱼ)
    return value, T(2) * value
end

@inline function DD.compute_density_diffusion(::OrderedDiffusion, kernel, constants, particles,
                                             x, gradient, d², ρᵢ, ρⱼ, invρᵢ, invρⱼ, i, j, types)
    value = (ρⱼ - ρᵢ) * dot(x, gradient) * invρⱼ + particles.Density[i] * invρᵢ
    return value, value / 2
end

function maximum_difference(actual, expected)
    return maximum(norm.(actual .- expected); init = 0.0)
end

function test_fields_close(actual, expected, tolerance)
    @test length(actual) == length(expected)
    for (a, b) in zip(actual, expected)
        @test size(a) == size(b)
        @test maximum_difference(a, b) <= tolerance * max(1, maximum(norm.(b); init = 0.0))
    end
end

@testset "radial pair algebra and supported combinations" begin
    rng = MersenneTwister(5813)
    for T in (Float32, Float64), D in (2, 3)
        constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(35), m₀ = T(1000 * 0.02^D), ν₀ = T(1e-4))
        kernel = SPHKernelInstance{D,T}(WendlandC2(); dx = constants.dx)
        @test GK.radial_interactions_supported(kernel, LinearDensityDiffusion(), ArtificialViscosity())
        @test !GK.radial_interactions_supported(kernel, ComplexDensityDiffusion(), ArtificialViscosity())
        @test !GK.radial_interactions_supported(kernel, LinearDensityDiffusion(), LaminarSPS())
        @test !GK.radial_interactions_supported(kernel, OrderedDiffusion(), ArtificialViscosity())
        @test !GK.radial_interactions_supported(kernel, LinearDensityDiffusion(), OrderedViscosity())
        cubic = SPHKernelInstance{D,T}(CubicSpline{T}(); dx = constants.dx)
        @test !GK.radial_interactions_supported(cubic, LinearDensityDiffusion(), ArtificialViscosity())
        tolerance = T === Float32 ? 2e-5 : 5e-14
        for q in (zero(T), eps(T), T(0.2), T(1), prevfloat(T(2)), T(2))
            direction = normalize(SVector{D,T}(randn(rng, T, D)))
            x = direction * (q * kernel.h)
            d² = dot(x, x)
            actual_q = sqrt(d²) * kernel.h⁻¹
            gradient = ∇Wᵢⱼ(kernel, actual_q, x)
            factor = SK.radial_gradient_factor(kernel, actual_q)
            @test gradient == factor * x
            inverse_distance = inv(d² + kernel.η²)
            for types in ([Fluid, Fluid], [Fluid, Fixed], [Fixed, Fluid], [Floating, Fluid], [Moving, Fixed])
                for model in (ZeroDensityDiffusion(), LinearDensityDiffusion(), ZeroGravityLinearDensityDiffusion())
                    for reverse_pair in (false, true)
                        r = reverse_pair ? -x : x
                        g = reverse_pair ? -gradient : gradient
                        i, j = reverse_pair ? (2, 1) : (1, 2)
                        ρᵢ, ρⱼ = reverse_pair ? (T(995), T(1007)) : (T(1007), T(995))
                        expected = compute_density_diffusion(model, kernel, constants, nothing,
                            r, g, d², ρᵢ, ρⱼ, inv(ρᵢ), inv(ρⱼ), i, j, types)
                        actual = DD.radial_density_diffusion(model, kernel, constants,
                            r, d², ρᵢ, ρⱼ, inv(ρⱼ), i, j, types, factor, inverse_distance)
                        @test isapprox(actual[1], expected[1]; rtol = tolerance, atol = tolerance)
                        @test actual[2] == -actual[1]
                        @test all(isfinite, actual)
                    end
                end
            end
            for viscosity in (ZeroViscosity(), ArtificialViscosity(), Laminar()), v in (x, -x, zero(x))
                expected, _ = compute_viscosity(viscosity, kernel, constants, nothing,
                    x, v, gradient, d², T(1007), T(995), inv(T(1007)), inv(T(995)), 1, 2)
                actual = VM.radial_viscosity(viscosity, kernel, constants, x, v, gradient,
                    T(1007), T(995), inverse_distance)
                @test isapprox(actual, expected; rtol = tolerance, atol = tolerance)
                reversed, _ = compute_viscosity(viscosity, kernel, constants, nothing,
                    -x, -v, -gradient, d², T(995), T(1007), inv(T(995)), inv(T(1007)), 2, 1)
                @test isapprox(actual, -reversed; rtol = tolerance, atol = tolerance)
                @test all(isfinite, actual)
            end
        end
        mktempdir() do save
            default = SimulationMetaData{D,T}(SimulationName = "Default", SaveLocation = save)
            @test !default.GPURadialInteractions
            enabled = SimulationMetaData{D,T}(SimulationName = "Radial", SaveLocation = save,
                GPURadialInteractions = true)
            @test enabled.GPURadialInteractions
        end
    end
end

function interaction_fixture(D, T, TP; reach = 1, kernel_type = WendlandC2())
    constants = SimulationConstants{T}(dx = T(0.02), c₀ = T(35), m₀ = T(1000 * 0.02^D), ν₀ = T(1e-4))
    kernel = SPHKernelInstance{D,T}(kernel_type; dx = constants.dx)
    points = [SVector{D,TP}(ntuple(d -> TP(idx[d]) * TP(0.02), D))
        for idx in CartesianIndices(ntuple(_ -> 0:3, D))][:]
    append!(points, [points[1], points[1] + SVector{D,TP}(ntuple(d -> d == 1 ? TP(kernel.H) : zero(TP), D)),
        points[1] + SVector{D,TP}(ntuple(d -> d == 1 ? TP(nextfloat(kernel.H)) : zero(TP), D))])
    geometry = [SPHGeometry{D,T}(points; Density = T(1000), GroupMarker = 1, Type = Fluid)]
    host = AllocateDataStructures(geometry; position_type = TP)
    for i in eachindex(host)
        host.Density[i] = T(995 + i % 11)
        host.Velocity[i] = SVector{D,T}(ntuple(d -> T(0.04) * sin(T(i * d)), D))
        host.Type[i] = i <= 2 ? Fixed : i <= 4 ? Moving : i <= 6 ? Floating : Fluid
    end
    Pressure!(host.Pressure, host.Density, constants)
    gpu = upload_particles(host; position_type = TP)
    cells = CellListWorkspace{D,TP}(length(gpu); reach)
    CL.rebuild_cell_list!(gpu, cells, kernel.H⁻¹)
    support = GPUSupportArrays{D,T}(length(gpu); position_type = TP, updated_mdbc = true)
    step = HostStep(T(1e-4), zero(T))
    GK.launch_inv_density!(support.InvDensity, gpu.Density, step)
    support.PosCells === nothing || GK.launch_pos_cells!(support.PosCells, gpu.Position,
        gpu.CellID, cells.grid_dev, step, kernel)
    boundary = support.boundary_data
    copyto!(boundary.active, [i % 3 != 0 for i in 1:length(gpu)])
    velocity = Array(gpu.Velocity)
    copyto!(boundary.velocity, -velocity)
    copyto!(boundary.tangent_velocity, velocity / 2)
    return (; constants, kernel, gpu, cells, support, step)
end

function interaction_fields(f, diffusion, viscosity; radial = nothing, lanes = 1, boundary_forces = true,
                            boundary_data = nothing, step = f.step)
    (; gpu, cells, support) = f
    extra = radial === nothing ? (;) : (; radial = Val(radial))
    GK.launch_interactions!(support.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient,
        support.∇Cᵢ, support.∇◌rᵢ, gpu.Position, gpu.Density, support.InvDensity, gpu.Pressure,
        gpu.Velocity, gpu.Type, cells.CellStart, gpu.CellID, cells.grid_dev, step,
        diffusion, viscosity, f.kernel, f.constants, Val(true), Val(true);
        lanes = Val(lanes), boundary_forces = Val(boundary_forces), pos_cells = support.PosCells,
        boundary_data, extra...)
    return map(Array, (support.dρdtI, gpu.Acceleration, gpu.Kernel, gpu.KernelGradient, support.∇Cᵢ, support.∇◌rᵢ))
end

function simulation_fixture(D, T, TP, boundary_mode; floating = false)
    wall = [SVector{D,TP}(ntuple(d -> d == D ? zero(TP) : TP(idx[d]) * TP(0.04), D))
        for idx in CartesianIndices(ntuple(_ -> 0:3, D - 1))][:]
    displacement = SVector{D,T}(ntuple(d -> d == D ? T(0.06) : zero(T), D))
    ghosts = wall .+ Ref(SVector{D,TP}(displacement))
    fluid = [SVector{D,TP}(ntuple(d -> TP(idx[d]) * TP(0.04), D))
        for idx in CartesianIndices(ntuple(d -> d == D ? (1:3) : (0:3), D))][:]
    walls = SPHGeometry{D,T}(Particles = particle_struct_array(wall, T(1000);
        GhostPoints = ghosts, GhostNormals = fill(displacement, length(wall))),
        GroupMarker = 1, Type = Moving, Motion = MotionDetails{D,T}(Velocity = T(0.05),
            Direction = SVector{D,T}(ntuple(d -> d == 1 ? one(T) : zero(T), D)), StartTime = -one(T), Duration = one(T)))
    geometry = [walls, SPHGeometry{D,T}(fluid; Density = T(1001), GroupMarker = 2, Type = Fluid)]
    if floating
        body = [SVector{D,TP}(ntuple(d -> TP(0.02) * TP(idx[d]) + (d == D ? TP(0.16) : TP(0.04)), D))
            for idx in CartesianIndices(ntuple(_ -> 0:1, D))][:]
        push!(geometry, SPHGeometry{D,T}(body; Density = T(1000), GroupMarker = 3, Type = Floating,
            Floating = FloatingDetails{T}(RelativeWeight = T(1.2))))
    end
    constants = SimulationConstants{T}(dx = T(0.04), m₀ = T(1000 * 0.04^D), c₀ = T(25))
    kernel = SPHKernelInstance{D,T}(WendlandC2(); dx = constants.dx)
    return (; geometry, constants, kernel, boundary_mode)
end

function simulate(f; radial, graph, scheme, precision = Float32, position_type = Float64,
                  diffusion = LinearDensityDiffusion(), viscosity = ArtificialViscosity())
    return mktempdir() do save
        D = length(eltype(f.geometry[1].Particles.Position))
        T = precision
        fields = ["Density", "Pressure", "Velocity", "Acceleration", "ID", "Kernel", "KernelGradient"]
        meta = SimulationMetaData{D,T,PlanarShifting,StoreKernelOutput,f.boundary_mode,NoLog}(
            SimulationName = "Radial", SaveLocation = save, SimulationTime = T(0.003),
            OutputTimes = T(0.001), GPUDoublePosition = position_type !== T, GPUUseGraph = graph,
            GPUMaxStepsPerSync = 8, GPURadialInteractions = radial, OutputVariables = fields,
            GPUOutputQueueBytes = 1024^2, VisualizeInParaview = false, OpenLogFile = false, ExportGridCells = true)
        host = AllocateDataStructures(f.geometry, meta)
        logger = SimulationLogger(save)
        try
            redirect_stdout(devnull) do
                RunSimulation(SimGeometry = f.geometry, SimConstants = f.constants, SimKernel = f.kernel,
                    SimMetaData = meta, SimParticles = host, SimLogger = logger,
                    SimDensityDiffusion = diffusion, SimViscosity = viscosity, SimTimeStepping = scheme)
            end
        finally
            close(logger.LoggerIo)
        end
        order = sortperm(host.ID)
        output = h5open(joinpath(save, "Radial.vtkhdf"), "r") do file
            root = file["VTKHDF"]
            (; times = read(root["Steps/Values"]), fields = Dict(name => size(root["PointData"][name]) for name in keys(root["PointData"])))
        end
        return (; steps = meta.Iteration, time = meta.TotalTime,
            state = map(name -> getproperty(host, name)[order], (:Position, :Velocity, :Density, :Pressure, :Acceleration)), output)
    end
end

function mode_switch_run(graph)
    D, T, TP = 2, Float32, Float64
    f = simulation_fixture(D, T, TP, NoMDBC)
    return mktempdir() do save
        meta = SimulationMetaData{D,T}(SimulationName = "Switch", SaveLocation = save,
            SimulationTime = T(0.009), OutputTimes = T(0.003), OutputIterationCounter = 1,
            GPUDoublePosition = true, GPUUseGraph = graph, GPUMaxStepsPerSync = 8)
        meta.TimeSteppingMode = SymplecticTimeStepping()
        host = AllocateDataStructures(f.geometry, meta)
        Pressure!(host.Pressure, host.Density, f.constants)
        gpu = upload_particles(host; position_type = TP)
        support = GPUSupportArrays{D,T}(length(gpu); position_type = TP)
        cells = CellListWorkspace{D,TP}(length(gpu))
        reduction = ReductionWorkspace{SVector{3,T}}(length(gpu))
        motion = MotionArrays(f.geometry, host)
        state = StepState{T}(dx = one(T) + f.kernel.h)
        old_graphs = CUDA.CuGraphExec[]
        for mode in (false, true, false)
            meta.GPURadialInteractions = mode
            SimulationLoop(LinearDensityDiffusion(), ArtificialViscosity(), f.kernel,
                meta, f.constants, gpu, cells, support, reduction, motion, state)
            @test state.radial_interactions == mode
            if graph
                @test !isempty(state.graphs)
                @test all(exec -> all(old -> exec !== old, old_graphs), values(state.graphs))
                old_graphs = collect(values(state.graphs))
            end
            meta.OutputIterationCounter += 1
        end
        order = sortperm(Array(gpu.ID))
        return (; steps = meta.Iteration, time = meta.TotalTime,
            state = map(name -> Array(getproperty(gpu, name))[order], (:Position, :Velocity, :Density, :Pressure, :Acceleration)))
    end
end

if CUDA.functional()
    @testset "radial GPU interactions preserve gating, state and reference fallbacks" begin
        for (D, T, TP, reach, lanes) in ((2, Float32, Float32, 1, 4), (3, Float32, Float64, 2, 1),
                                       (3, Float64, Float64, 1, 4), (2, Float64, Float64, 2, 32))
            f = interaction_fixture(D, T, TP; reach)
            tolerance = T === Float32 ? 8e-6 : 3e-13
            for (diffusion, viscosity) in ((LinearDensityDiffusion(), ArtificialViscosity()),
                (ZeroGravityLinearDensityDiffusion(), Laminar()), (ZeroDensityDiffusion(), ZeroViscosity()))
                for boundary in (nothing, f.support.boundary_data), forces in (false, true)
                    reference = interaction_fields(f, diffusion, viscosity; lanes, boundary_data = boundary, boundary_forces = forces)
                    explicit = interaction_fields(f, diffusion, viscosity; radial = false, lanes,
                        boundary_data = boundary, boundary_forces = forces)
                    @test reference == explicit
                    optimized = interaction_fields(f, diffusion, viscosity; radial = true, lanes,
                        boundary_data = boundary, boundary_forces = forces)
                    test_fields_close(optimized, reference, tolerance)
                end
            end
            for (diffusion, viscosity) in ((OrderedDiffusion(), ArtificialViscosity()),
                (LinearDensityDiffusion(), OrderedViscosity()), (ComplexDensityDiffusion(), Laminar()),
                (ZeroGravityLinearDensityDiffusion(), LaminarSPS()))
                reference = interaction_fields(f, diffusion, viscosity; lanes)
                @test interaction_fields(f, diffusion, viscosity; radial = true, lanes) == reference
            end
            cubic = interaction_fixture(D, T, TP; reach, kernel_type = CubicSpline{T}())
            reference = interaction_fields(cubic, LinearDensityDiffusion(), ArtificialViscosity(); lanes)
            @test interaction_fields(cubic, LinearDensityDiffusion(), ArtificialViscosity(); radial = true, lanes) == reference
        end
        f = interaction_fixture(2, Float32, Float64)
        before = interaction_fields(f, LinearDensityDiffusion(), ArtificialViscosity())
        interaction_fields(f, LinearDensityDiffusion(), ArtificialViscosity(); radial = true)
        @test CUDA.@allocated(interaction_fields(f, LinearDensityDiffusion(), ArtificialViscosity(); radial = false)) == 0
        @test CUDA.@allocated(interaction_fields(f, LinearDensityDiffusion(), ArtificialViscosity(); radial = true)) == 0
        before = interaction_fields(f, LinearDensityDiffusion(), ArtificialViscosity())
        state = StepState{Float32}()
        SPHExampleGPU.GPUStepState.set_slot!(state.i, state.ih, SPHExampleGPU.GPUStepState.I_STOP, SPHExampleGPU.GPUStepState.STOP_OUTPUT)
        @test interaction_fields(f, LinearDensityDiffusion(), ArtificialViscosity(); radial = true, step = state) == before
    end

    @testset "radial integration, boundaries, outputs and graph mode changes" begin
        for (D, T, TP, B, floating) in ((2, Float32, Float64, NoMDBC, true),
                                       (2, Float32, Float32, SimpleMDBC, false),
                                       (2, Float64, Float64, UpdatedMDBC, false),
                                       (3, Float32, Float64, NoMDBC, false))
            f = simulation_fixture(D, T, TP, B; floating)
            for scheme in (SymplecticTimeStepping(), SingleNeighborTimeStepping())
                reference = simulate(f; radial = false, graph = false, scheme, precision = T, position_type = TP)
                optimized = simulate(f; radial = true, graph = false, scheme, precision = T, position_type = TP)
                replay = simulate(f; radial = true, graph = true, scheme, precision = T, position_type = TP)
                @test replay == optimized
                @test optimized.steps == reference.steps
                @test optimized.output.fields == reference.output.fields
                @test length(optimized.output.times) == length(reference.output.times)
                @test optimized.time ≈ reference.time rtol = T === Float32 ? 2e-5 : 2e-13
                test_fields_close(optimized.state, reference.state, T === Float32 ? 5e-5 : 2e-12)
            end
        end
        @test mode_switch_run(true) == mode_switch_run(false)
        f = simulation_fixture(2, Float32, Float64, NoMDBC)
        reference = simulate(f; radial = false, graph = true, scheme = SymplecticTimeStepping(), viscosity = OrderedViscosity())
        result = @test_logs (:warn, r"GPURadialInteractions.*reference interactions") simulate(f;
            radial = true, graph = true, scheme = SymplecticTimeStepping(), viscosity = OrderedViscosity())
        @test result == reference
    end
else
    @info "CUDA unavailable; skipping radial device/integration checks."
end

end
