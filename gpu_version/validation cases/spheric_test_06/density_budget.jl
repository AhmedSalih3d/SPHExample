module DensityBudgetStudy

using SPHExampleGPU, CUDA, CSV, TOML, Test
using StaticArrays: SVector
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06
const SS = SPHExampleGPU.GPUStepState

mutable struct BudgetObserver{F}
    fields::F
    ids::Vector{Int}
    slots::Vector{Int}
    rebuild::Int
    start::Float64
    stop::Float64
    rows::Vector{NamedTuple}
    pending::Vector{NamedTuple}
end

function BudgetObserver(host; start = 0.3, stop = 0.9)
    chosen = Int[]
    for type in (Moving, Fluid), y in (2.1, 2.5, 2.9), layer in (0, 1)
        desired = SVector(1.0 + (type == Moving ? 1 : -1)*(layer+0.5)*0.0125, y)
        candidates = findall(==(type), host.Type)
        distances = [sum(abs2, host.Position[i] - desired) for i in candidates]
        push!(chosen, Int(host.ID[candidates[argmin(distances)]]))
    end
    unique!(chosen)
    fields = (; continuity = CUDA.zeros(Float32, length(host)),
        diffusion = CUDA.zeros(Float32, length(host)), absolute = CUDA.zeros(Float32, length(host)),
        rate = CUDA.zeros(Float32, length(host)), acceleration = CUDA.zeros(SVector{2,Float32}, length(host)))
    return BudgetObserver(fields, chosen, Int[], -1, start, stop, NamedTuple[], NamedTuple[])
end

function (observer::BudgetObserver)(stage, ctx, step)
    if stage == :corrector_stored || stage == :predictor_stored
        isempty(observer.pending) && return nothing
        density = stage == :predictor_stored ? ctx.sup.ρₙ⁺ : ctx.gpu.Density
        actual = Array(density[observer.slots])
        if stage == :predictor_stored
            for (record, stored) in zip(observer.pending, actual)
                raw = record.base_density + record.consumed_rate * (record.dt/2)
                expected = record.type == Int(Fluid) ? raw : max(raw, ctx.SimConstants.ρ₀)
                @test stored == expected
                push!(observer.rows, merge(record, (; stage = "predictor", raw_density = raw,
                    stored_density = stored, floor_increment = stored - raw, epsi = 0f0)))
            end
        else
            for (record, stored) in zip(observer.pending, actual)
                epsi = -(record.consumed_rate / record.evaluated_density) * record.dt
                raw = record.base_density * ((2 - epsi) / (2 + epsi))
                expected = record.type == Int(Fluid) ? raw : max(raw, ctx.SimConstants.ρ₀)
                @test stored == expected
                push!(observer.rows, merge(record, (; stage = "corrector", raw_density = raw,
                    stored_density = stored, floor_increment = stored - raw, epsi)))
            end
        end
        empty!(observer.pending)
        return nothing
    end
    step isa SS.HostStep && return nothing
    SS.readback!(ctx.state)
    ctx.state.ih[SS.I_STOP] == SS.STOP_NONE || return nothing
    time, dt = ctx.state.fh[SS.F_TIME], Float32(ctx.state.fh[SS.F_DT])
    observer.start <= time <= observer.stop || return nothing
    if observer.rebuild != ctx.cl.nrebuilds
        ids = Array(ctx.gpu.ID)
        observer.slots = [only(findall(==(id), ids)) for id in observer.ids]
        observer.rebuild = ctx.cl.nrebuilds
    end
    slots = observer.slots
    half = stage == :corrector_rate
    position, velocity = half ? SPHExampleGPU.SPHCellList.predictor_arrays(ctx.sup, ctx.gpu) :
        (ctx.gpu.Position, ctx.gpu.Velocity)
    density = half ? ctx.sup.ρₙ⁺ : ctx.gpu.Density
    pairs = half ? ctx.sup.PosCellsₙ⁺ : ctx.sup.PosCells
    launch_interactions!(observer.fields.rate, observer.fields.acceleration, ctx.gpu.Kernel,
        ctx.gpu.KernelGradient, ctx.sup.∇Cᵢ, ctx.sup.∇◌rᵢ,
        position, density, velocity, ctx.gpu.Type, ctx.cl.CellStart, ctx.gpu.CellID,
        ctx.cl.grid_dev, HostStep(dt, time), ctx.SimDensityDiffusion, ctx.SimViscosity,
        ctx.SimKernel, ctx.SimConstants, Val(false), Val(false);
        lanes = Val(1), boundary_forces = Val(false), pos_cells = pairs,
        fastmath = false, frozen_particles = get(ctx, :frozen_particles, Val(true)),
        density_budget = observer.fields)
    base = Array(ctx.gpu.Density[slots])
    evaluated = stage == :predictor_rate ? base : Array(ctx.sup.ρₙ⁺[slots])
    types = Array(ctx.gpu.Type[slots])
    c, d, absolute = (Array(getproperty(observer.fields, key)[slots]) for key in (:continuity, :diffusion, :absolute))
    rates = Array(ctx.sup.dρdtI[slots])
    empty!(observer.pending)
    for j in eachindex(slots)
        tolerance = max(1f-5, 64eps(Float32)*absolute[j])
        @test abs((c[j]+d[j])-rates[j]) <= tolerance
        if ctx.SimDensityDiffusion isa ZeroDensityDiffusion ||
                (ctx.SimDensityDiffusion isa LinearDensityDiffusion && types[j] != Fluid)
            @test d[j] == 0
        end
        push!(observer.pending, (; id = observer.ids[j], type = Int(types[j]),
            iteration = Int(ctx.state.ih[SS.I_ITER]), time, dt, base_density = base[j],
            evaluated_density = evaluated[j], continuity_rate = c[j], diffusion_rate = d[j],
            absolute_pair_rate = absolute[j], consumed_rate = rates[j],
            divergence_estimate = -c[j]/evaluated[j]))
    end
    return nothing
end

function run_budget(model_name; trace = true, dx = 0.0125, duration = 0.9, start = 0.3,
                    output_root = @__DIR__)
    models = Dict("original" => ZeroGravityLinearDensityDiffusion(),
        "fluid_only" => LinearDensityDiffusion(), "zero" => ZeroDensityDiffusion())
    haskey(models, model_name) || error("Unknown density model")
    tag = "$(model_name)_$(trace ? "replay" : "control")_n$(round(Int,1/dx))"
    save = joinpath(output_root, "runs", "density_budget", tag)
    results = joinpath(output_root, "results", "density_budget", tag)
    isfile(joinpath(save, "MovingSquare.vtkhdf")) && error("Refusing to overwrite $save")
    case = build_case(; dx, duration, interval = 0.02, sound_speed = 160.0, cfl = 0.2,
        kernel_h_factor = 1.2sqrt(2), graph = false, density_diffusion_model = models[model_name], save)
    case.metadata.GPUFusePredictor = false
    case.metadata.GPULanesPerParticle = 1
    motion = PrescribedMotion(3; translation = body_trajectory(motion_kind = :integrated))
    case = merge(case, (; motion, motion_kind = :integrated))
    host = AllocateDataStructures(case.geometry, case.metadata)
    observer = trace ? BudgetObserver(host; start, stop = duration) : nothing
    mkpath(save)
    logger = SimulationLogger(save)
    result = open(joinpath(save, "console.log"), "w") do io
        redirect_stdout(io) do
            try
                RunSimulation(SimGeometry = case.geometry, SimMetaData = case.metadata,
                    SimConstants = case.constants, SimKernel = case.kernel, SimLogger = logger,
                    SimParticles = host, SimViscosity = Laminar(), SimDensityDiffusion = models[model_name],
                    SimTimeStepping = SymplecticTimeStepping(), SimMotion = motion,
                    SimDensityObserver = observer)
            finally
                isopen(logger.LoggerIo) && close(logger.LoggerIo)
            end
        end
    end
    completed = SphericTest06.save_run_results(case, result; results_dir = results)
    if trace
        isempty(observer.pending) || error("Unfinished stage records")
        isempty(observer.rows) && error("No density-budget records captured")
        CSV.write(joinpath(results, "density_budget.csv"), observer.rows)
        open(joinpath(results, "trace_settings.toml"), "w") do io
            TOML.print(io, Dict("ids" => observer.ids, "start" => start, "stop" => duration,
                "graph" => false, "lanes" => 1))
        end
    end
    return completed
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    model = isempty(ARGS) ? "original" : ARGS[1]
    DensityBudgetStudy.@testset "Density stage budgets" begin
        DensityBudgetStudy.run_budget(model)
    end
end
