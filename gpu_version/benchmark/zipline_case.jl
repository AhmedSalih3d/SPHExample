# Configurable access to the direct let-block example, without duplicating it.
using SPHExampleGPU

function prepare_zipline_3d(; options...)
    scope = Module(gensym(:ZipLineCase))
    Core.eval(scope, :(include(path) = Base.include($scope, path)))
    path = joinpath(@__DIR__, "..", "example", "ZipLine3d.jl")
    configured = Set{Symbol}()
    result = try
        Base.include(scope, path) do expression
            expression isa Expr && expression.head === :let || return expression
            body = expression.args[end]
            for statement in body.args
                if statement isa Expr && statement.head === :(=) && statement.args[1] isa Symbol && haskey(options, statement.args[1])
                    statement.args[2] = QuoteNode(options[statement.args[1]])
                    push!(configured, statement.args[1])
                end
            end
            stop = findfirst(s -> s isa Expr && s.head === :call && first(s.args) === :mkpath, body.args)
            stop === nothing && error("ZipLine example output boundary not found")
            body.args = body.args[1:stop-1]
            push!(body.args, :((; geometry = SimulationGeometry, constants = SimConstants,
                kernel = SimKernel, meta = SimMetaData, dynamics = SimRigidBodyDynamics,
                viscosity = SimViscosity, diffusion = SimDensityDiffusion, save_surfaces = save_zipline_surfaces)))
            expression
        end
    catch error
        error isa LoadError ? throw(error.error) : rethrow()
    end
    all(name -> name in configured, keys(options)) || throw(ArgumentError("unknown ZipLine setting"))
    return result
end

function run_zipline_3d(; options...)
    case = prepare_zipline_3d(; options...)
    mkpath(case.meta.SaveLocation)
    particles = AllocateDataStructures(case.geometry, case.meta)
    logger = SimulationLogger(case.meta.SaveLocation)
    try
        return RunSimulation(SimGeometry = case.geometry, SimMetaData = case.meta, SimParticles = particles,
            SimConstants = case.constants, SimKernel = case.kernel, SimLogger = logger,
            SimViscosity = case.viscosity, SimDensityDiffusion = case.diffusion,
            SimTimeStepping = SymplecticTimeStepping(), SimRigidBodyDynamics = case.dynamics)
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end
