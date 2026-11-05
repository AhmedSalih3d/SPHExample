# Configure the direct let-block example without duplicating its geometry.
using SPHExampleGPU

function prepare_pelamis_2d(; options...)
    scope = Module(gensym(:PelamisCase))
    Core.eval(scope, :(include(path) = Base.include($scope, path)))
    path = joinpath(@__DIR__, "..", "example", "Pelamis2d.jl")
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
            stop === nothing && error("Pelamis example output boundary not found")
            body.args = body.args[1:stop-1]
            push!(body.args, :((; geometry = SimulationGeometry, constants = SimConstants,
                kernel = SimKernel, meta = SimMetaData, dynamics = SimRigidBodyDynamics,
                viscosity = SimViscosity, diffusion = SimDensityDiffusion, scheme = time_stepping,
                motion = PistonMotion,measurements = WaveGauges,piston_state = piston_state,
                save_surfaces = save_pelamis_surfaces)))
            expression
        end
    catch error
        error isa LoadError ? throw(error.error) : rethrow()
    end
    all(name -> name in configured, keys(options)) || throw(ArgumentError("unknown Pelamis setting"))
    return result
end

function run_pelamis_2d(; options...)
    case = prepare_pelamis_2d(; options...)
    mkpath(case.meta.SaveLocation)
    particles = AllocateDataStructures(case.geometry, case.meta)
    logger = SimulationLogger(case.meta.SaveLocation)
    try
        return RunSimulation(SimGeometry = case.geometry, SimMetaData = case.meta,
            SimConstants = case.constants, SimParticles = particles, SimKernel = case.kernel,
            SimLogger = logger, SimViscosity = case.viscosity, SimDensityDiffusion = case.diffusion,
            SimTimeStepping = case.scheme, SimRigidBodyDynamics = case.dynamics,
            SimMotion = case.motion,SimMeasurements = case.measurements)
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end
