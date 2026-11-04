# Read the example's setup without starting simulation/output. Tests and
# benchmarks share the actual let-block inputs rather than another geometry.
using SPHExampleGPU
include(joinpath(@__DIR__, "..", "example", "data", "poiseuille", "validation.jl"))

function prepare_poiseuille_2d(; options...)
    scope = Module(gensym(:PoiseuilleCase))
    Core.eval(scope, :(include(path) = Base.include($scope, path)))
    path = joinpath(@__DIR__, "..", "example", "Poiseuille2d.jl")
    configured = Set{Symbol}()
    setup = try
        Base.include(scope, path) do expression
            expression isa Expr && expression.head === :let || return expression
            body = expression.args[end]
            for statement in body.args
                if statement isa Expr && statement.head === :(=) && statement.args[1] isa Symbol
                    name = statement.args[1]
                    if haskey(options, name)
                        statement.args[2] = QuoteNode(options[name])
                        push!(configured, name)
                    end
                end
            end
            first_output = findfirst(statement -> statement isa Expr && statement.head === :call &&
                first(statement.args) === :mkpath, body.args)
            first_output === nothing && error("Poiseuille example output boundary not found")
            body.args = body.args[1:first_output-1]
            push!(body.args, :((; geometry = SimulationGeometry, constants = SimConstants,
                kernel = SimKernel, periodicity = Periodicity, forces = SimExternalForces,
                meta = SimMetaData, viscosity = SimViscosity, diffusion = SimDensityDiffusion)))
            return expression
        end
    catch error
        error isa LoadError ? throw(error.error) : rethrow()
    end
    all(name -> name in configured, keys(options)) || throw(ArgumentError("unknown Poiseuille setting"))
    return setup
end

poiseuille_2d_geometry(; options...) = prepare_poiseuille_2d(; options...)

function run_poiseuille_2d(; options...)
    case = prepare_poiseuille_2d(; options...)
    mkpath(case.meta.SaveLocation)
    particles = AllocateDataStructures(case.geometry, case.meta)
    logger = SimulationLogger(case.meta.SaveLocation)
    try
        return RunSimulation(SimGeometry = case.geometry, SimMetaData = case.meta, SimParticles = particles,
            SimConstants = case.constants, SimKernel = case.kernel, SimLogger = logger,
            SimTimeStepping = SymplecticTimeStepping(), SimViscosity = case.viscosity,
            SimDensityDiffusion = case.diffusion, SimExternalForces = case.forces)
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end
