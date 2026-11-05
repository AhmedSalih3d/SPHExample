# Configure direct let-block examples without duplicating their setup.
using SPHExampleGPU

function prepare_chrono_case(filename;options...)
    scope = Module(gensym(:ChronoReferenceCase))
    Core.eval(scope,:(include(path) = Base.include($scope,path)))
    configured = Set{Symbol}()
    result = try
        Base.include(scope,joinpath(@__DIR__,"..","example",filename)) do expression
            expression isa Expr && expression.head === :let || return expression
            body = expression.args[end]
            for statement in body.args
                if statement isa Expr && statement.head === :(=) && statement.args[1] isa Symbol && haskey(options,statement.args[1])
                    statement.args[2] = QuoteNode(options[statement.args[1]])
                    push!(configured,statement.args[1])
                end
            end
            stop = findfirst(s -> s isa Expr && s.head === :call && first(s.args) === :mkpath,body.args)
            stop === nothing && error("example output boundary not found")
            body.args = body.args[1:stop-1]
            push!(body.args,:((;geometry = SimulationGeometry,constants = SimConstants,kernel = SimKernel,meta = SimMetaData,
                dynamics = SimRigidBodyDynamics,viscosity = SimViscosity,diffusion = SimDensityDiffusion,scheme = time_stepping,
                motion = SimMotion,measurements = SimMeasurements,save_surfaces)))
            expression
        end
    catch error
        error isa LoadError ? throw(error.error) : rethrow()
    end
    all(name -> name in configured,keys(options)) || throw(ArgumentError("unknown example setting"))
    return result
end

function run_chrono_case(filename;options...)
    case = prepare_chrono_case(filename;options...)
    mkpath(case.meta.SaveLocation)
    logger = SimulationLogger(case.meta.SaveLocation)
    try
        result = RunSimulation(SimGeometry = case.geometry,SimMetaData = case.meta,SimConstants = case.constants,
            SimParticles = AllocateDataStructures(case.geometry,case.meta),SimKernel = case.kernel,SimLogger = logger,
            SimViscosity = case.viscosity,SimDensityDiffusion = case.diffusion,SimTimeStepping = case.scheme,
            SimRigidBodyDynamics = case.dynamics,SimMotion = case.motion,SimMeasurements = case.measurements)
        return result,case
    finally
        isopen(logger.LoggerIo) && close(logger.LoggerIo)
    end
end
