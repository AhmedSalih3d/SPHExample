# Held Float 1 about the benchmark draft, using the matched free-heave solver.
include(joinpath(@__DIR__, "case.jl"))
include(joinpath(@__DIR__, "full_case.jl"))
using CUDA

function restoring_study(half_width = 1.2)
    dx, duration = 0.025, 2.0
    tag = "float1_restoring_half$(half_width)"
    output = joinpath(@__DIR__, "results", tag)
    mkpath(output)
    for offset in (-dx, 0.0, dx)
        name = "offset_$(round(Int,1000offset))mm"
        path = joinpath(output, name * ".csv")
        isfile(path) && continue
        release = merge(first(RELEASES), (; offset))
        save = joinpath(@__DIR__, "runs", tag, name)
        isfile(joinpath(save, "FreeHeave.vtkhdf")) && error("Existing particle history in $save")
        case = build_full_case(release; dx, duration, half_width, pause_time = duration + 1,
            output_interval = 0.02, output_variables = ["Type"], save_location = save)
        mkpath(save)
        logger = SimulationLogger(save)
        host = AllocateDataStructures(case.geometry, case.metadata)
        println("Holding Float 1 at ", name)
        flush(stdout)
        result = try
            RunSimulation(SimGeometry = case.geometry, SimMetaData = case.metadata,
                SimConstants = case.constants, SimKernel = case.kernel, SimLogger = logger,
                SimParticles = host, SimViscosity = case.viscosity, SimDensityDiffusion = case.diffusion,
                SimTimeStepping = case.stepping, SimDamping = case.damping)
        finally
            isopen(logger.LoggerIo) && close(logger.LoggerIo)
        end
        result.final_state.complete && result.diagnostics.frozen_particles == 0 ||
            error("Incomplete held-float run")
        rows = CSV.File(joinpath(save, "FreeHeave_Floating.csv"))
        center = getproperty(rows, Symbol("Center:2"))
        target = case.equilibrium_center_z + offset
        all(z -> isapprox(z, target; atol = 1e-8, rtol = 0), center) || error("Float moved during hold")
        force = getproperty(rows, Symbol("Force:2"))
        all(isfinite, force) && any(!iszero, force) || error("Invalid held load")
        isapprox(last(rows.Time), duration; atol = 1e-8) || error("Incomplete force history")
        CSV.write(path, (; time = rows.Time, fluid_force_n = force,
            weight_n = fill(release.mass * Float64(case.constants.g), length(force))))
        GC.gc(true)
        CUDA.reclaim()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) <= 1 || error("Usage: float1_restoring_study.jl [tank half-width]")
    restoring_study(isempty(ARGS) ? 1.2 : parse(Float64, only(ARGS)))
end
