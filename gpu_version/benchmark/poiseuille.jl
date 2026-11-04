# Paired, unprofiled whole-solver comparison with the installed DualSPHysics 5.4.
# julia --project=. -t 2,0 benchmark/poiseuille.jl [DualSPHysics_root] [repeats] [output_dir]
using SPHExampleGPU, CUDA, CSV, StaticArrays, LinearAlgebra, Statistics, Dates

include(joinpath(@__DIR__, "poiseuille_case.jl"))

function profile_metrics(y, u, v, density, time)
    reference = poiseuille_velocity.(y, time)
    return (; relative_l2 = norm(u - reference) / norm(reference),
        max_error = maximum(abs, u - reference), max_transverse = maximum(abs, v),
        density_min = minimum(density), density_max = maximum(density))
end

function native_run(directory)
    result = nothing
    wall = open(joinpath(directory, "console.log"), "w") do io
        redirect_stdout(io) do
            @elapsed result = run_poiseuille_2d(; simulation_time = 0.5, output_interval = 0.002,
                visualize = false, open_log_file = false, save_location = directory)
        end
    end
    CUDA.synchronize()
    p = result.final_state.particles
    fluid = sort(findall(==(Fluid), p.Type); by = i -> p.ID[i])
    length(fluid) == 12250 && result.diagnostics.frozen_particles == 0 || error("Native run lost particles")
    result.final_state.complete || error("Native simulation did not complete")
    result.configuration.execution.graph_capture_failures == 0 || error("Native CUDA graph capture failed")
    y, u, v, rho = last.(p.Position[fluid]), first.(p.Velocity[fluid]), last.(p.Velocity[fluid]), p.Density[fluid]
    CSV.write(joinpath(directory, "profile.csv"),
        [(; id = p.ID[i], x = p.Position[i][1], y = p.Position[i][2], u = p.Velocity[i][1],
            v = p.Velocity[i][2], density = p.Density[i]) for i in fluid])
    return (; solver_seconds = result.performance.wall_time_seconds, wall_seconds = wall,
        steps = result.performance.steps, time = result.final_state.time,
        frames = result.diagnostics.output_frames, rebuilds = result.diagnostics.cell_list_rebuilds,
        source_rebuilds = result.diagnostics.periodic_source_rebuilds,
        readbacks = result.diagnostics.step_state_readbacks,
        profile_metrics(y, u, v, rho, result.final_state.time)...,
        state = hcat(first.(p.Position[fluid]), y, u, v, rho))
end

function dual_profile(tool, directory)
    open(joinpath(directory, "export.log"), "w") do io
        run(pipeline(`$tool -dirin $(joinpath(directory, "data")) -savecsv $(joinpath(directory, "particles")) -onlytype:-all,+fluid -vars:-all,+idp,+vel,+rhop -files:250`; stdout = io, stderr = io))
    end
    lines = readlines(joinpath(directory, "particles_0250.csv"))
    time = parse(Float64, first(split(lines[2], ';')))
    start = findfirst(line -> startswith(line, "Pos.x"), lines)
    data = CSV.File(IOBuffer(join(lines[start:end], '\n')); delim = ';')
    rows = sort(collect(data); by = row -> getproperty(row, :Idp))
    length(rows) == 12250 || error("DualSPHysics run lost particles")
    x = Float64[getproperty(row, Symbol("Pos.x [m]")) for row in rows]
    y = Float64[getproperty(row, Symbol("Pos.z [m]")) for row in rows]
    u = Float64[getproperty(row, Symbol("Vel.x [m/s]")) for row in rows]
    v = Float64[getproperty(row, Symbol("Vel.z [m/s]")) for row in rows]
    rho = Float64[getproperty(row, Symbol("Rhop [kg/m^3]")) for row in rows]
    CSV.write(joinpath(directory, "profile.csv"), [(; x = x[i], y = y[i], u = u[i], v = v[i], density = rho[i]) for i in eachindex(x)])
    return (; time, profile_metrics(y, u, v, rho, time)..., state = hcat(x, y, u, v, rho))
end

function dual_run(executable, tool, case, directory)
    wall = open(joinpath(directory, "console.log"), "w") do io
        @elapsed run(pipeline(`$executable -gpu $case $directory -dirdataout data -svres -svtimers:0`; stdout = io, stderr = io))
    end
    # Configuration strings can contain unquoted semicolons; timing/count
    # columns precede that field and are parsed without shifting later columns.
    lines = readlines(joinpath(directory, "Run.csv"))
    fields = Dict(zip(split(lines[1], ';'), split(lines[2], ';')))
    number(key) = parse(Float64, replace(fields[key], "," => ""))
    number("Np") == 12750 && number("PartsOut") == 0 || error("DualSPHysics particle count changed")
    return (; solver_seconds = number("TSimul"), wall_seconds = wall, steps = Int(number("Steps")),
        frames = Int(number("PartFiles")), rebuilds = missing, source_rebuilds = missing, readbacks = missing,
        dual_profile(tool, directory)...)
end

function benchmark_poiseuille(dual_root; repeats = 3, directory = joinpath(@__DIR__, "..", "output", "poiseuille_benchmark_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"))
    repeats >= 2 || throw(ArgumentError("use at least two repetitions to alternate order"))
    directory = abspath(directory)
    ispath(directory) && throw(ArgumentError("choose a new output directory to preserve previous results"))
    mkpath(directory)
    bin = joinpath(dual_root, "bin", "windows")
    executable, gencase, tool = joinpath.(bin, ("DualSPHysics5.4_win64.exe", "GenCase_win64.exe", "PartVTK_win64.exe"))
    source = joinpath(dual_root, "examples", "main", "15_Poiseuille", "CasePoiseuille_Def.xml")
    all(isfile, (source, executable, gencase, tool)) || error("DualSPHysics 5.4 installation not found")
    # Run the distributed main/15 definition unchanged: Verlet with laminar+SPS.
    # Julia retains its supported symplectic integrator and the same SPS model.
    definition, case = joinpath(directory, "CasePoiseuille_Def"), joinpath(directory, "CasePoiseuille")
    write(definition * ".xml", read(source, String))
    open(joinpath(directory, "gencase.log"), "w") do io
        run(pipeline(`$gencase $definition $case -save:all`; stdout = io, stderr = io))
    end
    write(joinpath(directory, "environment.txt"), "Julia $VERSION\nCUDA.jl $(pkgversion(CUDA))\nDevice $(CUDA.name(CUDA.device()))\nDualSPHysics $executable\nRevision $(readchomp(`git rev-parse HEAD`))\nUnprofiled; Julia symplectic/CUDA graphs; DualSPHysics stock Verlet; SPS in both; subdivision 1; Float32 fields/Float64 positions; output every 0.002 s.\n")
    # Full-duration warmups exercise the kernels and graph/rebuild paths.
    println("Warming both solvers; output: $directory")
    for solver in (:native, :dual54)
        save = mkpath(joinpath(directory, "warm_$solver"))
        solver == :native ? native_run(save) : dual_run(executable, tool, case, save)
    end
    records = NamedTuple[]
    states = Dict{Symbol, Matrix{Float64}}()
    for repetition in 1:repeats
        order = isodd(repetition) ? (:native, :dual54) : (:dual54, :native)
        for (slot, solver) in enumerate(order)
            save = mkpath(joinpath(directory, "$(solver)_$repetition"))
            GC.gc(true)
            result = solver == :native ? native_run(save) : dual_run(executable, tool, case, save)
            previous = get!(states, solver, Float64.(result.state))
            state_difference = maximum(abs, result.state - previous)
            result.relative_l2 < 0.05 || error("$solver failed the 5% analytical profile threshold")
            # CSV rows need identical field order across both solver results.
            push!(records, (; solver = String(solver), repetition, order = slot, particles = 12750, fluid = 12250,
                dx = 0.00002, solver_seconds = result.solver_seconds, wall_seconds = result.wall_seconds,
                steps = result.steps, time = result.time, frames = result.frames,
                rebuilds = result.rebuilds, source_rebuilds = result.source_rebuilds, readbacks = result.readbacks,
                relative_l2 = result.relative_l2, max_error = result.max_error, max_transverse = result.max_transverse,
                density_min = result.density_min, density_max = result.density_max,
                repeat_state_max_difference = state_difference))
            CSV.write(joinpath(directory, "results.csv"), records)
            println("$solver repeat $repetition: $(round(result.wall_seconds; digits = 3)) s wall, $(result.steps) steps, L2=$(round(100 * result.relative_l2; digits = 3))%")
        end
    end
    native = filter(r -> r.solver == "native", records)
    dual = filter(r -> r.solver == "dual54", records)
    speedups = [dual[i].wall_seconds / native[i].wall_seconds for i in eachindex(native)]
    println("Median paired wall speedup: $(round(median(speedups); digits = 2))x; pair ratios: $speedups")
    return records
end

if abspath(PROGRAM_FILE) == @__FILE__
    dual_root = length(ARGS) >= 1 ? ARGS[1] : raw"W:\DualSPHysics_v5.4"
    repeats = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
    options = length(ARGS) >= 3 ? (; directory = ARGS[3]) : (;)
    benchmark_poiseuille(dual_root; repeats, options...)
end
