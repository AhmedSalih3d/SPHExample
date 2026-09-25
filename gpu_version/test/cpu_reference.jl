# Run one benchmark case with the CPU package and store the final particle
# state, sorted by particle ID, in an HDF5 file. Used by the GPU test suite to
# compare against the reference implementation. Invoke from the repository
# root environment:
#
#     julia -t 8,0 --project=. gpu_version/test/cpu_reference.jl <case name> <sim time> <out.h5>
#
# (`-t N,0` avoids the interactive thread that Julia 1.12 adds with `-t auto`,
#  which the CPU code does not account for.)

using SPHExample
using HDF5
using StaticArrays

include(joinpath(@__DIR__, "..", "benchmark", "cases.jl"))

function main(args)
    casename, simtime_str, outfile = args
    idx  = findfirst(c -> c.name == casename, BENCH_CASES)
    idx === nothing && error("Unknown case $(casename)")
    case = BENCH_CASES[idx]
    simtime = parse(Float64, simtime_str)

    save = mktempdir()
    kw   = case.build(Float64, save)
    kw.SimMetaData.SimulationTime = simtime
    kw.SimMetaData.OutputTimes    = simtime
    particles = AllocateDataStructures(kw.SimGeometry)
    logger    = SimulationLogger(save; to_console = false)
    RunSimulation(; kw..., SimLogger = logger, SimParticles = particles)

    order = sortperm(particles.ID)
    h5open(outfile, "w") do fid
        fid["ID"]        = particles.ID[order]
        fid["Density"]   = particles.Density[order]
        fid["Pressure"]  = particles.Pressure[order]
        fid["Position"]  = stack(particles.Position[order])
        fid["Velocity"]  = stack(particles.Velocity[order])
        fid["Iteration"] = kw.SimMetaData.Iteration
        fid["TotalTime"] = kw.SimMetaData.TotalTime
    end
    println("reference written: ", outfile, " steps=", kw.SimMetaData.Iteration)
end

main(ARGS)
