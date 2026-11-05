using CSV,StaticArrays,LinearAlgebra,SPHExampleGPU

function chrono_joint_validation(case,directory)
    rows = collect(CSV.File(joinpath(directory,case.meta.SimulationName*"_Floating.csv")))
    D = length(first(first(case.geometry).Particles.Position))
    times = sort!(unique(r.Time for r in rows))
    residual = 0.
    for time in times
        frame = sort!(filter(r -> r.Time == time,rows);by = r -> r.Body)
        center = [SVector{D,Float64}(ntuple(d -> r[Symbol("Center:",d-1)],D)) for r in frame]
        state = D == 2 ? (;center,angle = [Float64(r.Angle) for r in frame]) :
            (;center,orientation = [SVector{4,Float64}(ntuple(d -> r[Symbol("Orientation:",d-1)],4)) for r in frame])
        if case.dynamics !== nothing
            residual = max(residual,maximum((e.translation_error for e in rigid_joint_errors(case.dynamics,case.geometry,state));init = 0.))
        end
    end
    return (;frames = length(times),body_samples = length(rows),max_joint_error_m = residual)
end
