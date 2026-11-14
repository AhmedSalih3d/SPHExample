# Recompute metrics and the report from saved coefficient histories; no simulation.
include(joinpath(@__DIR__, "case.jl"))
using .SphericTest06
using TOML
using Printf

function write_report()
    directory = joinpath(@__DIR__, "results", "validation")
    comparison = compare_histories(directory)
    metrics = comparison.metrics
    length(metrics) == 18 || error("The report requires all nine full validation cases")
    all(row -> row.full_interval, metrics) || error("The report requires full 8-second histories")
    metadata(re, n; tag = "validation") = TOML.parsefile(joinpath(@__DIR__, "results", tag,
        @sprintf("Re%03d_n%03d.toml", re, n)))
    metric(re, n, component) = only(filter(row -> row.reynolds == re &&
        isapprox(row.dx, 1/n) && row.component == component, metrics))
    passed = all(row -> row.drag_pass && row.state_ok, filter(row -> isapprox(row.dx, 1/60), metrics)) &&
        length(comparison.resolution) == 6 && all(row -> row.resolution_pass, comparison.resolution)
    open(joinpath(@__DIR__, "REPORT.md"), "w") do io
        println(io, "# SPHERIC Test 06 validation report\n")
        println(io, "Result: **", passed ? "PASS" : "FAIL — the tested DBC configuration is not validated", "**.\n")
        println(io, "Run date: 2026-11-14. All nine combinations of Re = 50, 100, 150 and L/dp = 20, 40, 60 completed the full 8 s with 401 frames each and no lost or frozen particles. The pressure and viscous drag errors fail the chosen criteria at the finest spacing.\n")
        println(io, "## Drag comparison\n")
        println(io, "Both SPH and the official finite difference histories are integrated over identical 0.1 s bins from 0.1 through 8.0 s (79 bins). The first 0.1 s is excluded because the reference starts at 0.003 s; no t = 0 reference is invented. Each bin uses piecewise-linear interpolation and the trapezoidal integral, including every original sample in the bin.\n")
        println(io, "Normalized RMS error = `norm(SPH_bin_means - FD_bin_means) / norm(FD_bin_means)`. Late bias is `mean(SPH) / mean(FD) - 1` over 4–8 s. No fitted scale, time shift, or reference extrapolation is applied. The raw histories remain in the three-column `.dat` files and CSV files.\n")
        println(io, "| Re | L/dp | Pressure NRMSE | Viscous NRMSE | Pressure late bias | Viscous late bias |")
        println(io, "|---:|---:|---:|---:|---:|---:|")
        for re in (50,100,150), n in (20,40,60)
            p, v = metric(re,n,:pressure), metric(re,n,:viscous)
            @printf(io, "| %d | %d | %.2f%% | %.2f%% | %+.2f%% | %+.2f%% |\n", re, n,
                100p.nrmse, 100v.nrmse, 100p.late_mean_relative_bias, 100v.late_mean_relative_bias)
        end
        println(io, "\n![Binned pressure and viscous drag](results/validation/drag_comparison.png)\n")
        println(io, "![Unfiltered SPH drag](results/validation/raw_drag.png)\n")
        println(io, "The SPH pressure histories contain strong startup oscillations absent from the incompressible reference. Refining particles alone does not remove this discrepancy. Viscous late-time means can be closer while the full transient error still fails; a late-time average alone would overstate agreement.\n")
        println(io, "## Resolution sensitivity\n")
        println(io, "Change between L/dp = 40 and 60, measured as the RMS difference divided by the RMS finite difference reference over the same bins:\n")
        println(io, "| Re | Pressure change | Viscous change |")
        println(io, "|---:|---:|---:|")
        for re in (50,100,150)
            p = only(filter(row -> row.reynolds == re && row.component == :pressure, comparison.resolution))
            v = only(filter(row -> row.reynolds == re && row.component == :viscous, comparison.resolution))
            @printf(io, "| %d | %.2f%% | %.2f%% |\n", re, 100p.relative_rms_change, 100v.relative_rms_change)
        end
        println(io, "\nThese three spacings do not establish convergence. In particular, the two finest pressure curves differ by more than the chosen 5% resolution criterion at all Reynolds numbers.\n")
        println(io, "## Physical and numerical checks\n")
        println(io, "| Re | L/dp | Fluid / square / fixed | Steps | Max density deviation | Max Mach | Min solid clearance [m] |")
        println(io, "|---:|---:|---:|---:|---:|---:|---:|")
        for re in (50,100,150), n in (20,40,60)
            m = metadata(re,n)
            @printf(io, "| %d | %d | %d / %d / %d | %d | %.3f%% | %.4f | %.6f |\n", re, n,
                m["fluid_particles"], m["square_particles"], m["fixed_particles"], m["steps"],
                100m["density_relative_max"], m["mach_max"], m["minimum_clearance"])
        end
        all_metadata = [metadata(re,n) for re in (50,100,150) for n in (20,40,60)]
        @printf(io, "\nMaximum prescribed center-position error is %.3e m; maximum wall-velocity error is %.3e m/s. Checks are performed on every saved frame. Negative clearance denotes a fluid particle center inside the nominal tank wall or moving square. These small penetrations are reported explicitly and fail the strict zero-penetration check.\n\n",
            maximum(m -> m["motion_error_max"], all_metadata), maximum(m -> m["motion_velocity_error_max"], all_metadata))
        println(io, "## Sound-speed sensitivity\n")
        println(io, "After the baseline runs showed a pressure discrepancy, two additional full runs varied only sound speed at Re = 100, L/dp = 60. They retain the same geometry, physical viscosity, output interval and comparison method. These are diagnostic runs, not a replacement for the baseline resolution study.\n")
        println(io, "| Sound speed [m/s] | Pressure NRMSE | Viscous NRMSE | Max density deviation | Max Mach |")
        println(io, "|---:|---:|---:|---:|---:|")
        for speed in (40,80,160)
            tag = speed == 40 ? "validation" : "sound_speed_$speed"
            extra = speed == 40 ? comparison : compare_histories(joinpath(@__DIR__, "results", tag))
            m = metadata(100,60; tag)
            p = only(filter(row -> row.reynolds == 100 && isapprox(row.dx,1/60) && row.component == :pressure, extra.metrics))
            v = only(filter(row -> row.reynolds == 100 && isapprox(row.dx,1/60) && row.component == :viscous, extra.metrics))
            @printf(io, "| %d | %.2f%% | %.2f%% | %.3f%% | %.4f |\n", speed,
                100p.nrmse, 100v.nrmse, 100m["density_relative_max"], m["mach_max"])
        end
        println(io, "\n![Sound-speed sensitivity at Re = 100](results/validation/sound_speed_comparison.png)\n")
        println(io, "\nThe source benchmark assumes incompressibility. Low particle-density variation by itself does not demonstrate that a weakly compressible pressure transient agrees with it. A dedicated boundary/pressure investigation and further independent convergence evidence are needed before claiming validation.\n")
        println(io, "## Qualitative particle fields\n")
        println(io, "![Finest-grid speed fields](results/validation/speed_fields.png)\n")
        println(io, "Speed snapshots at 5 and 8 s are qualitative. No quantitative vorticity or field-norm validation is claimed.\n")
        println(io, "## Provenance and verification\n")
        m = first(all_metadata)
        println(io, "Official input and reference provenance: [reference/README.md](reference/README.md). Solver source commit: `", m["solver_commit"], "`. Simulation code is unchanged. Julia ", m["julia_version"], ", CUDA.jl ", m["cuda_package_version"], ", CUDA driver ", m["cuda_driver_version"], ", ", m["gpu"], "; Windows. Each run's `.toml` records its settings and diagnostics. Reported wall times include startup/JIT/output and are not matched performance benchmarks.\n")
        println(io, "The focused tests passed 195 assertions covering fixture integrity, geometry and viscosity, absolute motion, force sign and action/reaction, GPU force parity in Float32/Float64, time integration, an end-to-end HDF run, and rejection of incomplete validation. The full repository CUDA suite was not run. Reproduction commands and the project-specific acceptance criteria are in [README.md](README.md).")
    end
    println("Wrote ", joinpath(@__DIR__, "REPORT.md"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_report()
end
