using CSV, TOML, Printf, Statistics

const DIRECTORY = joinpath(@__DIR__,"results","comparison")

function write_comparison_report()
    data = collect(CSV.File(joinpath(DIRECTORY,"summary.csv")))
    row(id,solver = "native") = only(filter(r -> r.case == id && r.solver == solver,data))
    motion = TOML.parsefile(joinpath(DIRECTORY,"motion.toml"))
    timing = collect(CSV.File(joinpath(DIRECTORY,"timings.csv")))
    aligned_timing = collect(CSV.File(joinpath(DIRECTORY,"timings_aligned.csv")))
    length(data) == 17 && length(timing) == 18 && length(aligned_timing) == 6 || error("Incomplete comparison")
    baseline = row("re100_c40")
    integrated = row("re100_integrated")
    open(joinpath(@__DIR__,"COMPARISON.md"),"w") do io
        println(io,"# Moving-square noise, accuracy and speed investigation\n")
        println(io,"Measured on 2026-11-14, with the installed DualSPHysics 5.4.355 and Julia GPU solver on an RTX 5080. **Both tested DBC configurations fail the incompressible SPHERIC drag comparison.** Numerical stability and successful completion do not establish pressure accuracy.\n")
        println(io,"## Why sound speed affects the result\n")
        println(io,"The Tait equation is `p = rho0*c0^2/7 * ((rho/rho0)^7 - 1)`. Near rho0, `delta_p ≈ c0^2*delta_rho`. Relative to the dynamic pressure `q = rho0*U^2/2`, this is `delta_p/q ≈ 2*(c0/U)^2*(delta_rho/rho0)`. At U = 1 and c0 = 40, a 0.1% density perturbation corresponds to 3.2 dynamic-pressure units. A small density error can therefore produce a large load fluctuation. Increasing c0 also changes the computed density perturbations; pressure noise need not increase or decrease monotonically.\n")
        println(io,"The solver advances pressure waves rather than solving an instantaneous incompressible pressure constraint. Across this 10 m closed tank, acoustic transit times are 0.25, 0.125 and 0.0625 s for c0 = 40, 80 and 160 m/s. The simple box fundamental `c0/(2*10)` is 2, 4 and 8 Hz. The square and its motion perturb these modes. At c0 = 40, both solvers show a dominant drag-error peak near 1.8–2.0 Hz across all three Reynolds numbers, consistent with tank-scale acoustic ringing. This is an inference from the formulation, spectra and controls, rather than a claim that every fluctuation is a single acoustic mode.\n")
        println(io,"A stable time integrator can preserve a bounded oscillation for a long time. It does not guarantee small oscillation amplitude, strong acoustic damping, accurate incompressible pressure, or correct wall traction. The usual Mach/density criterion also does not establish that the tank's acoustic transit time is short compared with the imposed acceleration transient.\n")
        println(io,"The density-diffusion coefficient in the discrete model scales with `delta*h*c0`, and the acoustic timestep scales approximately as `CFL*h/c0`. Changing sound speed therefore changes stiffness, wave times, damping and computational work together. See the [official formulation](https://github.com/DualSPHysics/DualSPHysics/wiki/3.-SPH-formulation) for the EOS, acoustic timestep and density-diffusion background.\n")
        println(io,"## Current images and all-Re comparison\n")
        println(io,"![Current fields at all Reynolds numbers](results/comparison/solver_fields.png)\n")
        println(io,"![Raw drag comparison](results/comparison/solver_drag.png)\n")
        println(io,"Same initial lattice: 78,400 fluid, 1,600 square and 4,864 fixed particles, dp = 0.025 m, h = sqrt(2) dp, unit transverse depth. Re = 50/100/150 uses nu = 0.02/0.01/(1/150) m²/s. Both use DBC, laminar viscosity, no shifting, no SPS, zero gravity, c0 = 40, CFL = 0.2, Float32 physics and double positions, through 8 s with 401 saved frames. GenCase coordinates and type counts are checked against Julia's lattice. These are prescribed settings for this study; the distributed DualSPHysics moving-square example has different motion, viscosity and shifting.\n")
        println(io,"The initial native reference density diffusion acts on wall pairs and retains its pair orientation; Dual's Molteni DDT is suppressed for fluid targets whose support contains a DBC wall. Native regularization is `(0.01h)^2`, versus `(0.1h)^2` in Dual. These formulation differences are isolated below. The additional operator-aligned Re = 100 control uses the latter regularization and native Fourtakas-inner DDT, which reduces to the same target-oriented, wall-suppressed Molteni expression at zero gravity. Timestepping details and neighbor rebuild policies still differ.\n")
        println(io,"| Re | Solver | Binned pressure NRMSE | Raw pressure NRMSE | Binned viscous NRMSE | Pressure noise RMS [Cd] | Viscous noise RMS [Cd] |")
        println(io,"|---:|---|---:|---:|---:|---:|---:|")
        for re in (50,100,150), solver in ("native","dual")
            r = row("re$(re)_c40",solver)
            @printf(io,"| %d | %s | %.2f%% | %.2f%% | %.2f%% | %.4f | %.4f |\n",re,solver,
                100r.pressure_nrmse,100r.pressure_raw_nrmse,100r.viscous_nrmse,r.pressure_noise_rms,r.viscous_noise_rms)
        end
        println(io,"\nNRMSE uses the original report's 79 identical 0.1 s bins over 0.1–8 s. The noise proxy is the RMS of `(SPH - FD) - centered_0.5s_mean(SPH - FD)` over 1.5–6.5 s, sampled every 0.02 s. It measures rapidly varying error relative to the same reference, not all physical unsteadiness. The plotted curves remain raw; the filter is used only for the declared metric. No fitted scale or time shift is applied.\n")
        println(io,"Raw NRMSE instead integrates squared error exactly for the two piecewise-linear saved histories on their combined time grid over 0.1–8 s. This matters in the sound-speed study: a fixed 0.1 s averaging bin attenuates an 8 Hz oscillation more strongly than a 2 Hz oscillation. A reduction in binned error alone therefore cannot demonstrate that the unfiltered pressure became quieter. Both metrics and the raw curves are retained.\n")
        println(io,"![Pressure-error spectra](results/comparison/noise_spectra.png)\n")
        println(io,"Spectra remove a linear trend and use a Hann window over 1.5–6.5 s. Their resolution is about 0.20 Hz; output Nyquist frequency is 25 Hz. Higher-frequency content can alias, so no physical identification of unresolved frequencies is claimed.\n")
        println(io,"![Raw sound-speed comparison](results/comparison/sound_speed_noise.png)\n")
        println(io,"## What the isolated controls establish\n")
        println(io,"| Re = 100 control | Solver | Binned pressure NRMSE | Raw pressure NRMSE | Pressure noise RMS [Cd] | Binned viscous NRMSE |")
        println(io,"|---|---|---:|---:|---:|---:|")
        controls = (("re100_c40","c0 = 40, original"),("re100_c80","c0 = 80"),
            ("re100_c160","c0 = 160"),("re100_integrated","Consistent integrated-velocity motion"),
            ("re100_no_ddt","Density diffusion disabled"),("re100_dual_eta","Dual regularization only"),
            ("re100_dual_operators","Dual-aligned diffusion and regularization"),("re100_float64","Float64 physics"))
        for (id,label) in controls, solver in ("native","dual")
            matches = filter(r -> r.case == id && r.solver == solver,data)
            isempty(matches) && continue
            r = only(matches)
            @printf(io,"| %s | %s | %.2f%% | %.2f%% | %.4f | %.2f%% |\n",label,solver,
                100r.pressure_nrmse,100r.pressure_raw_nrmse,r.pressure_noise_rms,100r.viscous_nrmse)
        end
        println(io,"\nThe sound-speed cost is visible in step counts: native uses 45,600 / 90,800 / 181,200 steps at c0 = 40 / 80 / 160; Dual uses 46,342 / 91,506 / 181,892. Approximately four times the steps are needed when c0 increases fourfold. The repeated timing study below is at c0 = 40; exploratory sound-speed wall times in `summary.csv` are single runs, not paired performance estimates.\n")
        @printf(io,"\nThe original Hermite motion matches rounded positions and supplied velocities simultaneously, creating %.3f m/s² RMS and %.3f m/s² maximum acceleration during 2–7 s, where the reference acceleration is essentially zero. Integrating the supplied velocity table gives internally consistent position/velocity/acceleration, changes position by at most %.2f micrometres, and reduces that acceleration RMS to %.2e m/s².\n\n",
            motion["late_hermite_acceleration_rms"],motion["late_hermite_acceleration_max"],
            1e6motion["integrated_position_difference_max"],motion["late_integrated_acceleration_rms"])
        @printf(io,"This motion correction reduces native pressure-noise RMS by only %.2f%% at c0 = 40. It fixes an input defect but leaves the dominant acoustic ringing. Turning diffusion off raises native pressure-noise RMS from %.4f to %.4f, while Dual changes from %.4f to %.4f; the two then become very similar. With diffusion and regularization aligned, native RMS is %.4f versus Dual %.4f. These controls support diffusion treatment as a major cause of the original inter-solver noise difference.\n\n",
            100*(1-integrated.pressure_noise_rms/baseline.pressure_noise_rms),baseline.pressure_noise_rms,
            row("re100_no_ddt").pressure_noise_rms,row("re100_c40","dual").pressure_noise_rms,
            row("re100_no_ddt","dual").pressure_noise_rms,row("re100_dual_operators").pressure_noise_rms,
            row("re100_c40","dual").pressure_noise_rms)
        println(io,"Float64 retains the large ringing, so Float32 roundoff is not its main explanation in this test. Changing regularization alone has a much smaller effect than changing diffusion. The vendor force tool is also independently reproduced on two snapshots with the same pair equations: pressure and viscous drag agree within 0.003%. The large oscillations are present in the particle state rather than being introduced by a stale acceleration buffer or an incorrect force selection.\n")
        println(io,"![Motion diagnostic and control](results/comparison/motion_diagnostic.png)\n")
        println(io,"## Warm whole-solver timings\n")
        println(io,"Full-duration warmups precede three unprofiled pairs per Reynolds number, alternating native/Dual order. The operator-aligned Re = 100 timing is a separate three-pair run. Every measured run has 84,864 particles, 401 output frames and 8 s duration. Particle generation and offline force/plot processing are excluded. Stepping, solver setup, particle output and finalization are included; Dual also includes process startup. Julia host-particle allocation occurs before its timed solver call. Output formats differ (VTKHDF versus BI4). These compare practical solver configurations on this 2D case; they do not establish equal trajectories, equal neighbor-search work or a general kernel-throughput advantage.\n")
        println(io,"| Configuration | Julia median wall [s] | Dual median wall [s] | Median paired ratio (Dual/Julia) | Pair range | Steps Julia / Dual |")
        println(io,"|---|---:|---:|---:|---:|---:|")
        for (label,re,source) in (("Re 50, original",50,timing),("Re 100, original",100,timing),
                ("Re 150, original",150,timing),("Re 100, operators aligned",100,aligned_timing))
            a = sort(filter(r -> r.reynolds == re && r.solver == "native",source); by = r -> r.repetition)
            b = sort(filter(r -> r.reynolds == re && r.solver == "dual",source); by = r -> r.repetition)
            length(a) == length(b) == 3 || error("Missing timing pairs")
            ratios = [b[i].wall_seconds/a[i].wall_seconds for i in 1:3]
            @printf(io,"| %s | %.3f | %.3f | %.2fx | %.2f–%.2fx | %d / %d |\n",label,
                median(r.wall_seconds for r in a),median(r.wall_seconds for r in b),median(ratios),
                minimum(ratios),maximum(ratios),first(a).steps,first(b).steps)
        end
        all_timing = [timing;aligned_timing]
        @printf(io,"\nAcross repeats, maximum within-solver differences at exported precision are %.3e m in position, %.3e m/s in velocity and %.3e kg/m³ in density after sorting by particle ID. Counts and final states are checked outside the timed interval. Native rebuild/readback counts are recorded in the timing CSVs; corresponding Dual counters are not exposed here. Native CUDA graphs and infrequent neighbor rebuilds differ from Dual's execution policy. No claim of matched rebuild/readback work or memory-bandwidth saturation is made.\n\n",
            maximum(r.repeat_position_max for r in all_timing),maximum(r.repeat_velocity_max for r in all_timing),maximum(r.repeat_density_max for r in all_timing))
        println(io,"![Timing comparison](results/comparison/timing_comparison.png)\n")
        println(io,"## Assessment and proposed solution\n")
        println(io,"1. **Use consistent prescribed motion.** The integrated-velocity control is implemented and tested. It preserves the supplied velocity and removes acceleration introduced by rounded position knots. The original benchmark results are retained as provenance.\n2. **Improve and validate the moving-wall treatment.** Add translation/acceleration support for the existing Updated mDBC path, including mirrored no-slip velocity, pressure cloning and permitted negative wall pressure. Recheck both pressure and viscous reactions. The current DBC viscous error becomes particularly concerning at Re = 150; increased sound speed alone does not repair this. This boundary change remains a proposal, not a demonstrated fix.\n3. **Choose pressure treatment for incompressible internal flow.** Test pressure/divergence cleaning or an incompressible projection mode if direct transient loads are the priority. If retaining WCSPH, choose c0 by convergence of the force history and by acoustic versus acceleration time scales, with adequately resolved output. A higher c0 is expensive and did not validate this case by itself.\n4. **Validate the solution across Re = 50, 100, 150.** Preserve the physical viscosities, resolve the wall/shear layers, compare both drag components over the full transient, repeat spatial and temporal refinement, and retain raw histories. Filtering plots cannot substitute for a converged solution.\n")
        println(io,"A relevant next method is the acoustic-wave damping formulation in [Fourtakas, Vacondio and Rogers, divergence cleaning for WCSPH](https://arxiv.org/abs/2410.06038). It is published research implemented in a DualSPHysics research formulation; availability in the installed stock executable is not assumed. It needs a separate implementation and validation study here.\n")
        println(io,"These tests identify important contributors and rule out several simple explanations. They do not yet validate a replacement boundary/pressure method or the vorticity field. The velocity images are qualitative.\n")
        println(io,"## Reproduce and provenance\n")
        println(io,"From `gpu_version`, with the existing Julia environment and installed DualSPHysics tools (`DUALSPHYSICS_ROOT` overrides the local installation path):\n")
        println(io,"```sh\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/comparison_tests.jl\"\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/dual_compare.jl\"\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/dual_compare.jl\" timing\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/dual_compare.jl\" timing_aligned\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/motion_diagnostics.jl\"\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/export_comparison.jl\"\npython \"validation cases/spheric_test_06/plot_comparison.py\"\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/comparison_report.jl\"\n```\n")
        println(io,"`results/comparison/environment.toml` records software, hardware, the executable checksum and common settings. Small coefficient histories, spectra, checks and timing records are tracked; large particle histories, generated definitions and vendor exports remain under ignored `runs/`. No vendor binaries or source are redistributed. The final focused regression run passed all 220 assertions: 25 motion/spectral checks and the original 195 geometry, force and GPU/output checks. The full repository CUDA suite was not run. Official reference provenance remains in [reference/README.md](reference/README.md).")
    end
    println("Wrote COMPARISON.md")
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_comparison_report()
end
