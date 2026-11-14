using CSV, TOML, Printf, Statistics

function write_edge_damping_report()
    directory = joinpath(@__DIR__,"results","edge_damping")
    rows = collect(CSV.File(joinpath(directory,"summary.csv")))
    baseline = collect(CSV.File(joinpath(@__DIR__,"results","comparison","summary.csv")))
    checks = collect(CSV.File(joinpath(directory,"state_checks.csv")))
    settings = TOML.parsefile(joinpath(directory,"settings.toml"))
    geometry = TOML.parsefile(joinpath(directory,"geometry.toml"))
    length(rows) == 12 && length(checks) == 6 || error("Incomplete damping experiment")
    base(re,solver) = only(filter(r -> r.case == "re$(re)_c40" && r.solver == solver,baseline))
    selected(rate,solver) = filter(r -> r.rate == rate && r.solver == solver,rows)
    open(joinpath(@__DIR__,"EDGE_DAMPING.md"),"w") do io
        println(io,"# Four-edge damping in Julia GPU and DualSPHysics\n")
        println(io,"Completed twelve new full-duration simulations: Re = 50, 100 and 150, two damping strengths and both solvers. Every run retained 84,864 particles and wrote 401 frames through approximately 8 s. **The sponges reduce pressure ringing but change the physical flow and do not validate the original SPHERIC Test 06.**\n")
        println(io,"## What was tested\n")
        println(io,"The tank is [0,10] × [0,5] m. Each of its four outer edges has a 0.5 m-wide damping band. The undamped core is [0.5,9.5] × [0.5,4.5] m. A single box sponge in each solver avoids applying multiple independent velocity reductions at corners. The outer limits are [0,10] × [0,5] m, with a 0.1 m overlimit outside the nominal tank. The active faces are left, right, bottom and top.\n")
        println(io,"Both physical velocity components are damped, using β = 10 or 100 s⁻¹ and the same quadratic spatial profile. Only fluid particles are affected. The square's prescribed velocity, tank particles and particle density are not directly damped. Julia uses `DampingBox` in x/y; Dual uses `<dampingbox>` in its x/z plane with `factorxyz=(1,0,1)`. For normalized penetration into the damping band s in [0,1], the accepted-step velocity update is:\n")
        println(io,"```text\ns = max(0, (0.5-x)/0.5, (x-9.5)/0.5, (0.5-y)/0.5, (y-4.5)/0.5), clamped to 1\nv_after = max(0, 1 - dt * β * s^2) * v_before\n```\n")
        println(io,"The overlimit extends the full-strength region beyond the outer box. At corners, the largest face penetration is used rather than a sum. The installed Dual source's `KerComputeDampingBox` in `JSphGpu_ker.cu` confirms the maximum-face rule, quadratic ramp, normal-fluid gate and nonnegative component factors. The [official passive-absorption formulation](https://github.com/DualSPHysics/DualSPHysics/wiki/3.-SPH-formulation#3101-passive-wave-absorption) describes this velocity reduction. Each result directory includes the actual configuration reported by Dual and its generated damping XML.\n")
        @printf(io,"The initial sponge contains %d of %d fluid particles (%.1f%%). The square never physically enters it, but its final right face is only %.3f mm from the band, compared with the %.3f mm kernel support radius. Near the end, fluid interacting with the square can therefore be affected directly by the sponge. Being outside the bands does not isolate the square from changes to the tank-wide flow.\n\n",
            geometry["initial_sponge_fluid_particles"],geometry["initial_fluid_particles"],
            100geometry["initial_sponge_fluid_particles"]/geometry["initial_fluid_particles"],
            1000geometry["square_to_sponge_final_gap"],1000geometry["kernel_support_radius"])
        println(io,"![Four-edge layout and damping profile](results/edge_damping/zone_layout.png)\n")
        println(io,"Everything else is retained from the undamped comparison: dp = 0.025 m, L/dp = 40, c0 = 40 m/s, rho0 = 1000 kg/m³, g = 0, h = √2 dp, Wendland C2, CFL = 0.2, symplectic stepping, DBC, laminar ν = 1/Re, no SPS and no shifting. Both start with the same 78,400 fluid, 1,600 moving-square and 4,864 tank particles. Positions and the clock use Float64; physics fields use Float32. The original Hermite motion is retained to isolate damping. Existing diffusion and viscosity-regularization differences between the solvers are retained and documented in [COMPARISON.md](COMPARISON.md). Compare each damped run to its own solver's undamped baseline.\n")
        println(io,"## Noise versus changes in drag\n")
        println(io,"Noise RMS uses the same definition as the previous study: form the pressure-drag error against the original finite-difference reference, subtract its centered 0.5 s moving average and compute RMS over 1.5–6.5 s on a 0.02 s grid. This is a reproducible oscillation metric, not an exact acoustic/physical-mode decomposition. A positive noise reduction means quieter than the same solver's undamped run. Mean-drag changes are ratios of time integrals over 2–6.5 s and 4–8 s. No filtering is applied to the saved forces or the raw plots.\n")
        println(io,"| Re | Solver | β [s⁻¹] | Pressure noise RMS [Cd] | Noise reduction | Mean pressure drag change, 2–6.5 s | Mean pressure drag change, 4–8 s |")
        println(io,"|---:|---|---:|---:|---:|---:|---:|")
        for r in sort(rows; by = r -> (r.reynolds,r.solver,r.rate))
            @printf(io,"| %d | %s | %d | %.4f | %.1f%% | %+.1f%% | %+.1f%% |\n",r.reynolds,r.solver,r.rate,
                r.pressure_noise_rms,100r.pressure_noise_reduction,100r.pressure_mean_change_central,100r.pressure_mean_change_late)
        end
        println(io,"\n![Noise reduction and mean load change](results/edge_damping/noise_and_bias.png)\n")
        for solver in ("native","dual"), rate in (10,100)
            group = selected(rate,solver)
            @printf(io,"For %s at β = %d, pressure-noise reduction spans %.1f–%.1f%% across the three Re values, while central-interval mean pressure drag changes by %+.1f–%+.1f%%.\n\n",
                solver,rate,100minimum(r.pressure_noise_reduction for r in group),100maximum(r.pressure_noise_reduction for r in group),
                100minimum(r.pressure_mean_change_central for r in group),100maximum(r.pressure_mean_change_central for r in group))
        end
        println(io,"![Raw pressure drag, all Re and both solvers](results/edge_damping/pressure_drag.png)\n")
        println(io,"![Pressure-drag error spectra](results/edge_damping/pressure_spectra.png)\n")
        println(io,"The spectrum uses the same 1.5–6.5 s interval, removes a linear drift and applies a Hann window. The dashed 2 Hz line is c0/(2Lx), a simple tank acoustic estimate. The finite record resolves about 0.20 Hz and output Nyquist is 25 Hz. Reduced peaks support reflected/compressible tank modes as contributors to the noise. The sponge also acts on the physical return flow, so this experiment alone cannot attribute all changes to acoustic reflections.\n")
        println(io,"## Reference error and viscous drag\n")
        println(io,"Raw NRMSE is the square root of the exact time-integrated squared difference between piecewise-linear force histories, normalized by the reference energy over 0.1–8 s. Binned NRMSE compares common 0.1 s time averages. Both are shown because averaging can hide the oscillations. The FD reference is for the original undamped equations; errors below quantify departure from that target, not validation of a separate damped problem.\n")
        println(io,"| Re | Solver | β [s⁻¹] | Raw pressure NRMSE | Binned pressure NRMSE | Raw viscous NRMSE | Binned viscous NRMSE |")
        println(io,"|---:|---|---:|---:|---:|---:|---:|")
        for re in (50,100,150), solver in ("native","dual")
            b = base(re,solver)
            @printf(io,"| %d | %s | 0 | %.2f%% | %.2f%% | %.2f%% | %.2f%% |\n",re,solver,
                100b.pressure_raw_nrmse,100b.pressure_nrmse,100b.viscous_raw_nrmse,100b.viscous_nrmse)
            for r in sort(filter(r -> r.reynolds == re && r.solver == solver,rows); by = r -> r.rate)
                @printf(io,"| %d | %s | %d | %.2f%% | %.2f%% | %.2f%% | %.2f%% |\n",re,solver,r.rate,
                    100r.pressure_raw_nrmse,100r.pressure_binned_nrmse,100r.viscous_raw_nrmse,100r.viscous_binned_nrmse)
            end
        end
        println(io,"\n![Raw viscous drag](results/edge_damping/viscous_drag.png)\n")
        println(io,"![Final fields at all three Re values](results/edge_damping/fields_beta100.png)\n")
        println(io,"The final-state panels use one common speed scale and approximately 8 s snapshots. The dashed contour is the inner sponge boundary. They show the computed particle velocities and are qualitative; vorticity or wall-shear accuracy has not been established.\n")
        println(io,"## Assessment and proposed solution\n")
        println(io,"Damping the four tank edges is a useful diagnostic: the oscillation level responds in both independent solvers, consistent with the earlier sound-speed and frequency controls. It does not demonstrate that all noise originates at the outer walls. Particle disorder, local boundary pressure treatment and the moving body can continue to generate compressible disturbances.\n")
        println(io,"For this closed-tank benchmark, an all-component velocity sponge adds an artificial momentum sink to a region occupying 28.6% of the initial fluid. It damps genuine recirculation along with numerical waves. The nominal ν and Re remain unchanged, but the governing problem has an additional force. A smoother curve is therefore insufficient evidence of improved physical accuracy. The mean-load and reference-error tables show the cost directly.\n")
        println(io,"Keep these cases as a separate diagnostic family. For the actual Test 06 solution, prioritize consistent prescribed motion, translation/acceleration-aware no-slip moving-wall treatment, and a pressure method suited to incompressible confined flow. A pressure/divergence-cleaning control that damps compressible error while preserving the mean flow, or an incompressible projection formulation, is a more relevant next comparison than stronger velocity sponges. Those are proposed studies, not demonstrated fixes. Retest both drag components, the wall layers and the velocity field across all three Re values with refinement.\n")
        println(io,"If an absorbing exterior is desired for a different physical problem, define that domain and reference solution explicitly, place the sponge farther from the body and establish width/strength/domain-size convergence. This run does not establish an optimum strength or reflection coefficient.\n")
        println(io,"## Completion, cost and reproducibility\n")
        @printf(io,"Native saved histories have maximum relative density departure %.4f%%, maximum Mach %.4f, maximum motion error %.2e m and maximum prescribed-velocity error %.2e m/s. Minimum fluid-center clearance to a nominal solid is %.5f m. These state checks describe the native full history; Dual's full-history checks verify unchanged normal-particle counts and zero exclusions, while exported final densities are finite. They do not establish impermeability or density bounds throughout the Dual histories.\n\n",
            100maximum(r.native_maximum_density_deviation for r in checks),maximum(r.native_maximum_mach for r in checks),
            maximum(r.native_maximum_motion_error for r in checks),maximum(r.native_maximum_velocity_error for r in checks),minimum(r.native_minimum_solid_clearance for r in checks))
        if minimum(r.native_minimum_solid_clearance for r in checks) < 0
            println(io,"The negative minimum clearance means some fluid-particle centers enter a nominal solid. Four-edge damping has not repaired the existing DBC impermeability error. No new run froze particles or stopped early, but these completion checks must not be confused with physical validation.\n")
        end
        println(io,"| Re | β [s⁻¹] | Julia steps | Dual steps | Julia wall [s] | Dual wall [s] |")
        println(io,"|---:|---:|---:|---:|---:|---:|")
        for re in (50,100,150), rate in (10,100)
            a = only(filter(r -> r.reynolds == re && r.rate == rate && r.solver == "native",rows))
            b = only(filter(r -> r.reynolds == re && r.rate == rate && r.solver == "dual",rows))
            @printf(io,"| %d | %d | %d | %d | %.3f | %.3f |\n",re,rate,a.steps,b.steps,a.wall_seconds,b.wall_seconds)
        end
        println(io,"\nThese are single exploratory solver calls, excluding offline force/plot analysis. The first native call includes cold compilation, and Dual includes process startup. They do not measure a paired speedup or a damping-overhead percentage. The earlier warm, alternating, three-pair speed study remains in [COMPARISON.md](COMPARISON.md). Native uses CUDA graphs and less frequent neighbor-list rebuilds; output formats differ.\n")
        @printf(io,"Environment: %s, Julia %s, CUDA.jl %s; DualSPHysics %s. The unchanged solver library and undamped baseline come from commit `%s`. The harness changes add damping configuration and composite-history support. Settings and executable SHA-256 are in `results/edge_damping/settings.toml`. All large histories, binaries and local logs remain in ignored `runs/` directories.\n\n",
            settings["gpu"],settings["julia"],settings["cuda_jl"],settings["dual_version"],settings["baseline_commit"])
        println(io,"Run from `gpu_version/` with the existing Julia project and a CUDA GPU. Set `DUALSPHYSICS_ROOT` to the installed DualSPHysics 5.4 directory when needed:\n")
        println(io,"```sh\njulia --project=. -t 2,0 test/damping.jl\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/edge_damping_tests.jl\"\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/test.jl\"\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/comparison_tests.jl\"\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/edge_damping.jl\"\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/export_edge_damping.jl\"\npython \"validation cases/spheric_test_06/plot_edge_damping.py\"\njulia --project=. \"validation cases/spheric_test_06/edge_damping_report.jl\"\n```\n")
        print(io,"The study resumes completed variants, validates each new initial lattice and refuses to overwrite existing histories. Remove or relocate only the appropriate local result/run directories if deliberately rerunning a variant. The plotting step requires the same optional matplotlib, NumPy and h5py environment used for the earlier figures. No package dependencies were changed. The new focused tests cover four edges/corners, protected particle types, square clearance, x/y ↔ x/z mapping, Float32/Float64 CUDA updates and an end-to-end composite VTKHDF history. All 349 focused assertions passed: 104 existing damping assertions, 25 new edge-damping assertions, 195 original Test 06 assertions and 25 comparison assertions. The full CUDA suite was not run for these harness-only changes.\n")
    end
end

write_edge_damping_report()
