using CSV, TOML, Printf, Statistics

function write_sound_speed_report()
    directory = joinpath(@__DIR__,"results","c160_comparison")
    rows = collect(CSV.File(joinpath(directory,"summary.csv")))
    timings = collect(CSV.File(joinpath(directory,"timings.csv")))
    settings = TOML.parsefile(joinpath(directory,"settings.toml"))
    length(rows) == 26 && length(timings) == 6 || error("Incomplete c160 comparison or timing pairs")
    row(re,variant,solver="native") = only(filter(r -> r.reynolds == re && r.variant == variant && r.solver == solver,rows))
    function state(re,variant)
        data = CSV.File(joinpath(@__DIR__,row(re,variant).data_path,"native_state.csv"))
        return (; density=maximum(data.density_relative_max),clearance=minimum(data.minimum_clearance),
            motion=maximum(data.motion_error),mach=maximum(data.max_speed)/row(re,variant).sound_speed)
    end
    open(joinpath(@__DIR__,"SOUND_SPEED_160.md"),"w") do io
        println(io,"# Sound speed 160 m/s and timestep controls in both solvers\n")
        println(io,"**Increasing c0 to 160 m/s reduces the measured pressure noise and raw pressure error relative to 40 m/s in both tested solvers at Re = 50, 100 and 150. It does not validate either DBC setup.** Smaller timesteps also change the result, and the controls below separate that effect from changing c0. All histories are unfiltered.\n")
        println(io,"## Why c0 and timestep both matter\n")
        println(io,"Near the reference density, the Tait equation gives Δp ≈ c0² Δρ, while the acoustic limit is approximately dt ≤ CFL h/c0. Going from 40 to 160 m/s makes the pressure response to a fixed density error 16 times stiffer and reduces the nominal acoustic timestep by four. It also changes acoustic travel times and the density-diffusion coefficient δ h c0. The fluid does not retain the same density error when c0 changes, so pressure noise need not vary monotonically with c0. This follows the [official DualSPHysics formulation](https://github.com/DualSPHysics/DualSPHysics/wiki/3.-SPH-formulation) and the equations implemented in this package.\n")
        println(io,"For a 10 m tank, the simple box-mode estimate c0/(2Lx) changes from 2 to 8 Hz. The moving square changes the actual modes; measured peaks are reported rather than assigned to an exact box eigenmode. Stable explicit stepping can retain bounded acoustic ringing. Stability therefore does not imply accurate incompressible wall forces.\n")
        println(io,"## Common setup and provenance\n")
        println(io,"The original closed 10 × 5 m tank and moving unit square are retained through 8 s: dp = 0.025 m, h = √2 dp, Uref = 1 m/s, ρ0 = 1000 kg/m³, zero gravity, Re = 50/100/150, laminar viscosity ν = 1/Re, Wendland C2, DBC, δ = 0.1, no shifting and no SPS. There are **no edge damping zones**. Both solvers have 78,400 fluid, 1,600 square and 4,864 fixed particles (84,864 total), Float32 physics and double positions, with 401 frames at 0.02 s intervals. Stock DualSPHysics 5.4.355 uses its x/z plane for Julia's physical x/y plane. GenCase lattices and particle types are verified against the common geometry.\n")
        println(io,"The original Hermite motion is retained. It includes acceleration associated with rounded position data; this experiment does not also change the forcing. The Julia reference density diffusion and regularization differ from stock Dual's wall-gated DDT and regularization, as detailed in [COMPARISON.md](COMPARISON.md). Equal settings do not make the two discrete operators identical.\n")
        println(io,"c0 = 40/CFL = 0.2 baselines and the Re = 100/c0 = 160/CFL = 0.2 pair reuse the previously published `9302adb` archive. All other controls are new. `summary.csv` identifies the source and relative data path of every row. Large source histories remain in ignored `runs/`; a fresh checkout must rerun the original comparison to recreate archive field images.\n")
        println(io,"## c0 = 40 versus 160 in both solvers\n")
        println(io,"Noise is the RMS of force error after subtracting its centered 0.5 s moving average, over 1.5–6.5 s. Raw NRMSE integrates squared piecewise-linear error against the official FD reference over 0.1–8 s. Both use the existing, tested `noise_metrics.jl`; no fitted scale or time shift is applied. The central noise window excludes startup, so full-transient raw errors and state diagnostics are retained.\n")
        println(io,"| Re | Solver | c0 [m/s] | Pressure noise RMS [Cd] | Raw pressure NRMSE | Binned pressure NRMSE | Raw viscous NRMSE | Steps |")
        println(io,"|---:|---|---:|---:|---:|---:|---:|---:|")
        for re in (50,100,150), solver in ("native","dual"), variant in ("baseline40","standard160")
            r = row(re,variant,solver)
            @printf(io,"| %d | %s | %d | %.4f | %.2f%% | %.2f%% | %.2f%% | %d |\n",re,solver,r.sound_speed,
                r.pressure_noise_rms,100r.pressure_raw_nrmse,100r.pressure_binned_nrmse,100r.viscous_raw_nrmse,r.steps)
        end
        println(io,"\n![Raw pressure drag](results/c160_comparison/pressure_drag.png)\n")
        println(io,"![Raw viscous drag](results/c160_comparison/viscous_drag.png)\n")
        println(io,"![Pressure drag zoom](results/c160_comparison/pressure_zoom.png)\n")
        println(io,"## Equal nominal acoustic timestep controls\n")
        println(io,"Both solvers also run c0 = 40 with CFL = 0.05. This matches the nominal acoustic cap of c0 = 160/CFL = 0.2: `0.05 h/40 = 0.2 h/160 ≈ 44.19 μs`. This is an adaptive-step control, not a shared imposed timestep sequence: force/viscous restrictions, output clipping and trajectories can differ. The measured step counts and 8 s/steps averages are reported. Keeping c0 fixed while lowering CFL probes timestep sensitivity; comparing the orange and teal curves then probes c0 at a similar acoustic cap.\n")
        println(io,"| Re | Solver | Control | Steps | Mean dt [μs] | Pressure noise RMS [Cd] | Raw pressure NRMSE |")
        println(io,"|---:|---|---|---:|---:|---:|---:|")
        for re in (50,100,150), solver in ("native","dual"), variant in ("step_control40","standard160")
            r = row(re,variant,solver)
            @printf(io,"| %d | %s | %s | %d | %.3f | %.4f | %.2f%% |\n",re,solver,
                variant == "step_control40" ? "40 m/s, CFL 0.05" : "160 m/s, CFL 0.2",
                r.steps,1e6r.mean_timestep,r.pressure_noise_rms,100r.pressure_raw_nrmse)
        end
        println(io,"\n![Noise and reference error](results/c160_comparison/noise_and_error.png)\n")
        for solver in ("native","dual"), re in (50,100,150)
            a,b,c = row(re,"baseline40",solver),row(re,"step_control40",solver),row(re,"standard160",solver)
            @printf(io,"At Re = %d in %s, reducing the timestep at c0 = 40 changes pressure noise by %+.1f%% and raw pressure NRMSE from %.2f%% to %.2f%%. At c0 = 160 with a similar cap, raw pressure NRMSE is %.2f%% and noise RMS is %.4f.\n\n",
                re,solver,100*(b.pressure_noise_rms/a.pressure_noise_rms-1),100a.pressure_raw_nrmse,
                100b.pressure_raw_nrmse,100c.pressure_raw_nrmse,c.pressure_noise_rms)
        end
        println(io,"The two changes are not equivalent. A smaller timestep alone does not reproduce the higher-c0 pressure history. These are timestep-sensitivity controls, not a formal convergence study; further dt and spatial refinement are still required. At Re = 100, the earlier Julia c0 = 80 noise RMS was 0.2818 versus 0.3804 at 160, so increasing c0 is not a monotonic noise-removal knob.\n")
        println(io,"## Optional Julia GPU cleaning at 160 m/s\n")
        println(io,"This pair uses the new optional cleaning procedure with κ = c0/Uref = 160, σ = 1, cleaning Courant = 0.1, and the documented `pressure_mode=:absolute` extension. Both on and off use physical CFL = 0.1, so the smaller physical CFL is controlled. Cleaning's accepted-state decay-speed restriction can still require extra steps. The stock DualSPHysics binary is unchanged and has no cleaning result in this study.\n")
        println(io,"| Re | Mode | Pressure noise RMS [Cd] | Raw pressure NRMSE | Raw viscous NRMSE | Max density departure | Steps |")
        println(io,"|---:|---|---:|---:|---:|---:|---:|")
        for re in (50,100,150), variant in ("cleaning_control160","cleaning160")
            r,s = row(re,variant),state(re,variant)
            @printf(io,"| %d | %s | %.4f | %.2f%% | %.2f%% | %.3f%% | %d |\n",re,
                variant == "cleaning160" ? "on" : "off",r.pressure_noise_rms,100r.pressure_raw_nrmse,
                100r.viscous_raw_nrmse,100s.density,r.steps)
        end
        println(io,"\n![Cleaning at 160 m/s](results/c160_comparison/cleaning_drag.png)\n")
        println(io,"The κ = 160 control reduces central-window noise but worsens full-transient pressure error at every Re, including a large negative startup excursion. It is not recommended as a validation fix. Re = 100 also checks the API default κ = 10 and an intermediate κ = 40 at the same physical and cleaning CFL:\n")
        println(io,"| Re = 100 mode | Pressure noise RMS [Cd] | Raw pressure NRMSE | Raw viscous NRMSE | Max density departure | Steps |")
        println(io,"|---|---:|---:|---:|---:|---:|")
        for (variant,label) in (("cleaning_control160","off"),("cleaning_k10_160","κ = 10"),
                ("cleaning_k40_160","κ = 40"),("cleaning160","κ = 160"))
            r,s = row(100,variant),state(100,variant)
            @printf(io,"| %s | %.4f | %.2f%% | %.2f%% | %.3f%% | %d |\n",label,r.pressure_noise_rms,
                100r.pressure_raw_nrmse,100r.viscous_raw_nrmse,100s.density,r.steps)
        end
        println(io,"\n![Cleaning decay sensitivity at 160 m/s](results/c160_comparison/cleaning_sensitivity.png)\n")
        println(io,"At Re = 100, κ = 10 gives a startup Cd,p minimum near −37.66 at t = 0.74 s and raw pressure NRMSE of about 256.3%, despite only modest central-window noise change. κ = 40 and 160 also worsen total pressure error relative to the same-CFL off control. None of these settings is a validated remedy. The scalar-gradient term changes the transient dynamics; a formal timestep-convergence and moving-wall study is still needed to separate integration, operator and boundary effects. For this benchmark at 160 m/s, leave cleaning off pending that work.\n")
        println(io,"The [cleaning implementation report](DIVERGENCE_CLEANING.md) contains equations, sign conventions, GPU allocation, supported configurations and the c0 = 40 controls. Its negative-pressure experiment also shows why a quiet central window alone is insufficient evidence of accuracy.\n")
        println(io,"## Current fields and state checks\n")
        println(io,"![Current c0 = 160 velocity fields](results/c160_comparison/velocity_fields.png)\n")
        println(io,"| Re | Julia mode | Max density departure | Max Mach | Minimum solid clearance [m] | Max motion error [m] |")
        println(io,"|---:|---|---:|---:|---:|---:|")
        for re in (50,100,150), variant in ("standard160","step_control40","cleaning_control160","cleaning160")
            s = state(re,variant)
            @printf(io,"| %d | %s | %.3f%% | %.4f | %.5f | %.2e |\n",re,variant,100s.density,s.mach,s.clearance,s.motion)
        end
        println(io,"\nAll new Julia frames were checked for finite state, particle counts, prescribed motion and no frozen particles. Dual runs rejected any excluded particles and completed all 401 frames with 84,864 particles. The full Julia diagnostics are in each `native_state.csv`; the same full-history state diagnostics were not exported for Dual. Negative clearance means fluid centers enter the nominal solid and fails the original impermeability criterion. Successful completion is not a validation pass.\n")
        println(io,"At the 0.02 s output cadence, Nyquist is 25 Hz. The 8 Hz box estimate has only 6.25 output samples per cycle; higher modes may alias. This limits spectral interpretation and means the saved-force noise metric does not measure every high-frequency fluctuation. A higher-cadence force diagnostic is needed before assigning unresolved spectral peaks.\n")
        println(io,"## Isolated whole-solver timing at Re = 100\n")
        a = sort(filter(r -> r.solver == "native",timings); by=r -> r.repetition)
        b = sort(filter(r -> r.solver == "dual",timings); by=r -> r.repetition)
        ratios = [b[i].wall_seconds/a[i].wall_seconds for i in 1:3]
        @printf(io,"On %s, c0 = 160/CFL = 0.2 uses full-duration warmups of each solver followed by three alternating, unprofiled pairs with no other validation job using the GPU. The median paired Dual/Julia wall-time ratio is %.2fx (range %.2f–%.2fx). These are whole configuration costs including setup, stepping and output, not matched-operator kernel timings.\n\n",settings["gpu"],median(ratios),minimum(ratios),maximum(ratios))
        println(io,"| Solver | Median wall [s] | Median reported solver [s] | Steps | Frames | Julia rebuilds | Julia step readbacks |")
        println(io,"|---|---:|---:|---:|---:|---:|---:|")
        for (solver,group) in (("native",a),("dual",b))
            @printf(io,"| %s | %.3f | %.3f | %d | %d | %s | %s |\n",solver,
                median(r.wall_seconds for r in group),median(r.solver_seconds for r in group),first(group).steps,
                first(group).frames,ismissing(first(group).rebuilds) ? "n/a" : string(first(group).rebuilds),
                ismissing(first(group).readbacks) ? "n/a" : string(first(group).readbacks))
        end
        println(io,"\nTimings exclude GenCase/Julia particle generation and offline force/plot analysis. Repeats compare final position, velocity and density by stable ID; all differences are retained in `timings.csv`. The exploratory per-case wall times in `summary.csv` overlapped regressions and are not used as performance evidence. No whole-solver VRAM percentage or memory-bandwidth claim is inferred.\n")
        println(io,"| Solver | Max repeat position difference [m] | Max repeat velocity difference [m/s] | Max repeat density difference [kg/m³] |")
        println(io,"|---|---:|---:|---:|")
        for (solver,group) in (("native",a),("dual",b))
            @printf(io,"| %s | %.3e | %.3e | %.3e |\n",solver,maximum(r.repeat_position_max for r in group),
                maximum(r.repeat_velocity_max for r in group),maximum(r.repeat_density_max for r in group))
        end
        println(io,"\nThese comparisons use exported precision, including stock Dual's CSV formatting. Matching final state does not establish that every intermediate floating-point operation is identical.\n")
        println(io,"## Assessment and next step\n")
        println(io,"c0 = 160 is a useful stiffness/transit-time control, but it is not a cure for wall-traction error. The comparison should retain raw pressure and viscous loads, central noise, startup transients and impermeability checks together. The current evidence supports keeping cleaning optional and testing consistent integrated-velocity motion and translating/accelerating no-slip wall treatments next, followed by spatial and timestep refinement. A lower noise RMS without better reference loads is not a successful validation.\n")
        println(io,"## Reproduce\n")
        @printf(io,"Julia %s, CUDA.jl %s, CUDA driver %s; GPU %s. No new dependency or modified DualSPHysics binary is required.\n\n",
            settings["julia"],settings["cuda_jl"],settings["cuda_driver"],settings["gpu"])
        println(io,"Run from `gpu_version/`; first run `dual_compare.jl` to recreate missing archived baselines. Run timing studies alone after the simulation/regression jobs finish:\n")
        println(io,"```sh\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/sound_speed_160.jl\"\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/sound_speed_160.jl\" fields\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/sound_speed_160.jl\" timing\npython \"validation cases/spheric_test_06/plot_sound_speed_160.py\"\njulia --project=. \"validation cases/spheric_test_06/sound_speed_160_report.jl\"\n```\n")
        print(io,"The script resumes saved `.toml` results and refuses to overwrite existing run histories. `native` and `dual` arguments select an individual solver. Run those modes sequentially if they may write the same case or summary file.\n")
    end
end

write_sound_speed_report()
