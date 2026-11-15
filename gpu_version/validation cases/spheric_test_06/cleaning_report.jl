using CSV, TOML, Printf, Statistics

function write_cleaning_report()
    directory = joinpath(@__DIR__,"results","divergence_cleaning")
    rows = collect(CSV.File(joinpath(directory,"summary.csv")))
    timings = collect(CSV.File(joinpath(directory,"timings.csv")))
    divergence = collect(CSV.File(joinpath(directory,"divergence.csv")))
    settings = TOML.parsefile(joinpath(directory,"settings.toml"))
    length(rows) == 11 && length(timings) == 6 || error("Incomplete cleaning study or timing pairs")
    row(re,variant) = only(filter(r -> r.reynolds == re && r.variant == variant,rows))
    function mean_divergence(re,variant)
        selected = filter(r -> r.reynolds == re && r.variant == variant && 1.5 <= r.time <= 6.5,divergence)
        return sqrt(mean(r.divergence_volume_rms^2 for r in selected))
    end
    open(joinpath(@__DIR__,"DIVERGENCE_CLEANING.md"),"w") do io
        println(io,"# Optional GPU velocity-divergence cleaning\n")
        println(io,"Implemented as `RunSimulation(SimDivergenceCleaning = DivergenceCleaning(...))` in the Julia CUDA package. It is experimental and off by default. The implementation follows the ψ-based hyperbolic/parabolic procedure of [Fourtakas, Vacondio and Rogers (2025)](https://doi.org/10.1016/j.compfluid.2025.106638), using the published paper supplied by the user. Eleven complete moving-square controls were run, including two extensions for negative gauge pressure, plus warm alternating timing pairs. **The default extension gives modest noise reduction and does not validate this DBC benchmark. The alternative zero-decay treatment in rarefactions produced large startup errors.**\n")
        println(io,"## Equations and explicit implementation choices\n")
        println(io,"```text\nD u_i / Dt = original SPH acceleration + Σ_j (ψ_i − ψ_j) ∇W_ij (m_j/ρ_j)\nD ψ_i / Dt = c0² Σ_j (u_i − u_j) · ∇W_ij (m_j/ρ_j) − ψ_i/τ_i\n1/τ_i = σ κ sqrt(abs(p_i)/ρ_i) / h\ndt ≤ C_clean h / max(c0, max_i(κ sqrt(abs(p_i)/ρ_i)))\n```\n")
        println(io,"The summation for ψ is the raw velocity difference divergence. It excludes every density-diffusion term. Both ψ and its derivative are evaluated at the same current or predictor state as the pressure/viscosity terms. ψ starts at zero and is advanced with the existing two-stage integrator: ψ_half = ψ_n + dt/2 RHS_n, then ψ_(n+1) = ψ_n + dt RHS_half. Fluid particles evolve ψ; fixed, prescribed and frozen particles have ψ = 0, following the paper's wall Dirichlet condition.\n")
        println(io,"The published Eq. 3.4 has a positive ψ/τ term on the RHS. This contradicts Eq. 3.2, the damped wave equation and the stated exponential decay. The implementation uses the negative, dissipative sign. A source-free decay regression checks this sign and the second-order scalar update.\n")
        println(io,"The paper's sqrt(p/ρ) is not real for negative gauge pressure. Here `pressure_mode=:absolute` (default) uses pressure magnitude only in the local decay speed. The alternative `:positive` uses max(p,0), leaving negative-pressure particles with hyperbolic transport but zero local decay. Neither clips actual pressure or density. Both are documented extensions; the paper does not specify this negative-pressure treatment. The default retains decay in rarefactions and avoided the large startup excursions seen with the alternative. The gradient is the paper's nonconservative difference operator, so exact momentum conservation or a pressure projection is not claimed.\n")
        println(io,"## Why negative-pressure handling is explicit\n")
        println(io,"The central noise window excludes the initial acceleration. With max(p,0), that window can look much quieter while the full transient becomes inaccurate. These are complete runs at the same Re, resolution and physical/cleaning CFL; only the decay-speed extension differs:\n")
        println(io,"| Re | Extension | Noise reduction, 1.5–6.5 s | Raw pressure NRMSE, 0.1–8 s | Max density departure |")
        println(io,"|---:|---|---:|---:|---:|")
        for re in (50,100,150), variant in ("positive","on")
            r = row(re,variant)
            @printf(io,"| %d | %s | %.1f%% | %.2f%% | %.2f%% |\n",re,r.pressure_mode,
                100r.pressure_noise_reduction,100r.pressure_raw_nrmse,100r.maximum_density_deviation)
        end
        println(io,"\nAt Re = 150 the positive-only treatment produces a pressure Cd of about −22.43 at t = 0.64 s and about 10.05% peak density deviation. Those startup excursions are outside the noise window. The default absolute-pressure result has about 1.10% peak density deviation and remains an imperfect DBC solution. Both histories and the κ = 10 controls are retained; the failed extension is not discarded from the evidence.\n")
        println(io,"![Negative-pressure extension controls](results/divergence_cleaning/pressure_extensions.png)\n")
        println(io,"## GPU execution and allocation\n")
        println(io,"The ψ gradient and velocity divergence are fused into the existing neighbor traversals, with gather lanes and no atomics. The predictor and corrector scalar updates are fused into particle integration, including eligible fused stationary predictors. A signal-speed reduction is fused into accepted-state integration and consumed by the existing device timestep selector; only initialization/rebuild refreshes need an extra elementwise reduction. There is no extra neighbor search or per-step host readback for cleaning. The physical acceleration and existing acoustic restrictions still apply.\n")
        println(io,"Three N-element working-precision device scalars hold ψ, ψ_half and its derivative. The predictor supplies idle permutation scratch at rebuilds, then ψ is copied back before that scratch is used for prediction. The scalar therefore follows stable particle IDs without persistent-ID indirection in interactions. The signal workspace adds one scalar per reduction block; its small host partial buffer is separate. At N = 84,864 in Float32 this adds 1,019,696 allocated device-buffer bytes (about 0.9725 MiB), excluding allocator reservations, program state and register spills. The final result downloads one N-scalar host ψ array in particle order. Disabled mode allocates none of these buffers and removes the cleaning branches through specialization.\n")
        println(io,"The initial public implementation supports standard symplectic stepping in 2D/3D, Float32/Float64, DBC/mDBC, prescribed motion and translated periodic queries. Unsupported configurations fail explicitly: single-neighbor/reference schemes, dense periodic images, variable resolution, open boundaries, floating reactions, timestep-provenance diagnostics and nonzero-clock restarts. A restart needs persistent ψ as well as particle state. See the GPU README and `DivergenceCleaning` docstring for usage and limitations.\n")
        println(io,"## Moving-square controls\n")
        println(io,"All runs retain the original 10 × 5 m closed tank, moving unit square, original Hermite motion, Re = 50/100/150 physical viscosities, c0 = 40 m/s, dp = 0.025 m, h = √2 dp, DBC, no SPS, no shifting and zero-gravity linear density diffusion δ = 0.1. They use Float64 positions/clock, Float32 physics, strict reference interaction arithmetic and R = Rx = 1. All have 84,864 particles and 401 frames over 8 s. **There are no edge sponges.**\n")
        println(io,"Both off and on runs use physical CFL = 0.1. On additionally uses C_clean = 0.1, σ = 1 and κ = c0/Uref = 40 for the nominal Uref = 1 m/s. The original CFL = 0.2 histories are shown separately in the raw-force figure, so a smaller timestep is not conflated with cleaning. A Re = 100 sensitivity run uses the API default κ = 10, corresponding to the usual nominal Mach 0.1 convention in the paper. No strength is selected by optimizing these reference errors.\n")
        println(io,"Noise is RMS of the drag error after subtracting a centered 0.5 s moving average, sampled over 1.5–6.5 s. Raw NRMSE integrates squared piecewise-linear error against the official FD reference over 0.1–8 s. Binned NRMSE compares common 0.1 s averages. Force histories are unfiltered. Mean-load changes below use time integrals over 2–6.5 s relative to the same-Re off/CFL-0.1 control.\n")
        println(io,"| Re | Mode | Pressure noise RMS [Cd] | Noise reduction | Mean pressure change | Raw pressure NRMSE | Binned pressure NRMSE | Raw viscous NRMSE |")
        println(io,"|---:|---|---:|---:|---:|---:|---:|---:|")
        for re in (50,100,150), variant in ("off","on","k10")
            re != 100 && variant == "k10" && continue
            r = row(re,variant)
            @printf(io,"| %d | %s | %.4f | %.1f%% | %+.1f%% | %.2f%% | %.2f%% | %.2f%% |\n",re,variant,
                r.pressure_noise_rms,100r.pressure_noise_reduction,100r.pressure_mean_change_central,
                100r.pressure_raw_nrmse,100r.pressure_binned_nrmse,100r.viscous_raw_nrmse)
        end
        println(io,"\n![Raw moving-square forces](results/divergence_cleaning/drag_comparison.png)\n")
        println(io,"![Raw pressure drag in the noise window](results/divergence_cleaning/pressure_zoom.png)\n")
        println(io,"![Kappa sensitivity and error spectra](results/divergence_cleaning/kappa_sensitivity.png)\n")
        println(io,"## Measured velocity divergence\n")
        println(io,"An offline Float64 GPU evaluation applies the raw volume-weighted velocity-difference operator directly to saved positions, velocities and densities, including wall velocities. Density diffusion is disabled for this diagnostic. It does not infer div(u) from the solver's total density derivative. Snapshots are evaluated every 0.2 s. The aggregate below is RMS of snapshot volume-weighted RMS values over the selected 1.6–6.4 s samples. This includes near-wall regions and does not establish convergence of the divergence operator.\n")
        println(io,"| Re | Off div(u) RMS [s⁻¹] | On div(u) RMS [s⁻¹] | Reduction |")
        println(io,"|---:|---:|---:|---:|")
        for re in (50,100,150)
            a,b = mean_divergence(re,"off"),mean_divergence(re,"on")
            @printf(io,"| %d | %.5f | %.5f | %.1f%% |\n",re,a,b,100*(1-b/a))
        end
        println(io,"\n![Measured divergence history](results/divergence_cleaning/divergence_rms.png)\n")
        println(io,"![Final velocity fields](results/divergence_cleaning/velocity_fields.png)\n")
        println(io,"## Cost and repeatability\n")
        a = sort(filter(r -> r.variant == "off",timings); by=r -> r.repetition)
        b = sort(filter(r -> r.variant == "on",timings); by=r -> r.repetition)
        ratios = [b[i].wall_seconds/a[i].wall_seconds for i in 1:3]
        @printf(io,"Re = 100 uses full-duration warmups of both modes in one process, followed by three unprofiled alternating off/on pairs. Median whole-solver wall time is %.3f s off and %.3f s on; median paired cost ratio is %.3fx (range %.3f–%.3fx). This includes stepping, setup, 401 output frames and final downloads, while excluding particle generation and offline force/plot analysis. It is a configuration cost comparison, not an isolated kernel-overhead estimate.\n\n",
            median(r.wall_seconds for r in a),median(r.wall_seconds for r in b),median(ratios),minimum(ratios),maximum(ratios))
        println(io,"| Mode | Steps | Rebuilds | Step-state readbacks | Median wall [s] |")
        println(io,"|---|---:|---:|---:|---:|")
        for (variant,group) in (("off",a),("on",b))
            @printf(io,"| %s | %d | %d | %d | %.3f |\n",variant,first(group).steps,first(group).rebuilds,first(group).readbacks,
                median(r.wall_seconds for r in group))
        end
        println(io,"\nThe local cleaning speed can require extra accepted steps, and physical trajectories change. Rebuild/readback counts are reported rather than assumed equal. Timing repeats verify identical position, velocity and density within each mode at exported precision; every recorded maximum repeat-state difference is zero. Exploratory study wall times overlapped independent regressions and are not used for performance claims; the timing pairs run separately after those jobs finish. No hardware-bandwidth claim or whole-solver VRAM percentage is inferred from these timings.\n")
        println(io,"## Assessment\n")
        for re in (50,100,150)
            r = row(re,"on")
            @printf(io,"At Re = %d, cleaning changes pressure-noise RMS by %+.1f%%, central mean pressure drag by %+.1f%%, and raw pressure NRMSE from %.2f%% to %.2f%%. Raw viscous NRMSE changes from %.2f%% to %.2f%%.\n\n",re,
                -100r.pressure_noise_reduction,100r.pressure_mean_change_central,100row(re,"off").pressure_raw_nrmse,
                100r.pressure_raw_nrmse,100row(re,"off").viscous_raw_nrmse,100r.viscous_raw_nrmse)
        end
        println(io,"The method acts on compressible velocity error throughout the domain, unlike a velocity sponge confined to the tank edges. Its numerical behavior must still be judged by force bias, physical viscosity, resolution and boundary errors. A quieter history by itself is insufficient. The current DBC treatment and rounded-motion interpolation are retained controls; translating/accelerating no-slip wall improvements and internally consistent motion remain relevant next steps. Neither an exact incompressible projection nor validated wall shear is claimed.\n")
        println(io,"| Re | Mode | Max density departure | Max Mach | Minimum solid clearance [m] | Steps |")
        println(io,"|---:|---|---:|---:|---:|---:|")
        for r in rows
            @printf(io,"| %d | %s | %.4f%% | %.4f | %.5f | %d |\n",r.reynolds,r.variant,
                100r.maximum_density_deviation,r.maximum_mach,r.minimum_solid_clearance,r.steps)
        end
        println(io,"\nNegative clearance means fluid-particle centers enter a nominal solid. Such a case fails the original impermeability criterion even if it completes and its forces become quieter. All saved states were finite, retained their particle counts and completed the prescribed motion; the maximum center/velocity checks remain in each `native_state.csv`. Spatial refinement and further wall/flow validation remain necessary.\n")
        println(io,"## Reproduce\n")
        @printf(io,"Environment: %s, Julia %s, CUDA.jl %s. The baseline repository commit before adding cleaning was `%s`. The supplied published PDF SHA-256 is `e648ba052dfd1024f933f94787175b9ee4add567944ffe64b32cdb340bb2ce14`. No PDF copy or new Julia dependency is added to the repository.\n\n",
            settings["gpu"],settings["julia"],settings["cuda_jl"],settings["library_base_commit"])
        println(io,"Run from `gpu_version/`:\n")
        println(io,"```sh\njulia --project=. -t 2,0 test/divergence_cleaning.jl\njulia --project=. -t 2,0 test/runtests.jl\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/divergence_cleaning.jl\"\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/divergence_cleaning.jl\" timing\njulia --project=. -t 2,0 \"validation cases/spheric_test_06/cleaning_diagnostics.jl\"\npython \"validation cases/spheric_test_06/plot_divergence_cleaning.py\"\njulia --project=. \"validation cases/spheric_test_06/cleaning_report.jl\"\n```\n")
        print(io,"The study resumes completed variants and refuses to overwrite existing histories. Large VTKHDF histories, local logs and temporary paper extracts stay in ignored `runs/`. The focused tests cover independent dense operators in both dimensions/precisions, density-diffusion separation, sorting, scalar decay, wall protection, cleaning CFL, predictor fusion, graph/direct agreement, translated periodic crossings and stationary mDBC. See [implementation checks](IMPLEMENTATION_VALIDATION.md) for the reproduced broad-suite failure, [160 m/s controls](SOUND_SPEED_160.md) for the high-c0 startup limitations, and [the timestep/pause audit](TIMESTEP_AUDIT.md) for the subsequent investigation.\n")
    end
end

write_cleaning_report()
