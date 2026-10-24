# CUDA graph kernel preparation: StillWedge

Measured on 2026-10-24 with Julia 1.12.7, CUDA.jl/CUDACore 6.4.2,
CUDA runtime 12.9, NVIDIA driver 616.92 and an RTX 5080, using `-t 4,0`.
The generated StillWedge example has 3,027 particles, Float32 physics,
Float64 positions, symplectic integration and particle/grid output every
0.01 s. Each run advances 4 s of physical time in 8,500 steps, with three
cell-list rebuilds, 405 control-state readbacks and 401 particle frames.

The experiment switches only kernel preparation. Both variants use the same
driver, kernels and improved capture diagnostics. The control disables
preparation, so its cold timings include capture fallback and warning costs.
These measurements are unprofiled complete simulations, including input
generation, solver setup, output and finalization; package loading and the
post-run correctness checks are excluded.

| Median elapsed time | Preparation disabled | Preparation enabled |
| --- | ---: | ---: |
| Warm complete call, nine alternating pairs | 0.328055 s | 0.330041 s |
| Warm stepping loop, same pairs | 0.257300 s | 0.258769 s |
| First complete call in a fresh process, three alternating pairs | 32.987128 s | 32.814798 s |
| First-call solver wall time, same fresh processes | 20.224932 s | 19.941691 s |

Preparation produced **no warm throughput improvement**: median complete-run
time increased by about 2 ms (0.6%). The median paired baseline/prepared ratio
was 0.9916, with an interquartile range of [0.9899, 0.9947]. The fresh-process
complete-call median decreased by about 0.17 s (0.5%). Only three cold pairs
were measured; their absolute times include JIT compilation in this worktree
with `--compiled-modules=existing`, and depend on its package-cache state.
They should not be treated as typical startup times for a precompiled package.

All three control cold runs failed their first capture with error 906,
`ERROR_STREAM_CAPTURE_IMPLICIT`, during CUDACore's kernel-relocation
`cuMemcpyHtoD_v2`. Each used 1,206 graph batches and one direct batch.
All three prepared cold runs used 1,207 graph batches, zero direct batches,
zero capture failures and an empty fallback list. Both warmed variants used
1,207 graph batches without failures.

Every warm pair matched particle positions, velocities, densities, IDs, types
and groups exactly after sorting by ID. Every exported particle/grid dataset
matched by element type, shape and SHA-256. Step, rebuild, readback and frame
counts and final simulation times also matched. The cold runs had the same
step/rebuild/readback/frame counts and final times.

The supported benefit is reliable first capture and clearer diagnostics,
with a small measured warm-run overhead in this case. The preparation pass
uses separate inactive device control state and leaves the live simulation
state and counters unchanged.

The focused suites passed 6,035 assertions covering configuration and warning
details, preparation without live-state changes, stationary predictors,
Updated mDBC, graph/direct integration and saved frames, floating/moving
bodies, accepted-state caches and timestep readback/stop transitions.

Raw data:

- [Nine warm pairs](graph_preparation_results.csv)
- [Three fresh-process pairs](graph_preparation_cold_results.csv)

Commands from `gpu_version`:

```powershell
julia --project=. --compiled-modules=existing -t 4,0 benchmark/graph_preparation.jl --rounds=9 --simtime=4
julia --project=. --compiled-modules=existing -t 4,0 -L test/effective_configuration.jl test/stationary_predictors.jl
julia --project=. --compiled-modules=existing -t 4,0 -L test/updated_mdbc.jl -L test/gpu_execution.jl -L test/final_state_caches.jl test/step_state.jl
```

For cold measurements, run these in separate Julia processes, alternating
their order across rounds and assigning each a different output filename:

```powershell
julia --project=. --compiled-modules=existing -t 4,0 benchmark/graph_preparation.jl --cold=baseline --simtime=4 --output=baseline.csv
julia --project=. --compiled-modules=existing -t 4,0 benchmark/graph_preparation.jl --cold=prepared --simtime=4 --output=prepared.csv
```
