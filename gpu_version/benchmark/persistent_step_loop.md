# Persistent step loop experiment

From the GPU package directory, run:

```sh
julia --project=. benchmark/persistent_step_loop.jl
```

The benchmark uses small, perturbed 2D Float32 fluid patches, all-pairs
Wendland C2 interactions, and a first-order integration step. All variants share
the force, density-rate, adaptive timestep and integration code. The timestep
uses force and pair-signal maxima and advances a Float64 clock in step order.
The split baseline launches three kernels per step. The fused baseline launches
the exact resident kernel once per step, so comparing its graph against the
resident batch isolates the effect of looping inside that kernel. The experiment
measures the full miniature timestep batch with identical particle and step
counts. It excludes cell-list rebuilds, boundary handling and simulation output.

Each sample starts from the same initial condition. After the timed
launch and synchronization, the benchmark checks final positions, velocities,
densities, accelerations and density rates, plus every timestep and cumulative
time, against the split reference. The comparisons are exact. The reference
also checks for finite state and positive density and timesteps. Graph capture,
warm-up, resets and validation readbacks are outside the timing. Variant order
rotates each round.

## RTX 5080 results

Windows WDDM, NVIDIA driver 616.92, Julia 1.12.7. Times are median
microseconds per batch over 21 warm rounds. The last column gives the median
paired speedup over the fused graph, with the middle 50% of paired ratios.
Every run passed the state and timestep checks.

| Particles | Steps | Split | Split graph | Fused | Fused graph | Resident | Paired speedup [Q1, Q3] |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 25.50 | 15.20 | 14.50 | 13.60 | 15.30 | 0.941 [0.856, 0.965] |
| 32 | 8 | 195.60 | 66.40 | 89.10 | 57.20 | 52.20 | 1.099 [1.069, 1.123] |
| 32 | 32 | 734.20 | 260.10 | 305.70 | 224.90 | 199.40 | 1.126 [1.090, 1.154] |
| 64 | 1 | 29.10 | 20.50 | 19.20 | 17.60 | 21.50 | 0.921 [0.551, 0.968] |
| 64 | 8 | 213.90 | 119.70 | 112.80 | 93.10 | 88.60 | 1.046 [1.039, 1.062] |
| 64 | 32 | 790.40 | 393.50 | 419.60 | 363.70 | 338.50 | 1.072 [1.061, 1.083] |

Against the fused graph, the resident loop was slower at one step and
1.05–1.13× faster at 8 or 32 steps. This is evidence for launch amortization on
the two miniature workloads. The production solver uses different interaction,
integration and orchestration paths, so a solver speedup would require a separate
end-to-end comparison.

A separate run on the same machine reproduced the direction of the result in
all six cases and passed every correctness check. Its paired speedups for 8
and 32 steps were 1.095× and 1.133× at 32 particles, and 1.075× and 1.076× at
64 particles.

## Synchronization and scope

Each launch uses exactly one block of one or two warps. All threads reach each
block barrier in the same order. The barriers publish the force results to the
timestep reduction, the chosen timestep to integration, and the updated state
to the next force evaluation. The batch cap of 32 bounds the work per launch
on a display GPU and provides a return point to the host. Watchdog limits
were not measured.

A single block uses only one SM. Extending the loop across blocks would
require a supported grid-wide barrier with memory ordering and a cooperative
launch sized to the compiled kernel's residency limit, including its register
and shared-memory usage. The benchmark exercises block-local synchronization
only and stays outside the production solver.
