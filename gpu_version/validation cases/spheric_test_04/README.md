# SPHERIC Test 04 — 2-D oscillatory flow over a rippled bed

Run from `gpu_version` with an NVIDIA GPU:

```powershell
julia --project=. -t 2,0 "validation cases/spheric_test_04/manual_case.jl"
```

Edit the settings at the bottom of [manual_case.jl](manual_case.jl). The
default runs oscillation 1 for one 8.46 s period with 5 mm particle spacing
and SimpleMDBC. Set `oscillation` to 1, 2, or 3 for velocity amplitudes
0.131, 0.196, or 0.294 m/s. Set `visualize=true` to open ParaView or load
`RippledBed.vtkhdf` there manually.

The setup uses the benchmark's 1.6 m × 0.5 m section and three circular-arc
ripples with 0.22 m wavelength and 0.19 m radius, giving approximately
0.035 m crest height. The solver models the oscillation as a spatially uniform
horizontal acceleration in a periodic channel with a fixed top and bed.
The velocity amplitude is the target in the inviscid limit. This setup is
a starting point for comparison with the published PIV vortex circulation,
centre and size. It does not yet calculate those validation metrics.
[SPHERIC benchmark specification](https://www.spheric-sph.org/tests/test-04)

Each oscillation has three reusable folders under `runs/`, for example
`runs/osc1_slot1` and `runs/osc1_slot2`. With `overwrite=true`, rerunning a
slot replaces its solver output files in place. Set it to `false` to protect
existing output. Use another slot to keep a comparison. A run writes a single
particle VTKHDF file, a ParaView loader, a log and its settings. The file stores
position, velocity, pressure and particle type at the requested output times.
Reduce `frames` to save space. The entry point caps output at 201 states per run,
including the initial state, and caps the lattice at 500,000 sites.
