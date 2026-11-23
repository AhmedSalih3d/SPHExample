# SPHERIC Test 04 — 2-D oscillatory flow over a rippled bed

Run from `gpu_version` with an NVIDIA GPU:

```powershell
julia --project=. -t 2,0 "validation cases/spheric_test_04/manual_case.jl"
```

Edit the settings at the top of [manual_case.jl](manual_case.jl). In your
editor, select the whole file with `Ctrl+A`, then use your Julia
execute-selection shortcut. The selection starts the run immediately. The
default runs oscillation 1 for one 8.46 s period with 2.5 mm particle spacing
and SimpleMDBC. Set `oscillation` to 1, 2, or 3 for velocity amplitudes
0.131, 0.196, or 0.294 m/s. ParaView opens by default with a fixed speed scale
of 0–0.17, 0–0.25, or 0–0.37 m/s, respectively. The view and `settings.txt`
show the case name, particle spacing, solver and forcing settings, and
exported variables. Set `visualize=false`
for unattended runs. The state and PNG are saved either way.

The setup uses the benchmark's 1.6 m × 0.5 m section and three circular-arc
ripples with 0.22 m wavelength and 0.19 m radius, giving approximately
0.035 m crest height. The solver models the oscillation as a spatially uniform
horizontal acceleration in a periodic channel with a fixed top and bed.
The velocity amplitude is the target in the inviscid limit. This setup is
a starting point for comparison with the published PIV vortex circulation,
centre and size.
[SPHERIC benchmark specification](https://www.spheric-sph.org/tests/test-04)

Each oscillation has three reusable folders under `runs/`, for example
`runs/osc1_slot1` and `runs/osc1_slot2`. With `overwrite=true`, rerunning a
slot replaces its solver output files in place. Set it to `false` to protect
existing output. Use another slot to keep a comparison. A run writes a single
particle VTKHDF file, a ParaView loader, a log and its settings. The file stores
position, velocity, pressure and particle type at the requested output times.
Reduce `frames` to save disk space and circulation analysis time. The default
writes 851 states including the initial state.

## Circulation comparison in ParaView

Before running, put Python (with NumPy, SciPy and h5py) and ParaView 6.1
(`pvpython` and `paraview`) on PATH. After the solver completes, Case 04
calculates circulation from every saved frame in the selected slot. It saves
both CSV histories, the combined particle and chart state
(`circulation_paraview.pvsm`), and the chart figure
(`circulation_paraview.png`) in that slot. The default opens the combined
state. Use **File > Load State** to reopen it with its VTKHDF and CSV inputs.

The chart overlays the selected oscillation's digitized Figure 2 markers on
the 2.5 mm and 5 mm analysis-grid histories. Phase advances from the start of
the simulation as 360 times elapsed time divided by the forcing period.
Experimental markers cover the published phases of one cycle.

The calculation interpolates fluid velocity onto each analysis grid and
integrates the strongest connected region of either sign above 10% of
the local maximum absolute vorticity. It excludes two particle spacings
above the bed and searches within one wavelength of the middle crest.
This pilot comparison starts from rest. Experimental vortex identity,
phase alignment, near-bed mask sensitivity, centre and size still need
verification.
