# SPHERIC Test 04 â€” 2-D oscillatory flow over a rippled bed

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
of 0â€“0.17, 0â€“0.25, or 0â€“0.37 m/s, respectively. The view and `settings.txt`
show the case name, particle spacing, solver and forcing settings, and
exported variables. Set `visualize=false`
for unattended runs. The state and PNG are saved either way.

The setup uses the benchmark's 1.6 m Ã— 0.5 m section and three circular-arc
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

The particle layout includes the circulation comparison. The second layout,
**Vortex centre and radius**, plots crest-relative horizontal and vertical
position and the vorticity-weighted RMS radius. Black markers on those charts
are approximate visual readings at 90 and 180 degrees from Earnshaw's thesis,
Figures 5.11, 5.13 and 5.15. The circulation points were digitized from SPHERIC
Figure 2. The SPHERIC caption interchanges the two higher-amplitude curves
relative to Earnshaw's labelled Figure 5.9. The analysis uses the author's
assignment: the peak near 0.0448 m^2/s belongs to osc2 and 0.0698 to osc3.

### Experimental convention and numerical extraction

Earnshaw identified the growing and previously shed vortices in the same
half-cycle. Following her convention, both appear on one life-cycle axis.
Growth occupies 0-180 degrees. The shed branch occupies 180-360 degrees and
its horizontal coordinate and sign are reflected into the growing vortex's
orientation. The numerical analysis uses the latest completed half-cycle,
including the earlier half-cycle to establish the identity of the shed vortex.
For a run shorter than a half-cycle it uses the available growing branch.
In the CSV, `simulation_phase_deg` and `time_s` give physical sampling time;
`phase_deg` gives the experimental life-cycle phase. A run from rest has
different flow history from the laboratory's repeated oscillations.

Each reconstructed velocity uses eight fluid neighbours with inverse-square
distance weights. Vorticity uses five-point quartic derivative stencils in x
and y. Stencils shift at the fluid boundary and require five valid grid nodes.
Nodes must be at least half a particle spacing above the bed and within two
particle spacings of fluid. The experiment used PIV and visual identification.
Here an automated tracker looks for a signed lee-side core at the central crest,
follows local maxima within its preceding footprint or a displacement gate of
twice the bulk speed amplitude times the frame interval plus two grid spacings,
and integrates the connected contour above 10% of its own peak. The growing
branch permits reacquisition on the same lee slope. Missing or ambiguous
measurements are NaN; a lost shed track ends. Circulation is the area integral
of vorticity magnitude on that component. Centre and RMS radius use the same
vorticity weights. Peak selection and the wall stencil remain numerical
approximations, particularly when the near-bed sheet has multiple cores.

Run `python test_circulation.py` from this directory to check the derivative,
signed contour selection, moments and missing shed-track behaviour.
[Earnshaw, experimental methods sections 4.3-4.4 and oscillatory results section 5.4](https://era.ed.ac.uk/server/api/core/bitstreams/13a36f81-cc75-422f-bc2b-11d36660c600/content).
[Earnshaw and Greated (1998)](https://doi.org/10.1007/s003480050229).

### Checks on the saved 2.5 mm particle runs

At the frame nearest life-cycle phase 90 degrees (89.79 degrees), the measured
outer speeds are 0.13100 and 0.19597 m/s. The target amplitudes are 0.131 and
0.196 m/s. The simulated orbital excursion amplitudes implied by these speeds
and the 8.46 s forcing period are 17.64 and 26.39 cm (benchmark 17.6 and
26.4 cm). This checks the forcing scale independently of vortex extraction.

| Oscillation | Analysis grid | Circulation (m^2/s) | X from crest (mm) | Y from crest (mm) | RMS radius (mm) |
|---|---:|---:|---:|---:|---:|
| 1 | 2.5 mm | 0.00662 | -10.1 | 1.9 | 5.5 |
| 1 | 5 mm | 0.00354 | -25.8 | -10.1 | 1.9 |
| 1 | Experiment, approximate | 0.0302 | -55 | 3 | 32 |
| 2 | 2.5 mm | 0.00597 | -12.1 | 2.9 | 6.2 |
| 2 | 5 mm | 0.01671 | -22.9 | 1.0 | 13.0 |
| 2 | Experiment, approximate | 0.0448 | -60 | 6 | 25 |

The extracted growing cores at this phase are smaller and closer to the
crest horizontally than the experimental structures. The two analysis grids
give materially different measurements. In these single-cycle runs, the
automated tracker loses the shed branch by 195 degrees on both grids; osc2's
fine grid has no assigned shed core. Later experimental markers therefore
have no verified numerical counterpart in these runs.
