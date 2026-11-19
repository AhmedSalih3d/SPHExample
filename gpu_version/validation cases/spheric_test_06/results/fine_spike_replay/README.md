# Fine-resolution pressure reversal

Replayed the Re = 100, L/dp = 160, CFL = 0.10, coefh = 2.0 reference DBC
startup with PlanarShifting and zero-gravity linear density diffusion.
The replay retained the original 20 ms clipping deadlines and sampled
accepted states approximately every 1 ms from 0.76 to 0.86 s. Both runs
took 81,498 steps. The particle positions, velocities, densities, types,
and markers at 0.82 s matched exactly. The pressure force sums matched
the original histories at all six original output times in the window.

Each square particle contributes its x pressure force to the nearest
nominal face. Particles equidistant from two faces contribute half to each.
Front, rear, and top-plus-bottom contributions sum to the original pressure
drag coefficient. The rear support and local fluid pressure probes use the
central 40 particles of the outer rear row.

- Rear fluid first entered kernel range of two central probes at 0.767 s.
- All 40 probes had neighbors at 0.809 s.
- Pressure drag reversed at 0.813 s. The largest local approach speed was
  3.65 m/s at that sample.
- Local rear fluid pressure peaked at 476 kPa at 0.814 s.
- Net pressure Cd reached −27.63 at 0.819 s. Its front, rear, and side
  contributions were +7.50, −34.89, and −0.24.

At 0.82 s the larger-timestep control had rear Cd −6.95 and peak rear
fluid pressure 9.55 kPa, compared with −34.83 and 188 kPa in the original
half-timestep history. The strongest net load follows the local fluid
pressure peak. These diagnostics resolve the load reversal and its
association with recontact within the saved time resolution.

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/fine_spike_replay.jl"
julia --project=. -t 2,0 "validation cases/spheric_test_06/analyze_fine_spike.jl" archived
julia --project=. -t 2,0 "validation cases/spheric_test_06/analyze_fine_spike.jl" control
julia --project=. -t 2,0 "validation cases/spheric_test_06/analyze_fine_spike.jl"
python "validation cases/spheric_test_06/plot_fine_spike.py"
```

[Face loads and rear contact](spike_face_loads.png) · [101 measured samples](fine_faces.csv)
