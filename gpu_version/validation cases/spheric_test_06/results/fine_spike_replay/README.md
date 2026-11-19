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

## Matched shifting controls

The two controls used the same geometry, motion, CFL, clipping deadlines and
output schedule as the planar replay. Both completed all 141 frames through
0.90 s with zero frozen particles. The classic full shifting run took 81,496
accepted steps and the unshifted run took 81,495.

| Shifting mode | Supported rear probes at 0.82 s | Pressure Cd at 0.82 s | Minimum pressure Cd, 0.76–0.86 s | Peak sampled rear fluid pressure, 0.76–0.86 s |
|---|---:|---:|---:|---:|
| Planar | 40/40 | −27.58 | −27.63 at 0.819 s | 476 kPa |
| Classic DualSPHysics full | 40/40 | +1.39 | −9.45 at 0.782 s | 417 kPa |
| Off | 0/40 | +6.67 | +2.76 at 0.860 s | 2.77 kPa |

Both new controls had zero supported central rear probes at 0.48 s.
Classic shifting brought fluid back earlier and reduced the largest sampled
drag reversal in the diagnostic window. The rear gap and a pressure pulse
still occurred. The comparison changes the complete shifting rule, so it
does not isolate the contribution of density weighting or clipping.

Run each control with `fine_spike_replay.jl`, then `analyze_fine_spike.jl`,
passing `dual-shifting` or `no-shifting` to both scripts. Run
`plot_fine_spike.py --compare-shifting` to reproduce the figure.

[Comparison plot](shifting_comparison.png) · [Classic shifting history](dual_shifting_history.csv) · [Unshifted history](no_shifting_history.csv)
