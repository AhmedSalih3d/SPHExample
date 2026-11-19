# Rear-wall contact and PlanarShifting

The moving-square tests use Re=100, L/dp=80, CFL=0.10, c0=160 m/s, and
fluid-only density diffusion. Each run saves 46 frames through 0.90 s.
The four original boundary comparisons use `NoShifting`. The fifth run
changes only the shifting mode to `PlanarShifting` for RegularizedMDBC.

The contact diagnostic uses the middle 20 of the 80 outer-layer rear-wall
particles, covering the central 0.25 m of the rear face. Face-to-fluid
distance is measured from the nominal face to the nearest fluid center.
Kernel contact is measured from the actual wall particle to fluid centers
within its support radius H = 2h = 42.43 mm. Initial face-to-fluid distance
is 0.5 dp = 6.25 mm. The approach velocity is the horizontal fluid velocity
minus the square velocity for fluid neighbors of the central owners.
Positive values indicate motion toward the rear wall in the square frame.
Approach velocity and pressure are absent where those owners have no fluid
neighbors.

| Boundary | Earlier frame, unsupported central owners | Spike frame, unsupported central owners | Pressure Cd at spike |
| --- | --- | --- | ---: |
| DBC | 20/20 at 0.78 s | 0/20 at 0.84 s | −14.00 |
| SimpleMDBC | 20/20 at 0.82 s | 4/20 at 0.88 s | −1.49 |
| RegularizedMDBC | 20/20 at 0.82 s | 0/20 at 0.88 s | −6.91 |

In RegularizedMDBC the median central face-to-fluid distance falls from
5.80 dp at 0.82 s to 2.63 dp at 0.86 s and 1.27 dp at 0.88 s. The peak
approach speed within the central owners' support at 0.88 s is 1.47 m/s.
The near-square pressure plot shows a 31.92 kPa fluid peak behind the
square. Within the central wall stencil the fluid pressure peaks at
13.76 kPa. DBCPlusImpedance retains fluid support for all 20 central
owners in every saved frame, while some sites outside the central patch
lose support. Its pressure Cd stays positive in the 0.70–0.90 s window.

This sequence is consistent with rear-side compression as fluid support
returns. The saved states are 20 ms apart, so they do not
identify the initiating instability or the peak load between outputs.
The measured gap is a region without fluid particle centers in a
single-phase SPH simulation.

With PlanarShifting, the central owners regain support earlier.
RegularizedMDBC's minimum pressure Cd moves from −6.91 at 0.88 s to −2.28
at 0.76 s. At the earlier event the fluid pressure reaches 154.86 kPa
within central wall support and peak approach speed is 1.94 m/s.
The final mean fluid pressure rises from 1.01 to 6.64 kPa. Relative L2
pressure and viscous drag errors over 0.10–0.90 s change from
50.00% and 30.91% to 42.81% and 29.23%. Minimum saved clearance changes
from −1.45 to −1.35 mm. A shallower drag minimum can coexist with a
stronger local pressure disturbance.

The negative concentration-gradient sign and `A * h * |v| * dt` scale agree
with [Jandaghian et al., Eqs. 22–24](https://arxiv.org/html/2110.10076).
Both the GPU and CPU legacy code use `m_j / rho_i` for the concentration
gradient, whereas Eq. 23 uses neighbor volume `m_j / rho_j`. The divergence
of position in the GPU shift uses fluid neighbors and the `r_ji` sign
convention. The current option moves particle positions without an explicit
displacement cap, a restriction on the wall-normal shift, or transport
corrections for density and momentum. Those are separate formulation
choices. The [consistent δ⁺-SPH equations](https://arxiv.org/html/2505.10215v1#S2.SS2)
illustrate the transport corrections. This audit tests the existing
PlanarShifting option as a whole, without isolating those choices.

[Contact and load histories](rear_wall_contact.png) · [Per-frame metrics](rear_wall_contact.csv)

The [recontact sequence](rear_wall_recontact_sequence.png) marks fluid
particle centers within the central wall particles' kernel support. The
[pressure close-ups](rear_wall_spike_pressure.png) show the rear of the
square at each observed force minimum, with the 0.88 s impedance frame
for comparison. A ring marks the peak fluid pressure in each displayed
region. Every pressure panel uses the same symmetric logarithmic scale.

From `gpu_version`, generate the four unshifted `half-step` runs using
`regularized_mdbc_experiment.jl` with `dbc`, `simple`, `regularized` and
`impedance`. Generate the shifted comparison and render the diagnostic
with these commands.

```sh
julia --project=. -t 2,0 "validation cases/spheric_test_06/regularized_mdbc_experiment.jl" regularized half-step-planar
python "validation cases/spheric_test_06/audit_rear_wall_contact.py"
python "validation cases/spheric_test_06/plot_rear_wall_recontact.py"
```
