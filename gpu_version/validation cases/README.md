# Validation cases

Reproducible comparisons with external benchmark data belong here. Each case
records its source, numerical settings, measured errors, and limitations.

- [SPHERIC Test 04 — oscillatory flow over a rippled bed](spheric_test_04/README.md)
  has a manual entry point with three forcing presets and reusable output slots.
- [SPHERIC Test 06: moving square](spheric_test_06/README.md): pressure and
  viscous drag at Reynolds numbers 50, 100, and 150, with a manual single-run
  entry point exposing the tank/body layout and simulation settings, plus a
  ParaView force/reference comparison.
- [SPHERIC Test 14: free heave](spheric_test_14/README.md) covers three
  releases of rounded floats and compares the raw simulated displacement
  histories with motion-capture measurements.
