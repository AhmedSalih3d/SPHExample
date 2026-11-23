"""Extract Case 04 circulation and the matching digitized SPHERIC Fig. 2 markers."""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

PERIOD = 8.46
CREST = 0.8
WAVELENGTH = 0.22

# Approximate centers of the filled osc1 markers in the published 493 x 431 figure.
# Plot axes span columns 82–480 (0–360 deg) and rows 375–53 (0–0.07 m²/s).
PIXELS = np.array([
    (82, 365), (114, 302), (148, 273), (165, 283), (182, 236),
    (199, 252), (216, 281), (249, 331), (283, 329), (316, 344),
    (349, 345), (383, 352), (399, 356),
])
# Open triangles and plus signs are oscillations 2 and 3 respectively.
REFERENCE_PIXELS = {
    1: PIXELS,
    2: np.array([(82,249), (115,213), (148,173), (165,187), (182,54),
                 (199,183), (216,241), (249,322), (282,341), (316,357),
                 (349,366), (366,357), (399,359)]),
    3: np.array([(114,365), (148,259), (165,173), (182,169), (199,205),
                 (216,236), (249,323), (282,342), (316,353), (349,367),
                 (366,367), (382,364), (399,361), (415,354)]),
}


def circulation(position, velocity, grid_spacing, particle_spacing):
    """Integrate the strongest connected signed vortex in the central ripple region."""
    x = np.arange(CREST - WAVELENGTH, CREST + WAVELENGTH + grid_spacing/2, grid_spacing)
    y = np.arange(0.0, 0.22 + grid_spacing/2, grid_spacing)
    xx, yy = np.meshgrid(x, y)
    distance, neighbors = cKDTree(position).query(np.column_stack((xx.ravel(), yy.ravel())), k=8)
    weights = 1 / np.maximum(distance, grid_spacing / 100)**2
    grid_velocity = ((velocity[neighbors] * weights[:, :, None]).sum(axis=1) /
                     weights.sum(axis=1)[:, None]).reshape(xx.shape + (2,))
    omega = (np.gradient(grid_velocity[:, :, 1], grid_spacing, axis=1)
             - np.gradient(grid_velocity[:, :, 0], grid_spacing, axis=0))
    relative = xx - CREST
    from_crest = (relative + WAVELENGTH/2) % WAVELENGTH - WAVELENGTH/2
    u = WAVELENGTH/2 - np.abs(from_crest)
    bed = 0.19 - np.sqrt(0.19**2 - u**2)
    # Exclude the first two particle spacings above the bed from this pilot metric.
    admissible = (yy >= bed + 2 * particle_spacing) & (distance[:, 0].reshape(xx.shape) < 2 * particle_spacing)
    peak = np.max(np.abs(omega[admissible]))
    if peak == 0:
        return 0.0
    vortices = []
    for sign in (-1, 1):
        signed = sign * omega
        regions, count = label(admissible & (signed >= 0.1 * peak))
        strength = np.bincount(regions.ravel(), weights=signed.ravel(), minlength=count+1)[1:]
        vortices.extend(strength * grid_spacing**2)
    return max(vortices, default=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("oscillation", type=int, choices=(1, 2, 3))
    parser.add_argument("particle_spacing", type=float)
    args = parser.parse_args()
    run = args.run

    with h5py.File(run / "RippledBed.vtkhdf") as handle:
        root = handle.get("VTKHDF/Particles", handle["VTKHDF"])
        times = root["Steps/Values"][:]
        offsets = root["Steps/PointOffsets"][:]
        counts = root["NumberOfPoints"][:]
        dt = np.diff(times)
        assert np.all(dt > 0)
        curves = []
        outer_velocity = []
        for offset, count in zip(offsets, counts):
            selection = slice(int(offset), int(offset + count))
            particle_type = root["PointData/Type"][selection]
            fluid = particle_type == 1
            if not np.any(fluid):
                raise ValueError("Expected Fluid particle type 1 in the export")
            xy = root["Points"][selection, :2][fluid]
            velocity = root["PointData/Velocity"][selection, :2][fluid]
            curves.append([circulation(xy, velocity, spacing, args.particle_spacing) for spacing in (0.0025, 0.005)])
            outer_velocity.append(np.mean(velocity[(xy[:, 1] >= 0.25) & (xy[:, 1] <= 0.40), 0]))
    curves = np.array(curves)
    phase = times * 360 / PERIOD
    np.savetxt(run / "circulation.csv", np.column_stack((phase, times, curves, outer_velocity)), delimiter=",",
               header="phase_deg,time_s,gamma_grid_2p5mm_m2_s,gamma_grid_5mm_m2_s,outer_mean_ux_m_s", comments="")
    pixels = REFERENCE_PIXELS[args.oscillation]
    phase_deg = (pixels[:, 0] - 82) * 360 / 398
    gamma = (375 - pixels[:, 1]) * 0.07 / 322
    np.savetxt(run / "circulation_reference.csv", np.column_stack((phase_deg, gamma)),
               delimiter=",", header="phase_deg,gamma_m2_s", comments="")


if __name__ == "__main__":
    main()
