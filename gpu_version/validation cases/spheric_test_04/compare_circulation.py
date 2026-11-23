"""Extract Case 04 circulation and the matching digitized SPHERIC Fig. 2 markers."""
import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import label, maximum_filter
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
# Match curves to the author-labelled Fig. 5.9: the highest peak belongs to osc3.
REFERENCE_PIXELS = {
    1: PIXELS,
    3: np.array([(82,249), (115,213), (148,173), (165,187), (182,54),
                 (199,183), (216,241), (249,322), (282,341), (316,357),
                 (349,366), (366,357), (399,359)]),
    2: np.array([(114,365), (148,259), (165,173), (182,169), (199,205),
                 (216,236), (249,323), (282,342), (316,353), (349,367),
                 (366,367), (382,364), (399,361), (415,354)]),
}


CREST_HEIGHT = 0.19 - np.sqrt(0.19**2 - (WAVELENGTH / 2)**2)
FIELDS = ("gamma", "x", "y", "radius", "peak")
MISSING = (np.nan,) * len(FIELDS)


def derivative(values, valid, spacing, axis):
    """Differentiate the quartic through five consecutive valid grid nodes.

    Prefer the centred stencil; shift it at fluid edges. A node with no
    five-point fluid stencil remains missing.
    """
    values = np.moveaxis(values, axis, -1)
    valid = np.moveaxis(valid, axis, -1)
    result = np.full(values.shape, np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(values, 5, axis=-1)
    usable = np.lib.stride_tricks.sliding_window_view(valid, 5, axis=-1).all(axis=-1)
    coefficients = ((-25,48,-36,16,-3), (-3,-10,18,-6,1), (1,-8,0,8,-1),
                    (-1,6,-18,10,3), (3,-16,36,-48,25))
    for i in (2, 1, 3, 0, 4):
        target = result[..., i:i + windows.shape[-2]]
        use = usable & np.isnan(target)
        target[use] = windows[use] @ coefficients[i] / (12 * spacing)
    return np.moveaxis(result, -1, axis)


def vorticity(tree, velocity, spacing, particle_spacing):
    # Extend beyond neighbouring crests so a shed vortex can travel with the flow.
    x = np.arange(CREST - 0.55, CREST + 0.55 + spacing/2, spacing)
    y = np.arange(0, 0.22 + spacing/2, spacing)
    xx, yy = np.meshgrid(x, y)
    distances, indices = tree.query(np.column_stack((xx.ravel(), yy.ravel())), k=8)
    weights = 1 / np.maximum(distances, spacing/100)**2
    vel = ((velocity[indices] * weights[:, :, None]).sum(axis=1) /
           weights.sum(axis=1)[:, None]).reshape(xx.shape + (2,))
    relative = xx - CREST
    folded = (relative + WAVELENGTH/2) % WAVELENGTH - WAVELENGTH/2
    arc_x = np.where(abs(relative) < 1.5*WAVELENGTH, WAVELENGTH/2 - abs(folded), 0)
    bed = 0.19 - np.sqrt(0.19**2 - arc_x**2)
    valid = (yy >= bed + particle_spacing/2) & (distances[:, 0].reshape(xx.shape) < 2*particle_spacing)
    omega = derivative(vel[:, :, 1], valid, spacing, 1) - derivative(vel[:, :, 0], valid, spacing, 0)
    return xx, yy, omega


class Vortex:
    """Identify a lee-side core once, then follow its signed local peak."""
    def __init__(self, sign):
        self.sign = sign
        self.core = None
        self.footprint = None
        self.lost = False

    def measure(self, xx, yy, omega, spacing, travel, growing):
        if not growing and (self.lost or self.core is None):
            return MISSING
        strength = np.where(np.isfinite(omega), self.sign * omega, -np.inf)
        peaks = (strength > 0) & (strength == maximum_filter(strength, size=3))
        lee = -self.sign * (xx - CREST)
        seed = (lee > 0) & (lee < WAVELENGTH/2) & (yy < CREST_HEIGHT + 0.045)
        if self.footprint is None:
            candidates = peaks & seed
        else:
            # A local maximum can move within an extended vortex when its core splits.
            # Require continuity with its last footprint or a bounded core displacement.
            x0, y0 = self.core
            nearby = np.hypot(xx-x0, yy-y0) <= travel + 2*spacing
            candidates = peaks & (self.footprint | nearby)
            if growing:
                candidates &= seed
                if not candidates.any():
                    candidates = peaks & seed
        if not candidates.any():
            self.lost = self.core is not None
            return MISSING
        core = np.unravel_index(np.argmax(np.where(candidates, strength, -np.inf)), xx.shape)
        # If the contour encloses a higher peak, recompute against that peak.
        while True:
            peak = strength[core]
            regions, _ = label(strength >= 0.1*peak)
            mask = regions == regions[core]
            highest = np.unravel_index(np.argmax(np.where(mask, strength, -np.inf)), xx.shape)
            if strength[highest] == peak:
                break
            if not candidates[highest] and np.hypot(xx[highest]-xx[core], yy[highest]-yy[core]) > travel + 2*spacing:
                self.lost = True  # merged structures cannot be assigned unambiguously
                if growing:
                    self.footprint = None
                return MISSING
            core = highest
        self.lost = False
        self.footprint = mask
        self.core = xx[core], yy[core]
        weights = strength[mask]
        x, y = np.average(xx[mask], weights=weights), np.average(yy[mask], weights=weights)
        radius = np.sqrt(np.average((xx[mask]-x)**2 + (yy[mask]-y)**2, weights=weights))
        return weights.sum()*spacing**2, x-CREST, y-CREST_HEIGHT, radius, peak


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
        phase = times * 360 / PERIOD
        # Select the latest fully sampled half-cycle with a preceding generation.
        # Output times may overshoot a reversal slightly due to solver stepping.
        halves = np.floor(phase / 180).astype(int)
        selected = max(0, int(np.floor((phase[-1] + 0.01)/180)) - 1)
        rows = []
        trackers = {}
        amplitude = (0.131, 0.196, 0.294)[args.oscillation-1]
        for i, (offset, count) in enumerate(zip(offsets, counts)):
            half = halves[i]
            # A sample on the final reversal closes the preceding half.
            if i == len(times)-1 and abs(phase[i] - 180*half) < 0.1 and half:
                half -= 1
            if half < selected-1 or half > selected:
                continue
            selection = slice(int(offset), int(offset + count))
            fluid = root["PointData/Type"][selection] == 1
            xy = root["Points"][selection, :2][fluid]
            velocity = root["PointData/Velocity"][selection, :2][fluid]
            tree = cKDTree(xy)
            outer = np.mean(velocity[(xy[:, 1] >= 0.25) & (xy[:, 1] <= 0.40), 0])
            local_phase = phase[i] - 180*half
            branches = {half: [], half-1: []}
            for spacing in (0.0025, 0.005):
                xx, yy, omega = vorticity(tree, velocity, spacing, args.particle_spacing)
                travel = 2 * amplitude * (times[i]-times[i-1]) if i else 0
                for generation in branches:
                    sign = -1 if generation % 2 == 0 else 1
                    key = (spacing, generation)
                    if key not in trackers:
                        trackers[key] = Vortex(sign)
                    gamma, x, y, radius, peak = trackers[key].measure(
                        xx, yy, omega, spacing, travel, generation == half)
                    # In the published orientation the growing vortex is left of the crest.
                    branches[generation].extend((gamma, sign*x, y, radius, peak))
            if half == selected:
                for generation, measurements in branches.items():
                    if generation >= 0:
                        rows.append([local_phase + 180*(half-generation), times[i], outer,
                                     phase[i], generation, *measurements])
    names = ["phase_deg", "time_s", "outer_mean_ux_m_s", "simulation_phase_deg", "generation"]
    for grid in ("2p5mm", "5mm"):
        names.extend([f"gamma_grid_{grid}_m2_s", f"x_{grid}_m", f"y_{grid}_m",
                      f"radius_{grid}_m", f"peak_{grid}_s_inv"])
    np.savetxt(run / "circulation.csv", sorted(rows), delimiter=",", header=",".join(names), comments="")
    write_reference(run, args.oscillation)


def write_reference(run, oscillation):
    pixels = REFERENCE_PIXELS[oscillation]
    phase_deg = (pixels[:, 0] - 82) * 360 / 398
    gamma = (375 - pixels[:, 1]) * 0.07 / 322
    # Rounded visual readings at nominal 90 and 180 degrees from the thesis
    # (Figs. 5.11, 5.13 and 5.15). The phase axis in those scans is skewed.
    landmarks = {
        1: [(-0.055, 0.003, 0.032), (-0.073, 0.003, 0.018)],
        2: [(-0.060, 0.006, 0.025), (-0.063, -0.005, 0.017)],
        3: [(-0.052, 0.008, 0.027), (-0.028, 0.029, 0.011)],
    }
    reference = np.full((len(gamma), 6), np.nan)
    reference[:, :2] = np.column_stack((phase_deg, gamma))
    for angle, moments in zip((90, 180), landmarks[oscillation]):
        reference[np.argmin(abs(phase_deg - angle)), 2:] = (angle, *moments)
    np.savetxt(run / "circulation_reference.csv", reference, delimiter=",",
               header="phase_deg,gamma_m2_s,landmark_phase_deg,x_m,y_m,radius_m", comments="")


if __name__ == "__main__":
    main()
