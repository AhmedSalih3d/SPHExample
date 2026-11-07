"""Offline checks and snapshots for StillWedge: python analyse_still_wedge_iisph.py OUTPUT_DIR.

Requires numpy, h5py and matplotlib; these are optional analysis dependencies.
The Julia simulation and its normal output do not require Python.
"""
import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def analyse(directory):
    directory = Path(directory)
    snapshots = []
    with h5py.File(directory / "StillWedge.vtkhdf", "r") as file:
        root = file["VTKHDF"]
        times = root["Steps/Values"][:]
        offsets = root["Steps/PointOffsets"][:]
        n = int(root["NumberOfPoints"][0])
        if n != 3027:
            raise ValueError("This analysis assumes the original 3027-particle StillWedge case at dx=0.02 m")
        selected = {int(np.argmin(np.abs(times - t))) for t in (0, 1, 2, times[-1])}
        max_speed, max_density = 0.0, 0.0
        fixed_displacement = 0.0
        for frame, (time, offset) in enumerate(zip(times, offsets)):
            columns = slice(int(offset), int(offset) + n)
            xy = root["Points"][columns, :2]
            types = root["PointData/Type"][columns]
            ids = root["PointData/ID"][columns]
            rho = root["PointData/Density"][columns]
            pressure = root["PointData/Pressure"][columns]
            velocity = root["PointData/Velocity"][columns, :2]
            fluid = types == 1
            fixed_order = np.flatnonzero(~fluid)[np.argsort(ids[~fluid])]
            if frame == 0:
                fluid_ids = np.sort(ids[fluid])
                fixed_ids = ids[fixed_order].copy()
                fixed_xy = xy[fixed_order].copy()
                water_level = float(xy[fluid, 1].max())
            assert np.array_equal(np.sort(ids[fluid]), fluid_ids), "Fluid identity/mass changed"
            assert np.array_equal(ids[fixed_order], fixed_ids), "Boundary identity changed"
            assert all(np.isfinite(a).all() for a in (xy, rho, pressure, velocity)), "Nonfinite frame"
            fixed_displacement = max(fixed_displacement, float(np.abs(xy[fixed_order] - fixed_xy).max()))
            max_speed = max(max_speed, float(np.linalg.norm(velocity[fluid], axis=1).max()))
            max_density = max(max_density, float(rho[fluid].max()))
            if frame in selected:
                snapshots.append((float(time), xy.copy(), fluid.copy(), pressure.copy()))

    # The reference case has dx=0.02 m, h=2dx and Wendland C2. This is the
    # normalized SPH interpolation used for physical fluid-pressure probes.
    h, mass, rho0, gravity = 0.04, 0.4, 1000.0, 9.81
    profile = []
    for y in (0.1, 0.2, 0.3, 0.4):
        q = np.linalg.norm(xy[fluid] - [0.3, y], axis=1) / h
        weight = np.where(q <= 2, 7 / (4 * np.pi * h**2) * (1 - q / 2)**4 * (2*q + 1), 0)
        weight *= mass / rho[fluid]
        measured = float(np.dot(weight, pressure[fluid]) / weight.sum())
        expected = rho0 * gravity * (water_level - y)
        profile.append(dict(x=0.3, y=y, pressure_pa=measured,
                            hydrostatic_pa=expected, relative_error=(measured - expected) / expected))
    with (directory / "StillWedge_IISPH_pressure_profile.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(profile[0]))
        writer.writeheader()
        writer.writerows(profile)

    figure, axes = plt.subplots(2, 2, figsize=(13, 5.6), sharex=True, sharey=True, constrained_layout=True)
    for axis, (time, points, mask, values) in zip(axes.flat, snapshots):
        axis.scatter(points[~mask, 0], points[~mask, 1], s=4, c="#818b96", linewidths=0)
        scatter = axis.scatter(points[mask, 0], points[mask, 1], c=values[mask]/1000,
                               s=3.5, cmap="viridis", vmin=0, vmax=rho0*gravity*water_level/1000, linewidths=0)
        axis.set(xlim=(-0.06, 2.26), ylim=(-0.06, 0.73), title=f"t = {time:.2f} s", xlabel="x (m)", ylabel="y (m)")
        axis.set_aspect("equal")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("StillWedge - IISPH, original 3,027-particle layout", fontsize=15)
    figure.colorbar(scatter, ax=axes, shrink=0.85, label=f"Pressure (kPa; capped at {rho0*gravity*water_level/1000:.2f})")
    figure.savefig(directory / "StillWedge_IISPH_snapshots.png", dpi=180)
    plt.close(figure)

    statistics = dict(frames=len(times), final_time=float(times[-1]), particle_count=n,
                      fluid_count=len(fluid_ids), fixed_max_displacement=fixed_displacement,
                      all_frame_max_speed=max_speed, all_frame_max_density=max_density,
                      final_density_min=float(rho[fluid].min()), final_density_max=float(rho[fluid].max()),
                      final_max_speed=float(np.linalg.norm(velocity[fluid], axis=1).max()),
                      pressure_profile=profile)
    (directory / "StillWedge_IISPH_frame_checks.json").write_text(json.dumps(statistics, indent=2))
    print(json.dumps(statistics, indent=2))
    return statistics


if __name__ == "__main__":
    analyse(sys.argv[1])
