"""Measure particle support and closing motion at the rear wall of the moving square."""
import csv
import tomllib
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


CASE = Path(__file__).resolve().parent
RUNS = CASE / "runs" / "regularized_mdbc_experiment"
OUT = CASE / "results" / "regularized_mdbc" / "rear_wall_contact"
STUDIES = (
    ("dbc_n80_half-step", "DBC", "#555555", "-"),
    ("control_n80_half-step", "SimpleMDBC", "#0072b2", "-"),
    ("regularized_n80_half-step", "RegularizedMDBC", "#d55e00", "-"),
    ("impedance_n80_half-step", "DBCPlusImpedance", "#009e73", "-"),
    ("regularized_n80_half-step-planar", "RegularizedMDBC + PlanarShifting", "#b14498", "--"),
)


def snapshot_metrics(points, velocity, density, pressure, markers, dp, h):
    """Use nominal face probes for geometry and outer-layer wall owners for kernel support."""
    fluid, square = markers == 2, markers == 3
    if fluid.sum() != 313600 or square.sum() != 6400:
        raise ValueError("Unexpected particle count in n80 frame")
    local = points - points[square].mean(axis=0)
    wall_speed = velocity[square, 0].mean(dtype=float)
    wall = local[square]
    rear = wall[np.isclose(wall[:, 0], wall[:, 0].min(), rtol=0, atol=1e-10)]
    rear = rear[np.argsort(rear[:, 1])]
    middle = np.abs(rear[:, 1]) <= 0.125
    if len(rear) != 80 or middle.sum() != 20:
        raise ValueError("Unexpected number of rear-wall probes")
    probes = rear.copy()
    probes[:, 0] = -0.5

    # Restrict the search to a box that contains the entire kernel stencil of each owner.
    radius = 2 * h
    region = fluid & (local[:, 0] < rear[0, 0] + radius + dp) & (np.abs(local[:, 1]) < 0.5 + radius + dp)
    water = local[region]
    tree = cKDTree(water)
    distance = tree.query(probes)[0]
    if np.any(distance > radius + dp):
        # A large void can put the nearest fluid outside the cropped stencil box.
        distance = cKDTree(local[fluid]).query(probes)[0]
    neighbors = tree.query_ball_point(rear, radius)
    counts = np.array([len(idx) for idx in neighbors])

    # Kernel fill uses the same Wendland C2 support and m/rho weights as the solver.
    alpha = 7 / (4 * np.pi * h**2)
    volume = 1000 * dp**2 / density[region].astype(float)
    fill = np.zeros(len(rear))
    for i, idx in enumerate(neighbors):
        q = np.linalg.norm(water[idx] - rear[i], axis=1) / h
        fill[i] = np.sum(volume[idx] * alpha * (1 - q / 2)**4 * (2*q + 1))
    participating = np.unique(np.concatenate([np.asarray(neighbors[i], dtype=int) for i in np.flatnonzero(middle)]))
    incoming = velocity[region, 0][participating] - wall_speed
    local_pressure = pressure[region][participating]
    return {
        "median_middle_face_distance_dp": np.median(distance[middle]) / dp,
        "maximum_middle_face_distance_dp": distance[middle].max() / dp,
        "middle_without_fluid_neighbors": np.count_nonzero(counts[middle] == 0),
        "rear_without_fluid_neighbors": np.count_nonzero(counts == 0),
        "median_middle_fluid_neighbors": np.median(counts[middle]),
        "median_middle_kernel_support": np.median(fill[middle]),
        "maximum_middle_approach_mps": max(0.0, incoming.max()) if len(incoming) else np.nan,
        "maximum_middle_fluid_pressure_kpa": local_pressure.max(initial=-np.inf) / 1000 if len(local_pressure) else np.nan,
    }


def read_study(tag, label, meta, *, include_mean_pressure=False):
    directory = RUNS / tag
    force = np.genfromtxt(directory / "native.csv", delimiter=",", names=True)
    rows = []
    with h5py.File(directory / "state" / "MovingSquare.vtkhdf") as handle:
        root = handle["VTKHDF/Particles"] if "VTKHDF/Particles" in handle else handle["VTKHDF"]
        times = root["Steps/Values"][:]
        np.testing.assert_array_equal(times, force["time"])
        for i, (offset, count) in enumerate(zip(root["Steps/PointOffsets"][:], root["NumberOfPoints"][:])):
            span = slice(int(offset), int(offset + count))
            pressure = root["PointData/Pressure"][span]
            markers = root["PointData/GroupMarker"][span]
            metrics = snapshot_metrics(
                root["Points"][span, :2], root["PointData/Velocity"][span, :2],
                root["PointData/Density"][span], pressure, markers, meta["dx"], meta["h"],
            )
            if include_mean_pressure:
                metrics["mean_fluid_pressure_kpa"] = pressure[markers == 2].mean(dtype=float) / 1000
            rows.append(dict(model=label, time_s=times[i], pressure_cd=force["cd_pressure"][i],
                             viscous_cd=force["cd_viscous"][i], **metrics))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, grid = plt.subplots(2, 3, figsize=(14, 7), layout="constrained", sharex=True)
    fields = (
        ("median_middle_face_distance_dp", "Median distance to fluid [dp]"),
        ("middle_without_fluid_neighbors", "Central owners without fluid [of 20]"),
        ("median_middle_kernel_support", "Median fluid kernel support"),
        ("maximum_middle_approach_mps", "Peak approach speed [m/s]"),
        ("maximum_middle_fluid_pressure_kpa", "Peak fluid pressure [kPa]"),
        ("pressure_cd", "Pressure drag Cd"),
    )
    rows = []
    common = None
    for tag, label, color, style in STUDIES:
        meta = tomllib.loads((RUNS / tag / "native.toml").read_text())
        settings = tuple(meta[key] for key in ("n", "reynolds", "dx", "h", "cfl", "sound_speed",
                         "motion", "diffusion", "density_diffusion_coefficient", "background_pressure",
                         "timestep_criterion", "wall_layers", "steps", "frames"))
        if common is None:
            common = settings
        if settings != common:
            raise ValueError(f"Unmatched startup settings in {tag}")
        expected_shift = "PlanarShifting" if tag.endswith("planar") else "NoShifting"
        if meta["shifting_mode"] != expected_shift:
            raise ValueError(f"Unexpected shifting mode in {tag}")
        records = read_study(tag, label, meta)
        rows.extend(records)
        for ax, (key, ylabel) in zip(grid.flat, fields):
            ax.plot([r["time_s"] for r in records], [r[key] for r in records],
                    color=color, ls=style, marker=".", ms=3, label=label)
            ax.set(xlim=(0.3, 0.9), xlabel="Time [s]", ylabel=ylabel)
            ax.grid(alpha=0.2)
    for ax in grid.flat:
        ax.axvline(0.88, color="0.5", lw=0.75, ls=":")
    grid[0, 0].legend(fontsize=8)
    fig.suptitle("Rear-wall contact and the startup spike\nCentral 0.25 m of the rear face · Re = 100 · L/dp = 80 · CFL = 0.10")
    fig.savefig(OUT / "rear_wall_contact.png", dpi=180)
    plt.close(fig)
    with (OUT / "rear_wall_contact.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
