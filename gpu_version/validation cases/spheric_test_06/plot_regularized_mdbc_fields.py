"""Render unfiltered fluid particles in the matched half-timestep spike frame."""
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import matplotlib.patheffects as effects
import numpy as np
from scipy.spatial import cKDTree


CASE = Path(__file__).resolve().parent
RUNS = CASE / "runs" / "regularized_mdbc_experiment"
OUT = CASE / "results" / "regularized_mdbc" / "half-step"
TIME = 0.88
X_LIMITS = (-1.4, 1.4)
Y_LIMITS = (-1.05, 1.05)
MODELS = (("dbc", "DBC (NoMDBC)"), ("control", "SimpleMDBC"),
          ("regularized", "RegularizedMDBC"), ("impedance", "DBCPlusImpedance"))


def read_frame(run, time=TIME):
    path = RUNS / run / "state" / "MovingSquare.vtkhdf"
    with h5py.File(path) as handle:
        root = handle["VTKHDF/Particles"] if "VTKHDF/Particles" in handle else handle["VTKHDF"]
        frames = np.flatnonzero(np.isclose(root["Steps/Values"][:], time, rtol=0, atol=1e-12))
        frame = frames.item()
        start = int(root["Steps/PointOffsets"][frame])
        span = slice(start, start + int(root["NumberOfPoints"][frame]))
        markers = root["PointData/GroupMarker"][span]
        assert tuple(np.bincount(markers)) == (0, 12100, 313600, 6400)
        positions = root["Points"][span, :2]
        relative = positions[markers == 2] - positions[markers == 3].mean(axis=0)
        inside = ((relative[:, 0] >= X_LIMITS[0]) & (relative[:, 0] <= X_LIMITS[1]) &
                  (relative[:, 1] >= Y_LIMITS[0]) & (relative[:, 1] <= Y_LIMITS[1]))
        velocity = root["PointData/Velocity"][span, :2][markers == 2][inside]
        pressure = root["PointData/Pressure"][span][markers == 2][inside] / 1000
    assert np.isfinite(velocity).all() and np.isfinite(pressure).all()
    return relative[inside], velocity, pressure


def velocity_arrows(ax, points, velocity):
    x, y = np.meshgrid(np.arange(-1.26, 1.3, 0.18), np.arange(-0.9, 0.95, 0.18))
    sample_points = np.column_stack((x.ravel(), y.ravel()))
    distance, indices = cKDTree(points).query(sample_points, distance_upper_bound=0.06)
    indices = indices[np.isfinite(distance)]
    indices = indices[np.linalg.norm(velocity[indices], axis=1) >= 0.08]
    scale = 8
    segment = points[indices, None, :] + velocity[indices, None, :] * np.linspace(0, 1, 5)[None, :, None] / scale
    indices = indices[(np.max(np.abs(segment), axis=2) > 0.52).all(axis=1)]
    arrows = ax.quiver(*points[indices].T, *velocity[indices].T, color="white",
                       angles="xy", scale_units="xy", scale=scale, width=0.004, zorder=5)
    arrows.set_path_effects([effects.withStroke(linewidth=0.8, foreground="#263648")])


def plot_fields(snapshots, field, *, models=MODELS, out=OUT, subject="", time=TIME):
    speed = field == "speed"
    values = [np.linalg.norm(velocity, axis=1) if speed else pressure
              for _, velocity, pressure in snapshots]
    low, high = min(v.min() for v in values), max(v.max() for v in values)
    norm = Normalize(0, np.ceil(2 * high) / 2) if speed else TwoSlopeNorm(
        vmin=2 * np.floor(low / 2), vcenter=0, vmax=2 * np.ceil(high / 2))
    rows = len(models) // 2
    fig, grid = plt.subplots(rows, 2, figsize=(10.8, 4.2 * rows), sharex=True, sharey=True, layout="constrained")
    axes = grid.ravel()
    for ax, (_, title), (position, velocity, _), value in zip(axes, models, snapshots, values):
        ax.set(xlim=X_LIMITS, ylim=Y_LIMITS, aspect="equal", title=title, facecolor="#edf1f4")
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, facecolor="#e0e5eb", edgecolor="#334155", zorder=1))
        points = ax.scatter(*position.T, c=value, s=2.6, marker="s", linewidths=0,
                            norm=norm, cmap="viridis" if speed else "RdBu_r", zorder=2)
        ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor="#334155", linewidth=0.8, zorder=3))
        ax.text(0, 0.11, "Square", color="#334155", ha="center", va="center", fontsize=10)
        ax.annotate("", xy=(0.16, -0.06), xytext=(-0.16, -0.06),
                    arrowprops=dict(arrowstyle="->", color="#334155", lw=1.4))
        if speed:
            velocity_arrows(ax, position, velocity)
    fig.supxlabel("x relative to the square center [m]")
    fig.supylabel("y relative to the square center [m]")
    fig.colorbar(points, ax=axes, shrink=0.85, pad=0.015,
                 label="Fluid speed [m/s]" if speed else "Fluid gauge pressure [kPa]")
    title = "Fluid speed and direction" if speed else "Fluid pressure"
    when = "" if time is None else f" at {time:.2f} s"
    suffix = "_comparison" if time is None else f"_at_{round(100 * time):03d}"
    fig.suptitle(f"{title}{when}{subject}\nSPHERIC Test 06 · Re = 100 · L/dp = 80 · CFL = 0.10", fontsize=14)
    fig.savefig(out / f"{field}{suffix}.png", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [read_frame(f"{model}_n80_half-step") for model, _ in MODELS]
    plot_fields(frames, "speed")
    plot_fields(frames, "pressure")
