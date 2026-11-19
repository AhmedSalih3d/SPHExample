"""Visualize saved fluid particle centers and pressure during rear-wall recontact."""
from functools import lru_cache
import tomllib

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial import cKDTree

from audit_rear_wall_contact import OUT, RUNS


UNSHIFTED = "regularized_n80_half-step"
SHIFTED = "regularized_n80_half-step-planar"
TIMES = (0.74, 0.76, 0.82, 0.88)
X_LIMITS = (-150, 30)
CONTACT_Y_LIMITS = (-132, 132)
PRESSURE_Y_LIMITS = (-235, 235)
FLUID = "#4580a5"
IN_SUPPORT = "#d05d21"
WALL = "#393f4d"


@lru_cache
def read_frame(run, time):
    folder = RUNS / run
    meta = tomllib.loads((folder / "native.toml").read_text())
    if meta["n"] != 80 or not np.isclose(meta["cfl"], 0.10):
        raise ValueError(f"Expected n80 half-timestep data in {folder}")
    with h5py.File(folder / "state" / "MovingSquare.vtkhdf") as handle:
        root = handle["VTKHDF/Particles"] if "VTKHDF/Particles" in handle else handle["VTKHDF"]
        frame = np.flatnonzero(np.isclose(root["Steps/Values"][:], time, atol=1e-12, rtol=0)).item()
        offset = int(root["Steps/PointOffsets"][frame])
        span = slice(offset, offset + int(root["NumberOfPoints"][frame]))
        markers = root["PointData/GroupMarker"][span]
        assert tuple(np.bincount(markers)) == (0, 12100, 313600, 6400)
        pos = root["Points"][span, :2]
        pressure = root["PointData/Pressure"][span][markers == 2] / 1000
    square = pos[markers == 3]
    origin = square.mean(axis=0) - np.array([0.5, 0])
    rear = (square - origin) * 1000
    central = rear[np.isclose(rear[:, 0], rear[:, 0].min(), atol=1e-7, rtol=0) & (abs(rear[:, 1]) <= 125)]
    assert len(central) == 20 and np.isfinite(pressure).all()
    return (pos[markers == 2] - origin) * 1000, pressure, central, 2000 * meta["h"]


def format_panel(ax, y_limits):
    ax.set(xlim=X_LIMITS, ylim=y_limits, aspect="equal", xticks=(-120, -80, -40, 0))
    ax.axvspan(0, X_LIMITS[1], color="#e8ebef", zorder=0)
    ax.axvline(0, lw=1.1, color=WALL, zorder=2)


def in_view(points, y_limits):
    return ((points[:, 0] >= X_LIMITS[0]) & (points[:, 0] <= X_LIMITS[1]) &
            (points[:, 1] >= y_limits[0]) & (points[:, 1] <= y_limits[1]))


def plot_sequence():
    fig, axes = plt.subplots(2, 4, figsize=(11.3, 8.0), sharex=True, sharey=True, layout="constrained")
    for row, (run, label) in enumerate(((UNSHIFTED, "No shifting"), (SHIFTED, "PlanarShifting"))):
        axes[row, 0].set_ylabel(f"{label}\ny from rear-face midpoint [mm]")
        for ax, time in zip(axes[row], TIMES):
            fluid, _, wall, radius = read_frame(run, time)
            tree = cKDTree(fluid)
            has_support = tree.query_ball_point(wall, radius, return_length=True) > 0
            probes = wall.copy()
            probes[:, 0] = 0
            median_gap = np.median(tree.query(probes)[0])
            local = fluid[in_view(fluid, CONTACT_Y_LIMITS)]
            neighbors = cKDTree(wall).query(local)[0] <= radius
            format_panel(ax, CONTACT_Y_LIMITS)
            ax.set_title(f"{time:.2f} s")
            ax.axvline(wall[0, 0] - radius, color="#68717f", lw=1.0, ls="--")
            ax.scatter(*local[~neighbors].T, s=7, color=FLUID, linewidths=0)
            ax.scatter(*local[neighbors].T, s=10, color=IN_SUPPORT, linewidths=0, zorder=3)
            ax.scatter(*wall.T, s=9, color=WALL, marker="s", linewidths=0, zorder=3)
            ax.text(0.04, 0.96, f"{has_support.sum()}/20 supported\n{median_gap:.1f} mm median gap",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=2))

    handles = [
        Line2D([], [], marker="o", color=FLUID, ls="", label="Fluid outside support"),
        Line2D([], [], marker="o", color=IN_SUPPORT, ls="", label="Fluid within support"),
        Line2D([], [], marker="s", color=WALL, ls="", label="Central wall particles"),
        Line2D([], [], color="#68717f", ls="--", label="Farthest reach of wall kernel"),
    ]
    fig.suptitle("Rear-wall gap and recontact\nRegularizedMDBC · Re = 100 · L/dp = 80 · CFL = 0.10")
    fig.get_layout_engine().set(rect=(0, 0.062, 1, 0.938))
    fig.supxlabel("Distance from the rear face [mm] · square to the right", y=0.035)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.002),
               ncols=4, fontsize=9, frameon=False)
    fig.savefig(OUT / "rear_wall_recontact_sequence.png", dpi=200)
    plt.close(fig)


def plot_pressure():
    panels = (
        ("dbc_n80_half-step", "DBC", 0.84),
        (UNSHIFTED, "RegularizedMDBC\nNo shifting", 0.88),
        (SHIFTED, "RegularizedMDBC\nPlanarShifting", 0.76),
        ("impedance_n80_half-step", "DBCPlusImpedance", 0.88),
    )
    snapshots = []
    for run, _, time in panels:
        fluid, pressure, _, _ = read_frame(run, time)
        visible = in_view(fluid, PRESSURE_Y_LIMITS)
        snapshots.append((fluid[visible], pressure[visible]))
    limit = 10 * np.ceil(max(np.abs(pressure).max() for _, pressure in snapshots) / 10)
    norm = SymLogNorm(linthresh=5, vmin=-limit, vmax=limit)

    fig, axes = plt.subplots(1, 4, figsize=(10.9, 6.1), sharex=True, sharey=True, layout="constrained")
    for ax, (_, label, time), (fluid, pressure) in zip(axes, panels, snapshots):
        format_panel(ax, PRESSURE_Y_LIMITS)
        ax.set_title(f"{label}\n{time:.2f} s", fontsize=10)
        image = ax.scatter(*fluid.T, c=pressure, s=13, marker="s", linewidths=0, cmap="RdBu_r", norm=norm)
        peak = int(np.argmax(pressure))
        ax.scatter(*fluid[peak], s=62, facecolors="none", edgecolors=WALL, linewidths=1, zorder=3)
        ax.text(0.05, 0.96, f"Peak {pressure.max():.1f} kPa", transform=ax.transAxes, va="top",
                fontsize=9, bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=2))
    axes[0].set_ylabel("y from rear-face midpoint [mm]")
    fig.supxlabel("Distance from the rear face [mm] · square to the right")
    fig.suptitle("Fluid pressure near the rear face\nEach model at its spike, with impedance at 0.88 s for comparison")
    fig.colorbar(image, ax=axes, ticks=[-limit, -20, -5, 0, 5, 20, limit], format="%g", shrink=0.9,
                 label="Fluid gauge pressure [kPa] · symmetric log scale")
    fig.savefig(OUT / "rear_wall_spike_pressure.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plot_sequence()
    plot_pressure()
