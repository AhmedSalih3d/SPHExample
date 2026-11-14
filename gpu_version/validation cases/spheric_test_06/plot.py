"""Render Julia validation results with matplotlib; no solver or metric logic.

From gpu_version:
    python "validation cases/spheric_test_06/plot.py"

numpy/matplotlib are needed for drag plots; h5py is optional for local particle
snapshots. The Julia solver does not require these Python packages.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def drag_figure(results):
    rows = read_rows(results / "binned.csv")
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
    colors = {20: "#d47828", 40: "#248f8d", 60: "#5751ad"}
    for row_index, reynolds in enumerate((50, 100, 150)):
        for column, component in enumerate(("pressure", "viscous")):
            ax = axes[row_index, column]
            selected = [row for row in rows if int(row["reynolds"]) == reynolds and row["component"] == component]
            spacings = sorted({float(row["dx"]) for row in selected}, reverse=True)
            for spacing in spacings:
                data = [row for row in selected if float(row["dx"]) == spacing]
                n = round(1 / spacing)
                ax.plot([float(row["time"]) for row in data], [float(row["prediction"]) for row in data],
                        color=colors.get(n), lw=1.5, label=f"SPH, L/dp = {n}")
            if spacings:
                data = [row for row in selected if float(row["dx"]) == min(spacings)]
                ax.plot([float(row["time"]) for row in data], [float(row["reference"]) for row in data],
                        color="#202c39", lw=1.7, linestyle="--", label="Finite difference reference")
            ax.set_title(f"Re = {reynolds}: {component} drag", loc="left", fontsize=11)
            ax.set_ylabel("$C_{D,p}$" if component == "pressure" else "$C_{D,v}$")
            ax.grid(alpha=0.2)
            ax.set_xlim(0, 8)
            if row_index == 2:
                ax.set_xlabel("Time [s]")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("SPHERIC Test 06 — moving square\nEqual 0.1 s time bins; Umax = 1, L = 1", fontsize=14)
    fig.savefig(results / "drag_comparison.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
    for row_index, reynolds in enumerate((50, 100, 150)):
        candidates = sorted(results.glob(f"Re{reynolds:03d}_n*.csv"))
        candidates = [path for path in candidates if not path.stem.endswith("_state")]
        if not candidates:
            continue
        data = read_rows(candidates[-1])
        for column, component in enumerate(("pressure", "viscous")):
            ax = axes[row_index, column]
            ax.plot([float(row["time"]) for row in data], [float(row[f"cd_{component}"]) for row in data],
                    color="#5751ad", lw=0.65, alpha=0.7, label="Raw SPH, finest spacing")
            selected = [row for row in rows if int(row["reynolds"]) == reynolds and row["component"] == component]
            fine = min(float(row["dx"]) for row in selected)
            selected = [row for row in selected if float(row["dx"]) == fine]
            ax.plot([float(row["time"]) for row in selected], [float(row["reference"]) for row in selected],
                    color="#202c39", lw=1.5, linestyle="--", label="Binned finite difference reference")
            ax.set_title(f"Re = {reynolds}: {component} drag", loc="left", fontsize=11)
            ax.set_ylabel("$C_{D,p}$" if component == "pressure" else "$C_{D,v}$")
            ax.grid(alpha=0.2)
            ax.set_xlim(0, 8)
            if row_index == 2:
                ax.set_xlabel("Time [s]")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("SPHERIC Test 06 — unfiltered SPH drag", fontsize=14)
    fig.savefig(results / "raw_drag.png", dpi=170)
    plt.close(fig)


def field_figure(results, runs):
    import h5py

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for row_index, reynolds in enumerate((50, 100, 150)):
        files = sorted(runs.glob(f"Re{reynolds:03d}_n*/MovingSquare.vtkhdf"))
        if not files:
            plt.close(fig)
            return
        with h5py.File(files[-1], "r") as file:
            root = file["VTKHDF"]
            times = root["Steps/Values"][:]
            for column, target in enumerate((5.0, 8.0)):
                frame = int(np.argmin(abs(times-target)))
                if abs(times[frame]-target) > 1e-7:
                    plt.close(fig)
                    return
                offset = int(root["Steps/PointOffsets"][frame])
                count = int(root["NumberOfPoints"][frame])
                region = slice(offset, offset+count)
                points = root["Points"][region, :2]
                velocity = root["PointData/Velocity"][region, :2]
                markers = root["PointData/GroupMarker"][region]
                fluid = markers == 2
                center = points[markers == 3].mean(axis=0)
                ax = axes[row_index, column]
                scatter = ax.scatter(points[fluid, 0], points[fluid, 1], c=np.linalg.norm(velocity[fluid], axis=1),
                                     s=0.35, cmap="viridis", vmin=0, vmax=2.5, rasterized=True)
                ax.add_patch(Rectangle(center-0.5, 1, 1, color="#293846"))
                ax.add_patch(Rectangle((0, 0), 10, 5, fill=False, edgecolor="#202c39", lw=1.1))
                ax.set(xlim=(-0.1, 10.1), ylim=(-0.1, 5.1), aspect="equal", xlabel="x [m]", ylabel="y [m]")
                ax.set_title(f"Re = {reynolds}, t = {target:g} s", loc="left", fontsize=11)
    fig.colorbar(scatter, ax=axes, shrink=0.75, label="Fluid speed [m/s]")
    fig.suptitle("SPHERIC Test 06 — finest SPH particle fields\nQualitative snapshots; drag metrics determine the comparison", fontsize=14)
    fig.savefig(results / "speed_fields.png", dpi=190)
    plt.close(fig)


def sound_speed_figure(results):
    directories = [(40, results), (80, results.parent / "sound_speed_80"), (160, results.parent / "sound_speed_160")]
    if not all((directory / "binned.csv").exists() for _, directory in directories):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for speed, directory in directories:
        rows = read_rows(directory / "binned.csv")
        for ax, component in zip(axes, ("pressure", "viscous")):
            selected = [row for row in rows if int(row["reynolds"]) == 100 and
                        round(1/float(row["dx"])) == 60 and row["component"] == component]
            ax.plot([float(row["time"]) for row in selected], [float(row["prediction"]) for row in selected],
                    lw=1.5, label=f"SPH, c0 = {speed} m/s")
            if speed == 160:
                ax.plot([float(row["time"]) for row in selected], [float(row["reference"]) for row in selected],
                        color="#202c39", lw=1.5, linestyle="--", label="Finite difference reference")
            ax.set(xlim=(0, 8), xlabel="Time [s]", title=f"{component.capitalize()} drag")
            ax.set_ylabel("$C_{D,p}$" if component == "pressure" else "$C_{D,v}$")
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Sound-speed sensitivity — Re = 100, L/dp = 60; equal 0.1 s time bins")
    fig.savefig(results / "sound_speed_comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="validation")
    parser.add_argument("--fields", action="store_true", help="Read local VTKHDF particle histories with h5py")
    args = parser.parse_args()
    case = Path(__file__).resolve().parent
    results = case / "results" / args.tag
    drag_figure(results)
    sound_speed_figure(results)
    if args.fields:
        field_figure(results, case / "runs" / args.tag)
