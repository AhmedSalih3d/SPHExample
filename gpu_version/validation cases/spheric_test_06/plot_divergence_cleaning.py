"""Render measured Julia GPU cleaning controls; no simulation or filtering here."""
import csv
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

CASE = Path(__file__).resolve().parent
RESULTS = CASE / "results/divergence_cleaning"
WORK = CASE / "runs/divergence_cleaning"
COLORS = {"off": "#9b6e45", "on": "#167f86", "k10": "#7954a1"}
LABELS = {"off": "Cleaning off, CFL = 0.1", "on": "Cleaning on, κ = 40", "k10": "Cleaning on, κ = 10"}


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def reference(reynolds):
    values = []
    for line in (CASE / f"reference/Force_Re{reynolds:03d}.dat").read_text().splitlines():
        try:
            value = [float(x) for x in line.split()]
            if len(value) == 3:
                values.append(value)
        except ValueError:
            pass
    return np.asarray(values)


def drag():
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
    for r, re in enumerate((50, 100, 150)):
        ref = reference(re)
        for c, component in enumerate(("pressure", "viscous")):
            ax = axes[r, c]
            original = rows(CASE / f"results/comparison/re{re}_c40/native.csv")
            ax.plot([float(row["time"]) for row in original], [float(row[f"cd_{component}"]) for row in original],
                    color="#a9aeb5", lw=0.7, ls=":", label="Earlier off, CFL = 0.2")
            for variant in ("off", "on"):
                data = rows(RESULTS / f"re{re}_{variant}/native.csv")
                ax.plot([float(row["time"]) for row in data], [float(row[f"cd_{component}"]) for row in data],
                        color=COLORS[variant], lw=0.95, label=LABELS[variant])
            ax.plot(ref[:, 0], ref[:, c+1], color="#273643", ls="--", lw=1.3, label="FD reference")
            ax.set(xlim=(0, 8), title=f"Re = {re}: {component} drag", ylabel="$C_{D,p}$" if c == 0 else "$C_{D,v}$")
            ax.grid(alpha=0.2)
            if r == 2:
                ax.set_xlabel("Time [s]")
    axes[0, 0].legend(fontsize=7.5, ncols=2, frameon=False)
    fig.suptitle("Optional GPU divergence cleaning: raw moving-square forces\nNo edge sponges; c0 = 40 m/s; L/dp = 40; matched CFL = 0.1 controls", fontsize=13)
    fig.savefig(RESULTS / "drag_comparison.png", dpi=180)
    plt.close(fig)


def sensitivity():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ref = reference(100)
    for c, component in enumerate(("pressure", "viscous")):
        for variant in ("off", "k10", "on"):
            data = rows(RESULTS / f"re100_{variant}/native.csv")
            axes[0, c].plot([float(row["time"]) for row in data], [float(row[f"cd_{component}"]) for row in data],
                            color=COLORS[variant], lw=0.8, label=LABELS[variant])
            spectrum = [row for row in rows(RESULTS / f"re100_{variant}/spectra.csv") if row["component"] == component]
            axes[1, c].plot([float(row["frequency"]) for row in spectrum], [float(row["amplitude"]) for row in spectrum],
                            color=COLORS[variant], lw=1)
        axes[0, c].plot(ref[:, 0], ref[:, c+1], color="#273643", ls="--", lw=1.1, label="FD reference")
        axes[0, c].set(xlim=(0, 8), xlabel="Time [s]", ylabel="$C_D$", title=f"Re = 100: {component} drag")
        axes[1, c].set(xlim=(0, 15), xlabel="Frequency [Hz]", ylabel="Error amplitude [$C_D$]", title="Error spectrum, 1.5–6.5 s")
        for ax in axes[:, c]:
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Local-decay sensitivity: κ = c0/Uref = 40 versus the API default κ = 10\nBoth use σ = 1 and cleaning Courant = 0.1", fontsize=12)
    fig.savefig(RESULTS / "kappa_sensitivity.png", dpi=180)
    plt.close(fig)


def pressure_extensions():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, constrained_layout=True)
    for ax, re in zip(axes, (50, 100, 150)):
        ref = reference(re)
        for variant, label, color in (("off", "Off", COLORS["off"]),
                                      ("positive", "Positive-only decay speed", "#ac4c54"),
                                      ("on", "Absolute-pressure decay speed", COLORS["on"])):
            data = rows(RESULTS / f"re{re}_{variant}/native.csv")
            ax.plot([float(row["time"]) for row in data], [float(row["cd_pressure"]) for row in data],
                    color=color, lw=0.85, label=label)
        ax.plot(ref[:, 0], ref[:, 1], color="#273643", ls="--", lw=1, label="FD reference")
        ax.set(xlim=(0, 1.5), xlabel="Time [s]", title=f"Re = {re}")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Pressure $C_D$")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("Startup behavior depends strongly on the negative-pressure extension\nκ = 40 in both cleaning cases; the paper does not specify handling of negative gauge pressures", fontsize=12)
    fig.savefig(RESULTS / "pressure_extensions.png", dpi=180)
    plt.close(fig)


def pressure_zoom():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, constrained_layout=True)
    for ax, re in zip(axes, (50, 100, 150)):
        ref = reference(re)
        for variant in ("off", "on"):
            data = rows(RESULTS / f"re{re}_{variant}/native.csv")
            ax.plot([float(row["time"]) for row in data], [float(row["cd_pressure"]) for row in data],
                    color=COLORS[variant], lw=0.9, label=LABELS[variant])
        ax.plot(ref[:, 0], ref[:, 1], color="#273643", ls="--", lw=1.1, label="FD reference")
        ax.set(xlim=(1.5, 6.5), ylim=(0.7, 4.5), xlabel="Time [s]", title=f"Re = {re}")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Pressure $C_D$")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Raw pressure drag in the 1.5–6.5 s noise window\nNo signal filtering; full transients and startup failures are shown separately", fontsize=12)
    fig.savefig(RESULTS / "pressure_zoom.png", dpi=180)
    plt.close(fig)


def divergence():
    data = rows(RESULTS / "divergence.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True, constrained_layout=True)
    for ax, re in zip(axes, (50, 100, 150)):
        for variant in ("off", "on"):
            selected = [row for row in data if int(row["reynolds"]) == re and row["variant"] == variant]
            ax.plot([float(row["time"]) for row in selected], [float(row["divergence_volume_rms"]) for row in selected],
                    color=COLORS[variant], lw=1.3, label=LABELS[variant])
        ax.set(xlim=(0, 8), xlabel="Time [s]", title=f"Re = {re}")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Volume-weighted RMS div(u) [s⁻¹]")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Velocity divergence measured independently on saved states\nRaw SPH difference operator, including wall velocities; density diffusion excluded", fontsize=12)
    fig.savefig(RESULTS / "divergence_rms.png", dpi=180)
    plt.close(fig)


def final_state(path):
    with h5py.File(path, "r") as file:
        root = file["VTKHDF"]
        offset, count = int(root["Steps/PointOffsets"][-1]), int(root["NumberOfPoints"][-1])
        part = slice(offset, offset+count)
        p = root["Points"][part, :2]
        v = root["PointData/Velocity"][part, :2]
        marker = root["PointData/GroupMarker"][part]
    return p, v, marker == 2, marker == 3


def fields():
    states = {(re, variant): final_state(WORK / f"re{re}_{variant}/MovingSquare.vtkhdf")
              for re in (50, 100, 150) for variant in ("off", "on")}
    maximum_speed = np.ceil(10*max(np.linalg.norm(v[fluid], axis=1).max() for _, v, fluid, _ in states.values()))/10
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for r, re in enumerate((50, 100, 150)):
        for c, variant in enumerate(("off", "on")):
            p, v, fluid, square = states[re, variant]
            ax = axes[r, c]
            dots = ax.scatter(p[fluid, 0], p[fluid, 1], c=np.linalg.norm(v[fluid], axis=1), s=0.45,
                              vmin=0, vmax=maximum_speed, cmap="viridis", rasterized=True)
            center = p[square].mean(axis=0)
            ax.add_patch(Rectangle(center-0.5, 1, 1, color="#273643"))
            ax.add_patch(Rectangle((0, 0), 10, 5, fill=False, edgecolor="#273643", lw=1))
            ax.set(xlim=(-0.1, 10.1), ylim=(-0.1, 5.1), aspect="equal", xlabel="x [m]", ylabel="y [m]",
                   title=f"Re = {re}: cleaning {variant}")
    fig.colorbar(dots, ax=axes, shrink=0.75, label="Fluid speed [m/s]")
    fig.suptitle("Current GPU fields at t = 8 s, with and without divergence cleaning\nMatched CFL controls; κ = 40; same speed scale in all panels; no edge damping", fontsize=12)
    fig.savefig(RESULTS / "velocity_fields.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    drag()
    sensitivity()
    pressure_extensions()
    pressure_zoom()
    divergence()
    fields()
