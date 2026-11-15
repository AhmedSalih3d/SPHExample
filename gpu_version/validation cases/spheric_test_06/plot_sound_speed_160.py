"""Render saved c0/timestep controls; numerical metrics are computed in Julia."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from plot_edge_damping import native_state, dual_state, reference

CASE = Path(__file__).resolve().parent
RESULTS = CASE / "results/c160_comparison"
WORK = CASE / "runs/c160_comparison"
ARCHIVE = CASE / "results/comparison"
COLORS = {"baseline40": "#929ba5", "standard160": "#167f86", "step_control40": "#be7b33",
          "cleaning_control160": "#be7b33", "cleaning160": "#167f86"}
LABELS = {"baseline40": "c0 = 40, CFL = 0.2", "standard160": "c0 = 160, CFL = 0.2",
          "step_control40": "c0 = 40, CFL = 0.05", "cleaning_control160": "c0 = 160, cleaning off",
          "cleaning160": "c0 = 160, cleaning on, κ = 160"}
SOLVERS = {"native": "Julia GPU", "dual": "DualSPHysics 5.4"}


def history(reynolds, solver, variant):
    if variant == "baseline40" or (variant == "standard160" and reynolds == 100):
        directory = ARCHIVE / f"re{reynolds}_c{40 if variant == 'baseline40' else 160}"
    else:
        directory = RESULTS / f"re{reynolds}_{variant}"
    with (directory / f"{solver}.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.asarray([float(row[key]) for row in rows])
            for key in ("time", "cd_pressure", "cd_viscous")}


def drag(component, zoom=False):
    fig, axes = plt.subplots(3, 2, figsize=(12, 9.5), sharex=True, constrained_layout=True)
    for r, re in enumerate((50, 100, 150)):
        ref = reference(re)
        for c, solver in enumerate(("native", "dual")):
            ax = axes[r, c]
            for variant in ("baseline40", "step_control40", "standard160"):
                data = history(re, solver, variant)
                selected = (data["time"] >= 2.5) & (data["time"] <= 3.5) if zoom else np.ones(len(data["time"]), dtype=bool)
                ax.plot(data["time"][selected], data[f"cd_{component}"][selected], color=COLORS[variant],
                        lw=0.95, label=LABELS[variant])
            selected = (ref[:, 0] >= 2.5) & (ref[:, 0] <= 3.5) if zoom else np.ones(len(ref), dtype=bool)
            ax.plot(ref[selected, 0], ref[selected, 1 if component == "pressure" else 2], color="#273643",
                    ls="--", lw=1.4, label="FD reference")
            ax.set(xlim=(2.5, 3.5) if zoom else (0, 8), title=f"Re = {re}: {SOLVERS[solver]}",
                   ylabel="$C_{D,p}$" if component == "pressure" else "$C_{D,v}$")
            ax.grid(alpha=0.2)
            if r == 2:
                ax.set_xlabel("Time [s]")
        limits = [ax.get_ylim() for ax in axes[r]]
        for ax in axes[r]:
            ax.set_ylim(min(lim[0] for lim in limits), max(lim[1] for lim in limits))
    axes[0, 0].legend(fontsize=7.5, ncols=2, frameon=False)
    fig.suptitle(f"Sound speed and timestep: raw {component} drag" + (" (zoom)" if zoom else "") +
                 "\nBoth stock solvers; no sponges or cleaning; orange and teal have equal nominal acoustic dt caps",
                 fontsize=12)
    fig.savefig(RESULTS / f"{component}_{'zoom' if zoom else 'drag'}.png", dpi=180)
    plt.close(fig)


def cleaning():
    fig, axes = plt.subplots(3, 2, figsize=(12, 9.5), sharex=True, constrained_layout=True)
    for r, re in enumerate((50, 100, 150)):
        ref = reference(re)
        for c, component in enumerate(("pressure", "viscous")):
            ax = axes[r, c]
            for variant in ("cleaning_control160", "cleaning160"):
                data = history(re, "native", variant)
                ax.plot(data["time"], data[f"cd_{component}"], color=COLORS[variant],
                        lw=0.9, label=LABELS[variant])
            ax.plot(ref[:, 0], ref[:, c+1], color="#273643", ls="--", lw=1.3, label="FD reference")
            ax.set(xlim=(0, 8), title=f"Re = {re}: {component}",
                   ylabel="$C_{D,p}$" if c == 0 else "$C_{D,v}$")
            ax.grid(alpha=0.2)
            if r == 2:
                ax.set_xlabel("Time [s]")
    axes[0, 0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("Optional Julia GPU cleaning at c0 = 160 m/s\nMatched physical CFL = 0.1; σ = 1; cleaning CFL = 0.1; absolute-pressure decay; no edge sponges", fontsize=12)
    fig.savefig(RESULTS / "cleaning_drag.png", dpi=180)
    plt.close(fig)


def metrics():
    with (RESULTS / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey="row", constrained_layout=True)
    for c, solver in enumerate(("native", "dual")):
        for variant, offset in (("baseline40", -0.25), ("step_control40", 0), ("standard160", 0.25)):
            selected = [next(row for row in rows if int(row["reynolds"]) == re and
                             row["solver"] == solver and row["variant"] == variant) for re in (50, 100, 150)]
            for r, (key, scale, label) in enumerate((("pressure_noise_rms", 1, "Pressure noise RMS [$C_D$]"),
                                                    ("pressure_raw_nrmse", 100, "Raw pressure NRMSE [%]"))):
                ax = axes[r, c]
                ax.bar(np.arange(3)+offset, [scale*float(row[key]) for row in selected], width=0.23,
                       color=COLORS[variant], label=LABELS[variant])
                ax.set(xticks=np.arange(3), xticklabels=(50, 100, 150), xlabel="Re", ylabel=label,
                       title=SOLVERS[solver] if r == 0 else None)
                ax.grid(axis="y", alpha=0.2)
        axes[0, c].legend(fontsize=8, frameon=False)
    fig.suptitle("Smaller timestep and higher sound speed have different effects\nNoise: high-pass error RMS, 1.5–6.5 s; NRMSE: full raw force error, 0.1–8 s", fontsize=12)
    fig.savefig(RESULTS / "noise_and_error.png", dpi=180)
    plt.close(fig)


def sensitivity():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ref = reference(100)
    variants = (("cleaning_control160", "Off", "#929ba5"),
                ("cleaning_k10_160", "κ = 10 (API default)", "#8a5aa4"),
                ("cleaning_k40_160", "κ = 40", "#be7b33"), ("cleaning160", "κ = 160", "#167f86"))
    for c, component in enumerate(("pressure", "viscous")):
        ax = axes[c]
        for variant, label, color in variants:
            data = history(100, "native", variant)
            ax.plot(data["time"], data[f"cd_{component}"], color=color, lw=0.9, label=label)
        ax.plot(ref[:, 0], ref[:, c+1], color="#273643", ls="--", lw=1.3, label="FD reference")
        ax.set(xlim=(0, 8), xlabel="Time [s]", title=f"Raw {component} drag",
               ylabel="$C_{D,p}$" if c == 0 else "$C_{D,v}$")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, frameon=False, ncols=2)
    fig.suptitle("Local-decay sensitivity at c0 = 160 m/s, Re = 100\nJulia GPU; matched physical CFL = 0.1; σ = 1; cleaning Courant = 0.1", fontsize=12)
    fig.savefig(RESULTS / "cleaning_sensitivity.png", dpi=180)
    plt.close(fig)


def fields():
    states = {}
    for re in (50, 100, 150):
        directory = CASE / "runs/comparison/re100_c160" if re == 100 else WORK / f"re{re}_standard160"
        states[re, "native"] = native_state(directory / "native/MovingSquare.vtkhdf")
        states[re, "dual"] = dual_state(directory / "dual/fields_0400.csv")
    maximum_speed = np.ceil(10*max(np.linalg.norm(v[fluid], axis=1).max()
                                   for _, v, fluid, _ in states.values()))/10
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for r, re in enumerate((50, 100, 150)):
        for c, solver in enumerate(("native", "dual")):
            ax = axes[r, c]
            p, v, fluid, moving = states[re, solver]
            center = p[moving].mean(axis=0)
            dots = ax.scatter(p[fluid, 0], p[fluid, 1], c=np.linalg.norm(v[fluid], axis=1), s=0.5,
                              vmin=0, vmax=maximum_speed, cmap="viridis", rasterized=True)
            ax.add_patch(Rectangle(center-0.5, 1, 1, color="#273643"))
            ax.add_patch(Rectangle((0, 0), 10, 5, fill=False, edgecolor="#273643", lw=1))
            ax.set(xlim=(-0.1, 10.1), ylim=(-0.1, 5.1), aspect="equal", xlabel="x [m]",
                   ylabel="Physical y [m]", title=f"Re = {re}: {SOLVERS[solver]}")
    fig.colorbar(dots, ax=axes, shrink=0.8, label="Fluid speed [m/s], shared scale")
    fig.suptitle("Current velocity fields at t = 8 s, c0 = 160 m/s\nCFL = 0.2; no edge sponges; cleaning disabled in both solvers", fontsize=12)
    fig.savefig(RESULTS / "velocity_fields.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    drag("pressure")
    drag("viscous")
    drag("pressure", zoom=True)
    cleaning()
    sensitivity()
    metrics()
    fields()
