"""Plot matched moving-square runs without filtering the displayed force histories."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE = Path(__file__).resolve().parent
RESULTS = CASE / "results/pairwise_timestep"

def history(reynolds, speed, criterion):
    with (RESULTS / f"re{reynolds}_c{speed}_{criterion}/native.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    data = {key: np.asarray([float(row[key]) for row in rows]) for key in ("time", "cd_pressure", "cd_viscous")}
    data["cd_total"] = data["cd_pressure"] + data["cd_viscous"]
    return data

def reference(reynolds):
    rows = []
    for line in (CASE / f"reference/Force_Re{reynolds:03d}.dat").read_text().splitlines():
        try:
            values = [float(x) for x in line.split()]
            if len(values) == 3:
                rows.append(values)
        except ValueError:
            pass
    data = np.asarray(rows)
    return {"time": data[:, 0], "cd_pressure": data[:, 1], "cd_viscous": data[:, 2], "cd_total": data[:, 1] + data[:, 2]}

def force_plot(component, zoom=False):
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
    for row, re in enumerate((50, 100, 150)):
        for col, speed in enumerate((40, 160)):
            ax = axes[row, col]
            for criterion, color, label in (("legacy", "#9a9a9a", "Legacy timestep"), ("pairwise", "#168b86", "New default")):
                data = history(re, speed, criterion)
                selected = (data["time"] >= 2.5) & (data["time"] <= 3.5) if zoom else np.ones(len(data["time"]), dtype=bool)
                ax.plot(data["time"][selected], data[f"cd_{component}"][selected], color=color, lw=0.95, label=label)
            data = reference(re)
            selected = (data["time"] >= 2.5) & (data["time"] <= 3.5) if zoom else np.ones(len(data["time"]), dtype=bool)
            ax.plot(data["time"][selected], data[f"cd_{component}"][selected], color="#222222", lw=1.3, ls="--", label="Official FD reference")
            ax.set(xlim=(2.5, 3.5) if zoom else (0, 8), title=f"Re = {re}, c0 = {speed} m/s", ylabel=f"Drag coefficient ({component})")
            ax.grid(alpha=0.2)
            if row == 2:
                ax.set_xlabel("Time (s)")
        limits = [ax.get_ylim() for ax in axes[row]]
        for ax in axes[row]:
            ax.set_ylim(min(l[0] for l in limits), max(l[1] for l in limits))
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Moving square: raw {component} drag" + (" (zoom)" if zoom else "") + "\nOnly timestep criterion changed; dp = 0.025 m, CFL = 0.2; no cleaning or damping")
    fig.savefig(RESULTS / f"{component}_{'zoom' if zoom else 'drag'}.png", dpi=170)
    plt.close(fig)

def errors():
    with (RESULTS / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, component in zip(axes, ("pressure", "viscous", "total")):
        for speed, color in ((40, "#4477aa"), (160, "#cc6677")):
            for criterion, marker, style in (("legacy", "o", "--"), ("pairwise", "s", "-")):
                selected = sorted((r for r in rows if int(r["sound_speed"]) == speed and r["criterion"] == criterion), key=lambda r: int(r["reynolds"]))
                ax.plot([int(r["reynolds"]) for r in selected], [100 * float(r[f"{component}_raw_nrmse"]) for r in selected], marker=marker, ls=style, color=color, label=f"c0={speed}, {criterion}")
        ax.set(title=component.capitalize(), xlabel="Reynolds number", ylabel="Relative L2 force-history error (%)", xticks=[50, 100, 150])
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Force accuracy against official FD histories, 0.1–8 s\nUnfiltered output histories; smaller is better")
    fig.savefig(RESULTS / "force_errors.png", dpi=170)
    plt.close(fig)

def validation_checks():
    checks = []
    for speed in (40, 160):
        for re in (50, 100, 150):
            for criterion in ("legacy", "pairwise"):
                directory = RESULTS / f"re{re}_c{speed}_{criterion}"
                with (directory / "native_state.csv").open(newline="") as stream:
                    state = list(csv.DictReader(stream))
                row = dict(reynolds=re, sound_speed=speed, criterion=criterion,
                    density_relative_max=max(float(r["density_relative_max"]) for r in state),
                    mach_max=max(float(r["max_speed"]) / speed for r in state),
                    minimum_clearance=min(float(r["minimum_clearance"]) for r in state),
                    motion_error_max=max(float(r["motion_error"]) for r in state),
                    motion_velocity_error_max=max(float(r["velocity_error"]) for r in state))
                checks.append(row)
    with (RESULTS / "state_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=checks[0])
        writer.writeheader()
        writer.writerows(checks)
    differences = {}
    for speed in (40, 160):
        for re in (50, 100, 150):
            if speed == 40 or re == 100:
                archive = CASE / f"results/comparison/re{re}_c{speed}/native.csv"
            else:
                archive = CASE / f"results/c160_comparison/re{re}_standard160/native.csv"
            with archive.open(newline="") as stream:
                old = list(csv.DictReader(stream))
            fresh = history(re, speed, "legacy")
            assert len(old) == len(fresh["time"]) == 401
            differences[f"re{re}_c{speed}"] = {
                key: float(np.max(np.abs(fresh[key] - np.asarray([float(r[key]) for r in old]))))
                for key in ("time", "cd_pressure", "cd_viscous")}
    (RESULTS / "legacy_archive_comparison.json").write_text(json.dumps(differences, indent=2) + "\n")
    with (RESULTS / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for speed in (40, 160):
        for re in (50, 100, 150):
            pair = {r["criterion"]: r for r in rows if int(r["reynolds"]) == re and int(r["sound_speed"]) == speed}
            print(f"Re={re} c0={speed}: " + "; ".join(
                f"{component} {100*float(pair['legacy'][component+'_raw_nrmse']):.2f}% -> {100*float(pair['pairwise'][component+'_raw_nrmse']):.2f}%"
                for component in ("pressure", "viscous", "total")))

if __name__ == "__main__":
    for component in ("pressure", "viscous", "total"):
        force_plot(component)
    force_plot("pressure", zoom=True)
    errors()
    validation_checks()
