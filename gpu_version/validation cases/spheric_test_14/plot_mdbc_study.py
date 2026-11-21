"""Compare matched boundary/density-diffusion heave controls without shifting time or datum."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import CASES, oscillation_period

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/mdbc_study"
DIFFUSIONS = ("LinearDensityDiffusion", "FourtakasDensityDiffusion")

def read_case(mode, diffusion, name):
    return np.genfromtxt(RESULTS / f"{mode}_{diffusion}" / f"{name}.csv", names=True, delimiter=",")

def analyze():
    records = []
    archive_checks = {}
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
    for row, (name, reference_file, title) in enumerate(CASES):
        reference = np.loadtxt(ROOT / "reference" / reference_file)
        for col, diffusion in enumerate(DIFFUSIONS):
            ax = axes[row, col]
            ax.plot(*reference.T, color="#222222", ls="--", lw=1.4, label="Axisymmetric experiment")
            for mode, color in (("NoMDBC", "#999999"), ("SimpleMDBC", "#168b86")):
                data = read_case(mode, diffusion, name)
                t, y = data["time"], data["heave_mm"]
                selected = (reference[:, 0] >= t[0]) & (reference[:, 0] <= t[-1])
                target = reference[selected]
                assert len(t) == 901 and abs(t[-1] - 4.5) < 1e-10 and np.all(np.isfinite(y))
                sign = -np.sign(y[0])
                excursion = lambda a, b, times, values: float(np.max(sign * values[(times >= a) & (times <= b)]))
                def peak_time(a, b, times, values):
                    mask = (times >= a) & (times <= b)
                    return float(times[mask][np.argmax(sign * values[mask])])
                first = excursion(0.2, 0.8, t, y)
                second = excursion(1.0, 1.8, t, y)
                ref_first = excursion(0.2, 0.8, reference[:, 0], reference[:, 1])
                ref_second = excursion(1.0, 1.8, reference[:, 0], reference[:, 1])
                records.append(dict(release=name, boundary=mode, diffusion=diffusion,
                    rmse_mm=float(np.sqrt(np.mean((np.interp(target[:, 0], t, y) - target[:, 1]) ** 2))),
                    period_s=float(oscillation_period(t, y)), reference_period_s=float(oscillation_period(*reference.T)),
                    late_mean_mm=float(np.mean(y[t >= 3.5])), first_opposite_excursion_mm=first,
                    early_peak_period_s=peak_time(1.0, 1.8, t, y)-peak_time(0.2, 0.8, t, y),
                    reference_early_peak_period_s=peak_time(1.0, 1.8, reference[:, 0], reference[:, 1])-peak_time(0.2, 0.8, reference[:, 0], reference[:, 1]),
                    second_opposite_excursion_mm=second, second_first_ratio=second / first,
                    reference_second_first_ratio=ref_second / ref_first))
                ax.plot(t, y, color=color, lw=1.25, label="DBC" if mode == "NoMDBC" else "Original mDBC")
                if mode == "NoMDBC" and diffusion == "LinearDensityDiffusion":
                    archive_path = ROOT / "results/planar_dp0.0125_half8.0_t4.5_200Hz______" / f"{name}.csv"
                    if archive_path.exists():
                        archive = np.genfromtxt(archive_path, names=True, delimiter=",")
                        archive_checks[name] = float(np.max(np.abs(y - np.interp(t, archive["time"], archive["heave_mm"]))))
            ax.axhline(0, color="#bbbbbb", lw=0.7)
            ax.set(title=f"{title} | {'Linear DDT' if col == 0 else 'Fourtakas inner DDT'}", ylabel="Heave (mm)", xlim=(0, 4.5))
            ax.grid(alpha=0.2)
            if row == 2:
                ax.set_xlabel("Time since release (s)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Test 14: matched Cartesian 2D DBC / mDBC controls\nSame mass per span, particles, Laminar viscosity, dp = 12.5 mm and h/dp = 2; no fitted shift or scale")
    fig.savefig(RESULTS / "heave_boundary_comparison.png", dpi=170)
    plt.close(fig)
    with (RESULTS / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    (RESULTS / "dbc_archive_check.json").write_text(json.dumps(archive_checks, indent=2) + "\n")
    for row in records:
        print(row)

def plot_held():
    directory = RESULTS / "held"
    with (directory / "summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for ax, name, title in zip(axes, ("float1_down", "float2_down"), ("Float 1", "Float 2")):
        for mode, color in (("NoMDBC", "#999999"), ("SimpleMDBC", "#168b86")):
            record = next(row for row in summary if row["release"] == name and row["boundary"] == mode)
            data = np.genfromtxt(directory / f"{name}_{mode}.csv", names=True, delimiter=",")
            error = 100 * (data["fluid_force_n_m"] / float(record["weight_n_m"]) - 1)
            label = "DBC" if mode == "NoMDBC" else "Original mDBC"
            ax.plot(data["time"][1:], error[1:], color=color, lw=0.9,
                    label=f"{label}: mean {100*float(record['relative_excess']):+.3f}% (2–3 s)")
        ax.axhline(0, color="black", ls="--", lw=0.8)
        ax.axvspan(2, 3, alpha=0.06, color="black")
        ax.set(title=title, ylabel="(Fluid load / weight − 1) × 100 (%)", xlim=(0, 3))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
    axes[-1].set_xlabel("Time since initialization (s)")
    fig.suptitle("Cartesian 2D buoyancy control: bodies held at their nominal draft\nLinear density diffusion, dp = 12.5 mm; frame zero precedes force evaluation and is omitted")
    fig.savefig(RESULTS / "held_buoyancy_comparison.png", dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    analyze()
    if (RESULTS / "held/summary.csv").exists():
        plot_held()
