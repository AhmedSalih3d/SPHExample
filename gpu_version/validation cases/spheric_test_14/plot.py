"""Plot unshifted Test 14 heave histories alongside the experimental means."""
import argparse
import csv
from pathlib import Path
import tomllib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CASES = (
    ("float1_down", "Fl1-0p074.txt", "Float 1 · −74 mm"),
    ("float1_up", "Fl1+0p76.txt", "Float 1 · +76 mm"),
    ("float2_down", "Fl2-0p088.txt", "Float 2 · −88 mm"),
)


def oscillation_period(t, y):
    crossings = np.flatnonzero((y[:-1] <= 0) & (y[1:] > 0))
    if len(crossings) < 2:
        return np.nan
    times = t[crossings] - y[crossings] * np.diff(t)[crossings] / np.diff(y)[crossings]
    return np.mean(np.diff(times))


def configuration_label(folder, key):
    with (folder / f"{key}.toml").open("rb") as stream:
        config = tomllib.load(stream)
    geometry = "Cartesian 2D, " if config.get("dimensions") == 2 else ""
    spacing = f"{1000*config['dx']:g}"
    if "fine_dx" in config:
        spacing += f" / {1000*config['fine_dx']:g}"
    label = f"{geometry}dp = {spacing} mm, tank width {2*config['half_width']:g} m"
    if "sponge_width" in config:
        label += f", sponge {config['sponge_width']:g} m"
    return label


def released_history(path):
    with path.with_suffix(".toml").open("rb") as stream:
        release_time = tomllib.load(stream).get("release_time", 0)
    data = np.genfromtxt(path, names=True, delimiter=",")
    data = data[data["time"] >= release_time - 1e-10]
    if len(data) < 2:
        raise ValueError(f"Expected at least two post-release samples in {path}")
    return np.maximum(0, data["time"] - release_time), data


def main(tag, comparison, release=None):
    folder = ROOT / "results" / Path(tag)
    cases = [case for case in CASES if release is None or case[0] == release]
    height = 4.5 if release else 8
    fig, axes = plt.subplots(len(cases), 1, figsize=(9, height), sharex=True, layout="constrained", squeeze=False)
    axes = axes[:, 0]
    metrics = []
    for ax, (key, filename, label) in zip(axes, cases):
        reference = np.loadtxt(ROOT / "reference" / filename)
        time, actual = released_history(folder / f"{key}.csv")
        heave = actual["heave_mm"]
        sample = reference[(reference[:, 0] >= time[0]) & (reference[:, 0] <= time[-1])]
        model = np.interp(sample[:, 0], time, heave)
        reference_period = oscillation_period(*sample.T)
        tail = time >= time[-1] - 1
        metrics.append(dict(case=key, rmse_mm=np.sqrt(np.mean((model - sample[:, 1])**2)),
                            reference_period_s=reference_period,
                            simulated_period_s=oscillation_period(time, heave),
                            last_second_mean_heave_mm=np.mean(heave[tail]),
                            maximum_horizontal_displacement_mm=np.max(np.hypot(actual["surge_mm"], actual["sway_mm"]))))
        ax.plot(*reference.T, color="0.2", ls="--", label="SPHERIC experiment")
        ax.plot(time, heave, color="#0072b2", label=configuration_label(folder, key))
        for index, other in enumerate(comparison):
            path = ROOT / "results" / other / f"{key}.csv"
            if path.exists():
                comparison_time, data = released_history(path)
                ax.plot(comparison_time, data["heave_mm"], color=("#d55e00", "#009e73")[index % 2],
                        alpha=.85, label=configuration_label(path.parent, key))
        ax.axhline(0, color="0.65", lw=.75)
        ax.set(title=label, ylabel="Heave (mm)")
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[-1].set(xlabel="Time since release (s)", xlim=(0, time[-1]))
    fig.suptitle("SPHERIC Test 14 · Free heave decay")
    prefix = f"{release}_" if release else ""
    plot_name = "heave_sensitivity.png" if comparison else "heave_comparison.png"
    fig.savefig(folder / (prefix + plot_name), dpi=180)
    plt.close(fig)
    with (folder / (prefix + "metrics.csv")).open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metrics[0])
        writer.writeheader()
        writer.writerows(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", nargs="?", default="full_UpdatedMDBC_dp0.025_half1.2_h2.0_t4.5")
    parser.add_argument("--compare", action="append", default=[])
    parser.add_argument("--release", choices=[case[0] for case in CASES])
    args = parser.parse_args()
    main(args.tag, args.compare, args.release)
