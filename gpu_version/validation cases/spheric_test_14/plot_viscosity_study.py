"""Compare full 3D UpdatedMDBC viscosity runs with the Test 14 measurements."""
import csv
import tomllib
from pathlib import Path

import numpy as np

from plot import CASES, oscillation_period
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/updated_viscosity_study"
MODELS = (
    ("Laminar", ROOT / "results/floating_updated_study/UpdatedMDBC", "#555555"),
    ("ZeroViscosity", RESULTS / "ZeroViscosity", "#0072b2"),
    ("ArtificialViscosity", RESULTS / "ArtificialViscosity", "#d55e00"),
    ("LaminarSPS", RESULTS / "LaminarSPS", "#009e73"),
)


def load_history(folder, name, model):
    with (folder / f"{name}.toml").open("rb") as stream:
        config = tomllib.load(stream)
    if config["viscosity"] != model or not config["complete"] or config["frozen_particles"]:
        raise ValueError(f"Invalid {model} run for {name}")
    data = np.genfromtxt(folder / f"{name}.csv", names=True, delimiter=",")
    if not np.all(np.isfinite(data["heave_mm"])):
        raise ValueError(f"Nonfinite heave for {name}, {model}")
    np.testing.assert_allclose(data["time"], np.linspace(0, 4.5, 901), atol=1e-10, rtol=0)
    return config, data


def peaks(time, heave, sign):
    return [float(np.max(sign * heave[(time >= start) & (time <= stop)]))
            for start, stop in ((0.2, 0.8), (1.0, 1.8))]


def main():
    records = []
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, layout="constrained")
    for ax, (name, reference_file, title) in zip(axes, CASES):
        reference = np.loadtxt(ROOT / "reference" / reference_file)
        ax.plot(*reference.T, "--", color="black", lw=1.5, label="Experiment")
        sample = reference[(reference[:, 0] >= 0) & (reference[:, 0] <= 4.5)]
        sign = -np.sign(sample[0, 1])
        ref_first, ref_second = peaks(*sample.T, sign)
        baseline, baseline_history = load_history(MODELS[0][1], name, "Laminar")
        for model, folder, color in MODELS:
            config, data = load_history(folder, name, model)
            # Output wall time and adaptive step count are measured, not matching inputs.
            varying = {"viscosity", "steps", "wall_seconds"}
            if {k: v for k, v in config.items() if k not in varying} != {
                    k: v for k, v in baseline.items() if k not in varying}:
                raise ValueError(f"{name}, {model} does not match the Laminar input metadata")
            time, heave = data["time"], data["heave_mm"]
            first, second = peaks(time, heave, sign)
            error = np.interp(sample[:, 0], time, heave) - sample[:, 1]
            records.append(dict(
                release=name, viscosity=model, rmse_mm=float(np.sqrt(np.mean(error**2))),
                rms_change_from_laminar_mm=float(np.sqrt(np.mean((heave - baseline_history["heave_mm"])**2))),
                period_s=oscillation_period(time, heave), reference_period_s=oscillation_period(*sample.T),
                first_excursion_mm=first, reference_first_excursion_mm=ref_first,
                second_first_ratio=second / first, reference_second_first_ratio=ref_second / ref_first,
                late_mean_mm=float(np.mean(heave[time >= 3.5])),
                max_horizontal_mm=float(np.max(np.hypot(data["surge_mm"], data["sway_mm"]))),
                steps=config["steps"],
            ))
            ax.plot(time, heave, color=color, label=model, lw=1.3, ls=":" if model == "LaminarSPS" else "-")
        ax.axhline(0, color="0.75", lw=0.7)
        ax.set(title=title, ylabel="Heave (mm)", xlim=(0, 4.5))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
    axes[-1].set_xlabel("Time since release (s)")
    fig.suptitle("Test 14 · UpdatedMDBC viscosity comparison\nFull 3D, dp = 25 mm, c₀ = 25 m/s, CFL = 0.2")
    fig.savefig(RESULTS / "comparison.png", dpi=160)
    plt.close(fig)
    with (RESULTS / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
