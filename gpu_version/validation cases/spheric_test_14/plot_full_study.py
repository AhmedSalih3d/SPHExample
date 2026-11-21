"""Raw 3D heave comparisons; no fitted shift, scale, filtering or datum."""
import csv
import math
import tomllib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import CASES, oscillation_period

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results/full_mdbc_study"

def main():
    records = []
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for ax, (name, filename, title) in zip(axes, CASES):
        reference = np.loadtxt(ROOT / "reference" / filename)
        ax.plot(*reference.T, color="#222222", ls="--", lw=1.5, label="Experiment")
        variants = [(RESULTS / "SimpleMDBC", "Full 3D mDBC, dp=25 mm", "#168b86"),
                    (RESULTS / "NoMDBC", "Full 3D DBC, dp=25 mm", "#d55e00"),
                    (ROOT / "results/mdbc_study/SimpleMDBC_LinearDensityDiffusion", "Cartesian 2D mDBC, dp=12.5 mm", "#999999")]
        for directory, label, color in variants:
            path = directory / f"{name}.csv"
            if not path.exists():
                continue
            data = np.genfromtxt(path, names=True, delimiter=",")
            t, y = data["time"], data["heave_mm"]
            assert len(t) == 901 and np.all(np.isfinite(y)) and abs(t[-1]-4.5) < 1e-10
            assert np.allclose(np.diff(t), 0.005, atol=1e-10, rtol=0)
            if directory.parent == RESULTS:
                with path.with_suffix(".toml").open("rb") as stream:
                    config = tomllib.load(stream)
                assert config["complete"] and config["frozen_particles"] == 0 and config["dimensions"] == 3
                with path.with_name(f"{name}_body.csv").open() as stream:
                    body = list(csv.DictReader(stream))
                assert len(body) == len(t)
                datum = float(body[0]["Center:2"])-y[0]/1000
                for time, heave, row in zip(t, y, body):
                    assert all(math.isfinite(float(value)) for value in row.values())
                    assert abs(float(row["Time"])-time) < 1e-10
                    assert abs(1000*(float(row["Center:2"])-datum)-heave) < 1e-8
                    assert abs(sum(float(row[f"Orientation:{i}"])**2 for i in range(4))-1) < 1e-5
            target = reference[(reference[:, 0] >= t[0]) & (reference[:, 0] <= t[-1])]
            sign = -np.sign(y[0])
            def peak_time(a, b, times, values):
                mask = (times >= a) & (times <= b)
                return float(times[mask][np.argmax(sign*values[mask])])
            records.append(dict(release=name, variant=label,
                rmse_mm=float(np.sqrt(np.mean((np.interp(target[:, 0], t, y)-target[:, 1])**2))),
                zero_crossing_period_s=float(oscillation_period(t, y)),
                early_peak_period_s=peak_time(1., 1.8, t, y)-peak_time(.2, .8, t, y),
                reference_early_peak_period_s=peak_time(1., 1.8, *reference.T)-peak_time(.2, .8, *reference.T),
                late_mean_mm=float(np.mean(y[t >= 3.5])),
                max_horizontal_displacement_mm=float(np.max(np.hypot(data["surge_mm"], data["sway_mm"])))))
            ax.plot(t, y, color=color, lw=1.4, label=label)
        ax.axhline(0, color="#bbbbbb", lw=.7)
        ax.set(title=title, ylabel="Heave (mm)", xlim=(0, 4.5))
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("Time since release (s)")
    fig.suptitle("Test 14: complete 3D floats, experimental masses and free rigid motion\nLaminar viscosity, linear density diffusion, h/dp=2; raw experimental datum and release time")
    fig.savefig(RESULTS / "heave_comparison.png", dpi=170)
    plt.close(fig)
    with (RESULTS / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(record)

if __name__ == "__main__":
    main()
