"""Matched floating UpdatedMDBC controls; raw time and experimental datum."""
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
RESULTS = ROOT / "results/floating_updated_study"

def main():
    records = []
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    variants = [(ROOT / "results/full_mdbc_study/SimpleMDBC", "SimpleMDBC", "#999999"),
        (RESULTS / "UpdatedMDBC", "UpdatedMDBC: no slip", "#168b86"),
        (RESULTS / "UpdatedMDBCWithImpedance", "Experimental normal pressure impedance", "#d55e00"),
        (RESULTS / "UpdatedMDBC_CFL0.1", "UpdatedMDBC: CFL 0.1", "#6d4ca4")]
    for ax, (name, filename, title) in zip(axes, CASES):
        reference = np.loadtxt(ROOT / "reference" / filename)
        ax.plot(*reference.T, color="#222222", ls="--", label="Experiment", lw=1.4)
        for directory, label, color in variants:
            path = directory / f"{name}.csv"
            if not path.exists():
                continue
            data = np.genfromtxt(path, names=True, delimiter=",")
            t, y = data["time"], data["heave_mm"]
            with path.with_suffix(".toml").open("rb") as stream:
                config = tomllib.load(stream)
            assert config["complete"] and config["frozen_particles"] == 0 and config["dimensions"] == 3
            assert len(t) == 901 and np.all(np.isfinite(y)) and abs(t[-1]-4.5) < 1e-10
            assert np.allclose(np.diff(t), .005, atol=1e-10, rtol=0)
            assert config["dx"] == .025 and config["mass"] == (23.42 if name == "float2_down" else 9.75)
            target = reference[(reference[:, 0] >= t[0]) & (reference[:, 0] <= t[-1])]
            records.append(dict(release=name, variant=label,
                rmse_mm=float(np.sqrt(np.mean((np.interp(target[:, 0],t,y)-target[:, 1])**2))),
                zero_crossing_period_s=float(oscillation_period(t,y)),
                late_mean_mm=float(np.mean(y[t>=3.5])),
                max_horizontal_mm=float(np.max(np.hypot(data["surge_mm"],data["sway_mm"])))))
            if directory.parent == RESULTS:
                with path.with_name(f"{name}_body.csv").open() as stream:
                    body = list(csv.DictReader(stream))
                assert len(body) == len(t)
                for row in body:
                    assert all(math.isfinite(float(value)) for value in row.values())
                    assert abs(sum(float(row[f"Orientation:{i}"])**2 for i in range(4))-1) < 1e-5
            ax.plot(t,y,color=color,lw=1.3,label=label,ls=":" if "CFL0.1" in directory.name else "-")
        ax.axhline(0,color="0.7",lw=.7)
        ax.set(title=title,ylabel="Heave (mm)",xlim=(0,4.5))
        ax.grid(alpha=.2)
        ax.legend(frameon=False,fontsize=8)
    axes[-1].set_xlabel("Time since release (s)")
    fig.suptitle("Full 3D floating-boundary comparison: dp=25 mm, c0=25 m/s, h/dp=2\nExperimental mass, free rigid motion, physical viscosity and identical tank/sponge; no fitted shift")
    fig.savefig(RESULTS / "heave_comparison.png",dpi=170)
    plt.close(fig)
    with (RESULTS / "summary.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    for record in records:
        print(record)

if __name__ == "__main__":
    main()
