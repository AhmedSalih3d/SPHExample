"""Verify and compare the matched three full n80 diffusion controls."""
import csv
import runpy
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "fluid_only_ddt"
analysis = runpy.run_path(str(ROOT / "plot_coefh_refinement.py"))


def main():
    reference = analysis["read_reference"](ROOT / "reference" / "Force_Re100.dat")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    records = []
    for label, directory, color in (
        ("Original DDT", "coefh_refinement/re100", "#0072b2"),
        ("No DDT", "zero_ddt_full", "#999999"),
        ("Fluid-fluid DDT only", "fluid_only_ddt", "#d55e00"),
    ):
        stem = ROOT / "results" / directory / "Re100_n080"
        d = np.genfromtxt(stem.with_suffix(".csv"), delimiter=",", names=True)
        s = np.genfromtxt(Path(str(stem) + "_state.csv"), delimiter=",", names=True)
        m = tomllib.loads(stem.with_suffix(".toml").read_text())
        assert len(d) == len(s) == 401 and d["time"][-1] == 8
        assert np.array_equal(d["time"], s["time"])
        assert all(np.isfinite(d[key]).all() for key in d.dtype.names)
        assert all(np.isfinite(s[key]).all() for key in s.dtype.names)
        assert np.all(s["fluid_particles"] == 313600) and np.all(s["square_particles"] == 6400)
        assert m["sound_speed"] == 160 and m["dx"] == .0125 and m["gravity"] == 0
        assert m["h"] == float(np.float32(1.2*np.sqrt(2)*.0125))
        assert m["cfl"] == float(np.float32(.2)) and m["motion_kind"] == "integrated"
        assert m["background_pressure"] == 0 and not m["edge_sponge"] and not m["divergence_cleaning"]
        row = dict(model=label)
        for component, col, ax in (("pressure", 1, axes[0, 0]), ("viscous", 2, axes[0, 1])):
            values = d[f"cd_{component}"]
            ax.plot(d["time"], values, color=color, lw=1, label=label)
            row[component+"_error"] = analysis["relative_error"](d["time"], values, reference[:, 0], reference[:, col])
        selected = (d["time"] >= .5) & (d["time"] <= 1)
        minimum = np.argmin(d["cd_pressure"][selected])
        row.update(minimum_startup_cd=float(d["cd_pressure"][selected][minimum]),
                   minimum_startup_time=float(d["time"][selected][minimum]),
                   final_mean_pressure=float(s["pressure_mean"][-1]),
                   penetration_mm=1000*max(0, -s["minimum_clearance"].min()))
        records.append(row)
        axes[1, 0].plot(d["time"][selected], d["cd_pressure"][selected], color=color, lw=1, label=label)
        axes[1, 1].plot(s["time"], s["pressure_mean"]/1000, color=color, lw=1, label=label)
    for ax, col in ((axes[0, 0], 1), (axes[0, 1], 2), (axes[1, 0], 1)):
        ax.plot(reference[:, 0], reference[:, col], color="black", ls="--", lw=1, label="Official FD")
    for ax, title, ylabel, limits in (
        (axes[0, 0], "Raw pressure drag", "Cd,p", (0, 8)),
        (axes[0, 1], "Raw viscous drag", "Cd,v", (0, 8)),
        (axes[1, 0], "Startup pressure drag", "Cd,p", (.5, 1)),
        (axes[1, 1], "Dynamically generated mean fluid pressure", "Gauge pressure [kPa]", (0, 8)),
    ):
        ax.set(title=title, ylabel=ylabel, xlabel="Time [s]", xlim=limits)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("Re=100, L/dp=80: diffusion pair-selection control\n"
                 "c0=160 m/s, coefh=1.2, integrated motion; no force filtering")
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=180)
    fig.savefig(OUT / "comparison.pdf")
    with (OUT / "comparison_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(records)


if __name__ == "__main__":
    main()
