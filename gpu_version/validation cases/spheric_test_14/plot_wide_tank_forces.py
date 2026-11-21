"""Plot the 200 Hz rigid-body fluid forces for the wide-tank releases."""
from pathlib import Path
import tomllib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import CASES

ROOT = Path(__file__).resolve().parent


def main():
    folder = ROOT / "results" / "benchmark_3d_48m"
    for key, _, label in CASES:
        settings = tomllib.loads((folder / f"{key}.toml").read_text())
        data = np.genfromtxt(folder / f"{key}_body.csv", names=True, delimiter=",")
        time = data["Time"]
        interval = settings["floating_output_interval_s"]
        if not np.allclose(np.diff(time), interval, atol=1e-8, rtol=0):
            raise ValueError(f"Unexpected force history spacing for {key}")
        elapsed = time - settings["release_time"]
        valid = time > 0
        weight = settings["mass"] * 9.81
        vertical = data["Force2"]
        fig, (ax, horizontal) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                             layout="constrained")
        ax.plot(elapsed[valid], vertical[valid], label="Vertical fluid force")
        ax.axhline(weight, color="black", linestyle="--", label=f"Weight ({weight:.2f} N)")
        horizontal.plot(elapsed[valid], data["Force0"][valid], label="Surge")
        horizontal.plot(elapsed[valid], data["Force1"][valid], label="Sway")
        for panel in (ax, horizontal):
            panel.axvline(0, color="0.4", linestyle=":")
            panel.grid(alpha=0.25)
            panel.legend()
            panel.set_ylabel("Force (N)")
        horizontal.set_xlabel("Time since release (s)")
        ax.set_title(f"{label} | 4.8 m tank | 200 Hz body force")
        fig.savefig(folder / f"{key}_forces.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
