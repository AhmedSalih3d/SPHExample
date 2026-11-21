"""Plot the 200 Hz rigid-body fluid forces for Test 14."""
import argparse
from pathlib import Path
import tomllib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import CASES

ROOT = Path(__file__).resolve().parent


def main(tag="benchmark_3d_48m", release=None):
    folder = ROOT / "results" / tag
    for key, _, label in CASES:
        if release is not None and key != release:
            continue
        settings = tomllib.loads((folder / f"{key}.toml").read_text())
        data = np.genfromtxt(folder / f"{key}_body.csv", names=True, delimiter=",")
        time = data["Time"]
        interval = settings["floating_output_interval_s"]
        if not np.allclose(np.diff(time), interval, atol=1e-8, rtol=0):
            raise ValueError(f"Unexpected force history spacing for {key}")
        held = settings.get("pause_time", 0) >= settings["duration"]
        elapsed = time - settings.get("release_time", 0)
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
            if not held:
                panel.axvline(0, color="0.4", linestyle=":")
            panel.grid(alpha=0.25)
            panel.legend()
            panel.set_ylabel("Force (N)")
        horizontal.set_xlabel("Time under hold (s)" if held else "Time since release (s)")
        case_label = label if not held else f"{label.split(' · ')[0]} held at {settings['draft']*1000:g} mm draft"
        ax.set_title(f"{case_label} | {2*settings['half_width']:g} m tank | {1/interval:g} Hz body force")
        fig.savefig(folder / f"{key}_forces.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", nargs="?", default="benchmark_3d_48m")
    parser.add_argument("--release", choices=[key for key, _, _ in CASES])
    args = parser.parse_args()
    main(args.tag, args.release)
