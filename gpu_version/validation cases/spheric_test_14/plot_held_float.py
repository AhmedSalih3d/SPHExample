"""Compare Float 1's held hydrodynamic load with its experimental weight."""
import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main(duration):
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    summaries = []
    start = duration * 2 / 3
    for spacing, color in ((0.025, "#d55e00"), (0.0125, "#0072b2")):
        folder = ROOT / "results" / f"held_float_dp{spacing}_t{duration}"
        data = np.genfromtxt(folder / "vertical_force.csv", delimiter=",", names=True)
        time, force = data["time"], data["fluid_force_n"]
        weight = data["weight_n"][0]
        # Frame zero is written before the first force evaluation.
        ax.plot(time[1:], force[1:], color=color, label=f"dp = {spacing * 1000:g} mm")
        interval = time >= start - 1e-9
        late = force[interval]
        summaries.append(dict(dp_mm=spacing * 1000, start_s=start, stop_s=time[-1],
                              mean_force_n=np.mean(late), std_force_n=np.std(late),
                              mean_excess_n=np.mean(late) - weight, weight_n=weight))
    ax.axhline(weight, color="0.2", ls="--", label=f"Float weight ({weight:.2f} N)")
    ax.axvspan(start, duration, color="0.5", alpha=0.10, label="Averaging interval")
    ax.set(xlabel="Time since initialization (s)", ylabel="Upward fluid force (N)",
           title="SPHERIC Test 14 | Float 1 held at the experimental draft", xlim=(0, duration))
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    out = ROOT / "results" / f"held_float_t{duration}"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "buoyancy_comparison.png", dpi=180)
    with (out / "force_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("duration", nargs="?", type=float, default=3.0)
    main(parser.parse_args().duration)
