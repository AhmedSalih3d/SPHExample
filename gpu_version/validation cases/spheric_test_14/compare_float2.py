"""Compare physical-time Float 2 releases against the 200 Hz SPHERIC mean."""
import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def compare(study):
    results = ROOT / "results"
    study = results / study
    reference = np.loadtxt(ROOT / "reference" / "Fl2-0p088.txt")
    baseline = results / "floating_updated_study" / "UpdatedMDBC"
    cases = [("Experiment", reference)]
    for label, folder in [("Nominal draft, 2.4 m tank", baseline),
                          ("Held-load draft, 2.4 m tank", study / "release"),
                          ("Held-load draft, 3.6 m tank", study / "wide_release"),
                          ("Hydrostatic draft, 3.6 m tank", study / "wide_hydrostatic")]:
        if folder.name.startswith("wide_") and not (folder / "float2_down.toml").exists():
            continue
        raw = np.genfromtxt(folder / "float2_down.csv", names=True, delimiter=",")
        cases.append((label, np.column_stack((raw["time"], raw["heave_mm"]))))
    fig, (heave, error) = plt.subplots(2, 1, sharex=True, figsize=(9, 6),
                                      height_ratios=(2, 1), layout="constrained")
    rows = []
    for label, data in cases:
        t, y = data.T
        if t[0] > reference[0, 0] + 1e-8 or t[-1] < reference[-1, 0] - 1e-8:
            raise ValueError(f"Incomplete release history for {label}")
        residual = np.interp(reference[:, 0], t, y) - reference[:, 1]
        peaks = find_peaks(y, distance=int(0.6 / np.median(np.diff(t))), prominence=5)[0]
        lows = find_peaks(-y, distance=int(0.6 / np.median(np.diff(t))), prominence=5)[0]
        rows.append(dict(trace=label, rmse_mm=np.sqrt(np.mean(residual**2)),
                         late_mean_mm=np.mean(y[t >= 3.5]), first_crest_mm=y[peaks[0]],
                         first_crest_s=t[peaks[0]], fourth_crest_mm=y[peaks[3]],
                         fourth_trough_mm=y[lows[3]], fourth_swing_mm=y[peaks[3]] - y[lows[3]]))
        line, = heave.plot(t, y, label=label, color="black" if label == "Experiment" else None)
        if label != "Experiment":
            error.plot(reference[:, 0], residual, color=line.get_color())
    heave.legend()
    heave.set(title="SPHERIC Test 14, Float 2 released from -88 mm", ylabel="Heave (mm)")
    error.set(xlabel="Time since release (s)", ylabel="SPH - experiment (mm)")
    for ax in (heave, error):
        ax.axhline(0, color="grey", lw=0.6)
        ax.grid(alpha=0.2)
    fig.savefig(study / "comparison.png", dpi=160)
    plt.close(fig)
    with (study / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", nargs="?", default="float2_equilibrium_half1.2")
    compare(parser.parse_args().study)
