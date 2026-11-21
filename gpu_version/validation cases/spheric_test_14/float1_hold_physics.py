"""Compare Float 1 physics on a clock that starts at the simulated release."""
import csv
from pathlib import Path
import tomllib

import numpy as np
from scipy.signal import find_peaks
from benchmark_onset import onset
from plot import released_history

ROOT = Path(__file__).resolve().parent


def main():
    results = ROOT / "results"
    study = results / "benchmark_3d_25mm_hold_725ms"
    key = "float1_down"
    reference = np.loadtxt(ROOT / "reference" / "Fl1-0p074.txt")
    rows = []
    traces = [("Experiment", reference[:, 0], reference[:, 1])]
    for label, folder in (
        ("Immediate release", results / "benchmark_3d_25mm"),
        ("60 ms hold", results / "benchmark_3d_25mm_hold"),
        ("725 ms hold", study),
    ):
        t, data = released_history(folder / f"{key}.csv")
        with (folder / f"{key}.toml").open("rb") as stream:
            settings = tomllib.load(stream)
        pause = settings["pause_time"]
        if pause and "release_time" not in settings:
            post_release = t >= pause - 1e-10
            t, data = np.maximum(0, t[post_release] - pause), data[post_release]
        traces.append((label, t, data["heave_mm"]))
        body = np.genfromtxt(folder / f"{key}_body.csv", names=True, delimiter=",")
        pre_release = (body["Time"] > 0) & (body["Time"] <= pause + 1e-10)
        if np.any(pre_release):
            print(label, "force at release (N)", body["Force2"][pre_release][-1],
                  "held center movement (mm)", 1000 * np.max(np.abs(
                      body["Center2"][pre_release] - body["Center2"][0])))
    common = reference[:, 0] <= min(t[-1] for _, t, _ in traces) + 1e-10
    for label, t, y in traces:
        peaks = find_peaks(y, prominence=5, distance=100)[0]
        troughs = find_peaks(-y, prominence=5, distance=100)[0]
        aligned = np.interp(reference[common, 0], t, y)
        delay = np.median([onset(*reference.T, 1, mm) - onset(t, y, 1, mm) for mm in (1, 5, 10)])
        overlap = common & (reference[:, 0] >= max(0, delay)) & (reference[:, 0] <= t[-1] + delay)
        shifted = np.interp(reference[overlap, 0], t + delay, y)
        rows.append(dict(trace=label, release_time_rmse_mm=np.sqrt(np.mean(
            (aligned - reference[common, 1])**2)), onset_delay_ms=1000*delay,
            onset_aligned_rmse_mm=np.sqrt(np.mean((shifted - reference[overlap, 1])**2)),
            first_crest_s=t[peaks[0]],
            first_crest_mm=y[peaks[0]], first_trough_mm=y[troughs[0]],
            first_crest_period_s=t[peaks[1]] - t[peaks[0]],
            fourth_swing_mm=y[peaks[3]] - y[troughs[3]]))
    with (study / "release_physics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
