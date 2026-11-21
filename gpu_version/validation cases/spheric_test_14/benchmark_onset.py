"""Compare the three 3D releases in raw time and with an early-motion delay."""
import csv
from pathlib import Path
import tomllib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot import CASES, released_history

ROOT = Path(__file__).resolve().parent


def onset(time, heave, direction, threshold):
    motion = direction * (heave - heave[0])
    i = np.flatnonzero(motion >= threshold)[0]
    return np.interp(threshold, motion[i-1:i+1], time[i-1:i+1])


def main():
    folder = ROOT / "results" / "benchmark_3d_25mm"
    rows = []
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, layout="constrained")
    for ax, (key, filename, label) in zip(axes, CASES):
        t_exp, y_exp = np.loadtxt(ROOT / "reference" / filename).T
        t_sim, history = released_history(folder / f"{key}.csv")
        if not (np.isclose(t_sim[0], t_exp[0]) and np.isclose(t_sim[-1], t_exp[-1])):
            raise ValueError(f"Incomplete benchmark history for {key}")
        y_sim = history["heave_mm"]
        direction = -np.sign(y_sim[0])
        delays = [onset(t_exp, y_exp, direction, mm) - onset(t_sim, y_sim, direction, mm)
                  for mm in (1, 5, 10)]
        delay = np.median(delays)
        t_aligned = t_sim + delay
        common = (t_exp >= max(t_sim[0], t_aligned[0])) & (t_exp <= min(t_sim[-1], t_aligned[-1]))
        raw_error = np.interp(t_exp, t_sim, y_sim) - y_exp
        aligned_error = np.interp(t_exp[common], t_aligned, y_sim) - y_exp[common]
        rows.append(dict(case=key, onset_delay_ms=1000*delay, physical_hold_rmse_mm="",
                         delay_1mm_ms=1000*delays[0], delay_5mm_ms=1000*delays[1],
                         delay_10mm_ms=1000*delays[2], raw_rmse_mm=np.sqrt(np.mean(raw_error**2)),
                         common_raw_rmse_mm=np.sqrt(np.mean(raw_error[common]**2)),
                         aligned_rmse_mm=np.sqrt(np.mean(aligned_error**2)), aligned_samples=common.sum()))
        ax.plot(t_exp, y_exp, color="black", label="Experiment")
        ax.plot(t_sim, y_sim, label="3D SPH, raw")
        ax.plot(t_aligned, y_sim, "--", label=f"3D SPH, shifted {1000*delay:+.0f} ms")
        held_path = folder.with_name(folder.name + "_hold") / f"{key}.csv"
        held_config = held_path.with_suffix(".toml")
        if held_config.exists():
            with held_config.open("rb") as stream:
                pause_ms = 1000 * tomllib.load(stream)["pause_time"]
            t_held, held = released_history(held_path)
            y_held = held["heave_mm"]
            control_error = np.interp(t_exp, t_held, y_held) - y_exp
            rows[-1]["physical_hold_rmse_mm"] = np.sqrt(np.mean(control_error**2))
            ax.plot(t_held, y_held, ":", label=f"3D SPH, {pause_ms:g} ms physical hold")
        ax.set(title=label, ylabel="Heave (mm)")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[-1].set(xlabel="Benchmark time (s)", xlim=(0, 4.5))
    fig.savefig(folder / "onset_comparison.png", dpi=160)
    plt.close(fig)
    with (folder / "onset_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
