"""Compare Float 1 peak timing and, when available, held restoring loads."""
from pathlib import Path
import csv
import tomllib

import numpy as np
from plot import oscillation_period
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/float1_period"
RUNS = (
    ("2.4 m tank", "full_UpdatedMDBC_dp0.025_half1.2_h2.0_t4.5"),
    ("4.8 m tank", "full_UpdatedMDBC_dp0.025_half2.4_h2.0_t4.5"),
)


def peaks(time, heave):
    indices = np.flatnonzero((heave[1:-1] > heave[:-2]) & (heave[1:-1] >= heave[2:])) + 1
    indices = indices[(time[indices] > 0.25) & (heave[indices] > 10)]
    # Keep the largest crest when the experimental trace has several adjacent maxima.
    kept = []
    for i in sorted(indices, key=lambda i: heave[i], reverse=True):
        if all(abs(time[i] - time[j]) >= 0.5 for j in kept):
            kept.append(i)
    kept.sort()
    return time[kept], heave[kept]


def first_rise(time, heave, rise_mm):
    threshold = heave[0] + rise_mm
    i = np.flatnonzero(heave >= threshold)[0]
    return np.interp(threshold, heave[i-1:i+1], time[i-1:i+1])


def timing():
    reference = np.loadtxt(ROOT / "reference/Fl1-0p074.txt")
    reference = reference[(reference[:, 0] >= 0) & (reference[:, 0] <= 4.5)]
    curves = [("Experiment", *reference.T)]
    configs = []
    for label, folder in RUNS:
        path = ROOT / "results" / folder
        configs.append(tomllib.loads((path / "float1_down.toml").read_text()))
        history = np.genfromtxt(path / "float1_down.csv", delimiter=",", names=True)
        curves.append((label, history["time"], history["heave_mm"]))
    measured = {"half_width", "steps", "wall_seconds", "fluid_particles", "boundary_particles"}
    assert {k: v for k, v in configs[0].items() if k not in measured} == {
        k: v for k, v in configs[1].items() if k not in measured}
    assert all(c["complete"] and not c["frozen_particles"] for c in configs)
    fig, (heave_ax, lag_ax) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    reference_peaks, _ = peaks(*reference.T)
    reference_onset = first_rise(*reference.T, 1)
    records = []
    for (label, time, heave), color in zip(curves, ("black", "#0072b2", "#d55e00")):
        peak_times, height = peaks(time, heave)
        heave_ax.plot(time, heave, color=color, ls="--" if label == "Experiment" else "-", label=label)
        heave_ax.scatter(peak_times, height, color=color, s=13)
        intervals = np.diff(peak_times)
        if label != "Experiment":
            n = min(len(peak_times), len(reference_peaks))
            lag_ax.plot(reference_peaks[:n], 1000 * (peak_times[:n] - reference_peaks[:n]),
                        "o-", color=color, label=label)
        onset = first_rise(time, heave, 1)
        lag = reference_onset - onset
        comparison = reference[(reference[:, 0] >= max(0, lag)) &
                               (reference[:, 0] <= min(time[-1], time[-1] + lag))]
        errors = [np.sqrt(np.mean((np.interp(comparison[:, 0] - shift, time, heave) -
                                  comparison[:, 1])**2)) for shift in (0, lag)]
        records.append(dict(case=label, first_rise_1mm_s=onset, first_rise_5mm_s=first_rise(time, heave, 5),
                            first_rise_10mm_s=first_rise(time, heave, 10),
                            first_rise_lag_s=lag, raw_rmse_overlap_mm=errors[0],
                            aligned_rmse_overlap_mm=errors[1],
                            first_peak_s=peak_times[0], first_peak_mm=height[0],
                            first_peak_interval_s=intervals[0], mean_peak_interval_s=np.mean(intervals),
                            rising_zero_crossing_period_s=oscillation_period(time, heave),
                            fifth_peak_s=peak_times[4]))
    heave_ax.set(ylabel="Heave (mm)", title="Float 1 downward release")
    lag_ax.set(xlabel="Time since release (s)", ylabel="Peak timing error (ms)", xlim=(0, 4.5))
    lag_ax.axhline(0, color="black", lw=.7)
    for ax in (heave_ax, lag_ax):
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.savefig(OUTPUT / "peak_timing.png", dpi=160)
    plt.close(fig)
    return records


def restoring():
    radius, draft, mass, gravity = .15, .187, 9.75, 9.81
    area = np.pi * radius**2
    records = []
    for offset in (-25, 0, 25):
        path = ROOT / "results/float1_restoring_half1.2" / f"offset_{offset}mm.csv"
        if not path.exists():
            return
        data = np.genfromtxt(path, delimiter=",", names=True)
        time, force = data["time"], data["fluid_force_n"]
        static = 1000 * gravity * area * (2 * radius / 3 + draft - offset / 1000 - radius)
        records.append(dict(offset_mm=offset, mean_force_n=np.mean(force[time >= 1]),
                            early_mean_n=np.mean(force[(time >= 1) & (time < 1.5)]),
                            late_mean_n=np.mean(force[time >= 1.5]), hydrostatic_n=static))
    write_rows(OUTPUT / "restoring.csv", records)
    offsets = np.array([r["offset_mm"] for r in records])
    loads = np.array([r["mean_force_n"] for r in records])
    stiffness = -(loads[-1] - loads[0]) / .05
    print(f"Measured stiffness {stiffness:.2f} N/m. Nominal stiffness {1000*gravity*area:.2f} N/m.")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(offsets, loads, "o-", label="UpdatedMDBC, 1–2 s mean")
    ax.plot(offsets, [r["hydrostatic_n"] for r in records], "s--", label="Displaced volume")
    ax.axhline(mass * gravity, color="black", lw=.8, label="Float weight")
    ax.set(xlabel="Heave relative to nominal draft (mm)", ylabel="Vertical force (N)")
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT / "restoring.png", dpi=160)
    plt.close(fig)


def write_rows(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_rows(OUTPUT / "timing.csv", timing())
    restoring()
