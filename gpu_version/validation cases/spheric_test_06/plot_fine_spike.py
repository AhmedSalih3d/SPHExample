"""Plot the force and rear-wall support around the fine-resolution spike."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import argparse

OUT = Path(__file__).resolve().parent / "results" / "fine_spike_replay"


def main():
    fine = np.genfromtxt(OUT / "fine_faces.csv", delimiter=",", names=True)
    archive = np.genfromtxt(OUT / "archived_faces.csv", delimiter=",", names=True)
    t = fine["time"]
    assert len(fine) == 101 and (np.diff(t) > 0).all()
    np.testing.assert_allclose(t, np.linspace(.76, .86, 101), atol=12e-6, rtol=0)
    np.testing.assert_allclose(fine["pressure_cd"], fine["front_cd"] + fine["rear_cd"] + fine["side_cd"], atol=1e-10)
    fig, axes = plt.subplots(3, 1, sharex=True, layout="constrained", figsize=(10, 8))
    for key, label in (("pressure_cd", "Net"), ("front_cd", "Front"), ("rear_cd", "Rear"), ("side_cd", "Top + bottom")):
        axes[0].plot(t, fine[key], label=label)
    axes[0].plot(archive["time"], archive["pressure_cd"], "ko", ms=4, label="Original 20 ms samples")
    axes[0].axhline(0, color="0.55", lw=.8)
    axes[0].set_ylabel("Pressure drag Cd")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(t, fine["rear_median_support"], color="#0072b2")
    support = axes[1].twinx()
    support.step(t, fine["rear_supported"], color="#d55e00", where="mid", alpha=.8)
    support.set_ylabel("Supported probes [of 40]", color="#d55e00")
    axes[1].set_ylabel("Median rear kernel support", color="#0072b2")
    axes[2].plot(t, fine["peak_rear_fluid_pressure_kpa"], color="#0072b2")
    approach = axes[2].twinx()
    approach.plot(t, fine["max_rear_approach_mps"], color="#d55e00")
    approach.set_ylabel("Peak approach speed [m/s]", color="#d55e00")
    axes[2].set(ylabel="Peak rear fluid pressure [kPa]", xlabel="Simulated time [s]")
    for ax in axes:
        ax.axvline(.82, color=".5", ls=":", lw=.9)
        ax.grid(alpha=.22)
    fig.suptitle("Moving-square pressure reversal · Re = 100 · L/dp = 160 · CFL = 0.10")
    fig.savefig(OUT / "spike_face_loads.png", dpi=180)
    plt.close(fig)


def compare_shifting():
    fig, axes = plt.subplots(3, 1, sharex=True, layout="constrained", figsize=(9, 7))
    fields = ("rear_supported", "pressure_cd", "peak_rear_fluid_pressure_kpa")
    for label, filename in (("Planar", "fine_faces.csv"), ("Off", "no_shifting_history.csv"),
                            ("DualSPHysics classic full", "dual_shifting_history.csv")):
        rows = np.genfromtxt(OUT / filename, delimiter=",", names=True)
        rows = rows[(rows["time"] >= .759999) & (rows["time"] <= .86002)]
        np.testing.assert_allclose(rows["time"], np.linspace(.76, .86, 101), atol=12e-6, rtol=0)
        np.testing.assert_allclose(rows["pressure_cd"], rows["front_cd"] + rows["rear_cd"] + rows["side_cd"], atol=1e-10)
        for ax, field in zip(axes, fields):
            ax.plot(rows["time"], rows[field], label=label)
    for ax, ylabel in zip(axes, ("Rear probes supported (of 40)", "Pressure drag Cd", "Peak rear pressure [kPa]")):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.22)
    axes[0].legend()
    axes[1].axhline(0, color="0.5", lw=.8)
    axes[2].set_xlabel("Simulated time [s]")
    fig.suptitle("Rear contact and pressure during square acceleration · Re = 100 · L/dp = 160")
    fig.savefig(OUT / "shifting_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-shifting", action="store_true")
    if parser.parse_args().compare_shifting:
        compare_shifting()
    else:
        main()
