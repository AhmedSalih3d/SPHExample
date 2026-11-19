"""Compare the zero-diffusion shifted DBC run with the original DBC control."""
import csv
import argparse

from plot_regularized_mdbc_fields import CASE, RUNS, read_frame, plot_fields
from plot_coefh_refinement import read_table, read_reference, integral
import matplotlib.pyplot as plt
import numpy as np


MODELS = (("dbc_n80_half-step", "Fluid-only diffusion · NoShifting"),
          ("dbc_n80_half-step-planar-zero", "Zero diffusion · PlanarShifting"))
OUT = CASE / "results" / "regularized_mdbc" / "dbc_planar_zero"


def main(models=MODELS, out=OUT, *, compare_diffusion=False):
    out.mkdir(parents=True, exist_ok=True)
    ref = read_reference(CASE / "reference" / "Force_Re100.dat")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained", sharex=True)
    rows = []
    for tag, label in models:
        force = read_table(RUNS / tag / "native.csv")
        state = read_table(RUNS / tag / "native_state.csv")
        t = force["time"]
        np.testing.assert_allclose(t, np.linspace(0, 0.9, 46), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(t, state["time"])
        p, v = force["cd_pressure"], force["cd_viscous"]
        clearance = state["minimum_clearance"] * 1000
        drift = (state["mean_fluid_pressure_pa"] - state["mean_fluid_pressure_pa"][0]) / 1000
        i = np.flatnonzero(t >= 0.7)[np.argmin(p[t >= 0.7])]
        knots = np.unique(np.r_[0.1, 0.9, t[(t > 0.1) & (t < 0.9)], ref[(ref[:, 0] > 0.1) & (ref[:, 0] < 0.9), 0]])
        errors = []
        for values, col in ((p, 1), (v, 2)):
            target = np.interp(knots, ref[:, 0], ref[:, col])
            errors.append(np.sqrt(integral(knots, np.interp(knots, t, values) - target, squared=True) /
                                  integral(knots, target, squared=True)))
        rows.append(dict(model=label, minimum_pressure_cd=p[i], time_s=t[i], pressure_l2=errors[0],
                         viscous_l2=errors[1], minimum_clearance_mm=clearance.min(), final_drift_kpa=drift[-1]))
        for ax, values in zip(axes.flat, (p, v, clearance, drift)):
            ax.plot(t, values, label=label)
    for ax, col in zip(axes[0], (1, 2)):
        ax.plot(ref[:, 0], ref[:, col], "k--", label="FD reference")
    for ax, label in zip(axes.flat, ("Pressure Cd", "Viscous Cd", "Minimum clearance [mm]", "Mean pressure drift [kPa]")):
        ax.set(xlim=(0, 0.9), xlabel="Time [s]", ylabel=label)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].axhline(0, color="0.5", lw=0.8)
    fig.suptitle("Original DBC · Re = 100 · L/dp = 80 · CFL = 0.10")
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    times = tuple(dict.fromkeys((rows[1]["time_s"] if compare_diffusion else 0.84, 0.88)))
    panels = tuple((tag, f"{label}\n{time:.2f} s") for time in times for tag, label in models)
    frames = [read_frame(tag, time) for time in times for tag, _ in models]
    for field in ("speed", "pressure"):
        plot_fields(frames, field, models=panels, out=out, subject=" · Original DBC", time=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-gravity", action="store_true")
    if parser.parse_args().zero_gravity:
        main((MODELS[1], ("dbc_n80_half-step-planar-zero-gravity", "Zero-gravity linear diffusion · PlanarShifting")),
             OUT.with_name("dbc_planar_zero_gravity"), compare_diffusion=True)
    else:
        main()
