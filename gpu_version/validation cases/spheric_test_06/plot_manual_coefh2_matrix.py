"""Compare all completed manual-case coefh 2.0 runs with the Test 06 reference."""
import csv
import re
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_coefh_refinement import integral, read_reference, read_table


CASE = Path(__file__).resolve().parent
OUT = CASE / "results" / "manual_coefh2_matrix"
WALLS = {"dbc": "DBC", "simple-mdbc": "SimpleMDBC"}
MODELS = {
    "zero-gravity-linear": "Zero-gravity linear",
    "zero": "Zero diffusion",
    "fluid-linear": "Fluid-only linear",
    "complex": "Complex",
    "fourtakas-inner": "Fourtakas inner (DDT2)",
    "fourtakas-full": "Fourtakas full (DDT3)",
}
RESOLUTIONS = (20, 40, 80)
REYNOLDS = (50, 100, 150)
COLORS = dict(zip(RESOLUTIONS, ("#0072b2", "#e69f00", "#009e73")))


def relative_error(t, values, ref_t, reference, end):
    knots = np.unique(np.r_[0.1, end, t[(t > 0.1) & (t < end)], ref_t[(ref_t > 0.1) & (ref_t < end)]])
    expected = np.interp(knots, ref_t, reference)
    return np.sqrt(integral(knots, np.interp(knots, t, values) - expected, squared=True) /
                   integral(knots, expected, squared=True))


def analyze(record, reference):
    meta = tomllib.loads(record.read_text())
    force = read_table(record.with_suffix(".csv"))
    state = read_table(record.with_name(record.stem + "_state.csv"))
    t = force["time"]
    np.testing.assert_allclose(t, np.linspace(0, 8, 401), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(t, state["time"])
    p, v = force["cd_pressure"], force["cd_viscous"]
    assert np.isfinite(p).all() and np.isfinite(v).all()
    ref_t, rp, rv = reference.T
    selection = np.flatnonzero((t >= 0.7) & (t <= 0.9))
    i = selection[np.argmin(p[selection])]
    return force, dict(
        startup_min_pressure_cd=p[i], startup_min_time_s=t[i],
        startup_pressure_l2=relative_error(t, p, ref_t, rp, 0.9),
        pressure_l2=relative_error(t, p, ref_t, rp, 8),
        viscous_l2=relative_error(t, v, ref_t, rv, 8),
        total_l2=relative_error(t, p + v, ref_t, rp + rv, 8),
        minimum_clearance_mm=1000 * state["minimum_clearance"].min(),
        final_mean_pressure_change_kpa=(state["pressure_mean"][-1] - state["pressure_mean"][0]) / 1000,
        density_relative_max=meta["density_relative_max"], frames=meta["frames"], steps=meta["steps"],
    )


def main():
    rows = []
    for re, wall in ((re, wall) for re in REYNOLDS for wall in WALLS):
        reference = read_reference(CASE / "reference" / f"Force_Re{re:03}.dat")
        fig, axes = plt.subplots(6, 2, figsize=(12.5, 14), sharex=True, layout="constrained")
        present = False
        for (model, title), (pressure_ax, viscous_ax) in zip(MODELS.items(), axes):
            for ax, col, label in ((pressure_ax, 1, "Pressure Cd"), (viscous_ax, 2, "Viscous Cd")):
                ax.plot(reference[:, 0], reference[:, col], "k--", lw=1.2, label="FD reference")
                ax.set(ylabel=label, xlim=(0, 8))
                ax.grid(alpha=0.22)
            pressure_ax.set_title(title, loc="left")
            for n in RESOLUTIONS:
                record = OUT / wall / model / f"Re{re:03}_n{n:03}.toml"
                if not record.exists():
                    continue
                force, metrics = analyze(record, reference)
                rows.append(dict(reynolds=re, n=n, dx=1/n, boundary=wall, diffusion=model, **metrics))
                present = True
                for ax, key in ((pressure_ax, "cd_pressure"), (viscous_ax, "cd_viscous")):
                    ax.plot(force["time"], force[key], lw=1, color=COLORS[n], label=f"L/dp = {n}")
            pressure_ax.legend(fontsize=8, ncol=2, loc="upper right")
        for ax in axes[-1]:
            ax.set_xlabel("Time [s]")
        if present:
            fig.suptitle(f"SPHERIC Test 06 · Re = {re} · {WALLS[wall]} · PlanarShifting · coefh = 2.0")
            fig.savefig(OUT / f"Re{re:03}_{wall}_drag.png", dpi=150)
        plt.close(fig)

    if not rows:
        raise RuntimeError("No completed simulations found")
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    expected = len(REYNOLDS) * len(RESOLUTIONS) * len(WALLS) * len(MODELS)
    print(f"Compared {len(rows)} of {expected} runs")


if __name__ == "__main__":
    main()
