"""Plot unfiltered DBC+ startup controls, optionally including the impedance variant."""
import argparse
import csv
import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "dbc_plus"


def main(include_impedance=False, full=False):
    analysis = runpy.run_path(str(ROOT / "plot_coefh_refinement.py"))
    reference = analysis["read_reference"](ROOT / "reference" / "Force_Re100.dat")
    duration = 8.0 if full else 0.9
    expected_frames = 401 if full else 46
    cases = [("DBC, fluid-only DDT", ROOT / "results" / "fluid_only_ddt" / "Re100_n080", "#0072b2")]
    if not full:
        cases.append(("Experimental DBC+", OUT / "n80_t0.9_graph" / "native", "#d55e00"))
    if include_impedance or full:
        cases.append(("DBC+ with impedance", OUT / f"impedance_n80_t{duration}_graph" / "native", "#009e73"))

    def relative_error(time, values, column):
        ref_t = reference[:, 0]
        t = np.unique(np.r_[0.1, duration, time[(time > .1) & (time < duration)],
                            ref_t[(ref_t > .1) & (ref_t < duration)]])
        target = np.interp(t, ref_t, reference[:, column])
        diff = np.interp(t, time, values) - target
        energy = analysis["integral"]
        return float(np.sqrt(energy(t, diff, squared=True) / energy(t, target, squared=True)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    records = []
    for label, stem, color in cases:
        d = np.genfromtxt(stem.with_suffix(".csv"), delimiter=",", names=True)
        s = np.genfromtxt(Path(str(stem) + "_state.csv"), delimiter=",", names=True)
        mask = d["time"] <= duration + 1e-9
        d, s = d[mask], s[mask]
        assert len(d) == len(s) == expected_frames and d["time"][-1] == duration
        assert np.array_equal(d["time"], s["time"])
        assert all(np.isfinite(d[key]).all() for key in d.dtype.names)
        assert all(np.isfinite(s[key]).all() for key in s.dtype.names)
        assert np.all(s["fluid_particles"] == 313600) and np.all(s["square_particles"] == 6400)
        for ax, values in ((axes[0, 0], d["cd_pressure"]), (axes[0, 1], d["cd_viscous"]),
                           (axes[1, 0], -1000*np.minimum(s["minimum_clearance"], 0)),
                           (axes[1, 1], 100*s["density_relative_max"])):
            ax.plot(d["time"], values, color=color, label=label)
        penetrated = s["minimum_clearance"] < 0
        row = dict(mode=label, pressure_min=float(d["cd_pressure"].min()),
                   pressure_min_time=float(d["time"][d["cd_pressure"].argmin()]),
                   pressure_peak=float(d["cd_pressure"].max()), viscous_peak=float(d["cd_viscous"].max()),
                   penetration_mm=float(-1000*min(s["minimum_clearance"].min(), 0)),
                   density_departure_percent=float(100*s["density_relative_max"].max()))
        if include_impedance or full:
            row.update(minimum_clearance_mm=float(1000*s["minimum_clearance"].min()),
                       first_penetration_s=float(s["time"][penetrated][0]) if penetrated.any() else "",
                       pressure_error=relative_error(d["time"], d["cd_pressure"], 1),
                       viscous_error=relative_error(d["time"], d["cd_viscous"], 2))
        records.append(row)
    for ax, col in ((axes[0, 0], 1), (axes[0, 1], 2)):
        mask = reference[:, 0] <= duration
        ax.plot(reference[mask, 0], reference[mask, col], color="black", ls="--", label="Official FD")
    for ax, ylabel in zip(axes.flat, ("Pressure Cd", "Viscous Cd", "Maximum penetration [mm]", "Density departure [%]")):
        ax.set(xlim=(0, duration), xlabel="Time [s]", ylabel=ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle(f"DBC+ pressure-boundary controls: Re=100, L/dp=80, 0-{duration:g} s\n"
                 "c0=160, coefh=1.2, fluid-only DDT, physical DBC viscosity; all curves raw")
    fig.tight_layout()
    plot_name = "impedance_full_comparison" if full else ("impedance_comparison" if include_impedance else "startup_comparison")
    fig.savefig(OUT / f"{plot_name}.png", dpi=180)
    fig.savefig(OUT / f"{plot_name}.pdf")
    summary_name = "impedance_full_summary.csv" if full else ("impedance_summary.csv" if include_impedance else "summary.csv")
    with (OUT / summary_name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--impedance", action="store_true", help="Include the DBC+ impedance startup run.")
    options.add_argument("--full", action="store_true", help="Compare the full 8 s DBC+ impedance run against DBC.")
    args = parser.parse_args()
    main(args.impedance, args.full)
