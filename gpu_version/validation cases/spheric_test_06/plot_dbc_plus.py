"""Plot unfiltered experimental DBC+ startup and its matched fluid-only-DDT DBC control."""
import csv
import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "dbc_plus"


def main():
    read_reference = runpy.run_path(str(ROOT / "plot_coefh_refinement.py"))["read_reference"]
    reference = read_reference(ROOT / "reference" / "Force_Re100.dat")
    cases = (
        ("DBC, fluid-only DDT", ROOT / "results" / "fluid_only_ddt" / "Re100_n080", "#0072b2"),
        ("Experimental DBC+", OUT / "n80_t0.9_graph" / "native", "#d55e00"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    records = []
    for label, stem, color in cases:
        d = np.genfromtxt(stem.with_suffix(".csv"), delimiter=",", names=True)
        s = np.genfromtxt(Path(str(stem) + "_state.csv"), delimiter=",", names=True)
        mask = d["time"] <= .9 + 1e-9
        d, s = d[mask], s[mask]
        assert len(d) == len(s) == 46 and np.array_equal(d["time"], s["time"])
        assert all(np.isfinite(d[key]).all() for key in d.dtype.names)
        assert all(np.isfinite(s[key]).all() for key in s.dtype.names)
        assert np.all(s["fluid_particles"] == 313600) and np.all(s["square_particles"] == 6400)
        for ax, values in ((axes[0, 0], d["cd_pressure"]), (axes[0, 1], d["cd_viscous"]),
                           (axes[1, 0], -1000*np.minimum(s["minimum_clearance"], 0)),
                           (axes[1, 1], 100*s["density_relative_max"])):
            ax.plot(d["time"], values, color=color, label=label)
        records.append(dict(mode=label, pressure_min=float(d["cd_pressure"].min()),
                            pressure_min_time=float(d["time"][d["cd_pressure"].argmin()]),
                            pressure_peak=float(d["cd_pressure"].max()), viscous_peak=float(d["cd_viscous"].max()),
                            penetration_mm=float(-1000*min(s["minimum_clearance"].min(), 0)),
                            density_departure_percent=float(100*s["density_relative_max"].max())))
    for ax, col in ((axes[0, 0], 1), (axes[0, 1], 2)):
        mask = reference[:, 0] <= .9
        ax.plot(reference[mask, 0], reference[mask, col], color="black", ls="--", label="Official FD")
    for ax, ylabel in zip(axes.flat, ("Pressure Cd", "Viscous Cd", "Maximum penetration [mm]", "Density departure [%]")):
        ax.set(xlim=(0, .9), xlabel="Time [s]", ylabel=ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("DBC+ isolated pressure-boundary startup control: Re=100, L/dp=80\n"
                 "c0=160, coefh=1.2, fluid-only DDT, physical DBC viscosity; all curves raw")
    fig.tight_layout()
    fig.savefig(OUT / "startup_comparison.png", dpi=180)
    fig.savefig(OUT / "startup_comparison.pdf")
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(records)


if __name__ == "__main__":
    main()
