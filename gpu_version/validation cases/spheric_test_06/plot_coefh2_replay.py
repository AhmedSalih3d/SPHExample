"""Compare each completed coefh replay against its saved baseline."""
import csv
import math
import tomllib

import h5py
import numpy as np

from plot_coefh_refinement import integral, read_reference, read_table
from plot_regularized_mdbc_fields import CASE, RUNS, plt


OUT = CASE / "results" / "regularized_mdbc" / "coefh2"


def relative_error(t, values, reference, column):
    rt = reference[:, 0]
    knots = np.union1d(t[t >= 0.1], rt[(rt > 0.1) & (rt < t[-1])])
    expected = np.interp(knots, rt, reference[:, column])
    return np.sqrt(integral(knots, np.interp(knots, t, values) - expected, squared=True) /
                   integral(knots, expected, squared=True))


def clearance_event(directory, times):
    diagnostics = read_table(directory / "native_state.csv")
    np.testing.assert_array_equal(diagnostics["time"], times)
    index = int(np.argmin(diagnostics["minimum_clearance"]))
    with h5py.File(directory / "state" / "MovingSquare.vtkhdf") as handle:
        root = handle["VTKHDF/Particles"] if "VTKHDF/Particles" in handle else handle["VTKHDF"]
        np.testing.assert_allclose(root["Steps/Values"][index], times[index], atol=1e-12, rtol=0)
        offset = int(root["Steps/PointOffsets"][index])
        span = slice(offset, offset + int(root["NumberOfPoints"][index]))
        markers = root["PointData/GroupMarker"][span]
        points = root["Points"][span, :2]
    fluid = points[markers == 2]
    center = points[markers == 3].mean(axis=0)
    square_clearance = np.max(np.abs(fluid - center), axis=1).min() - 0.5
    tank_clearance = min(fluid.min(), (np.array([10.0, 5.0]) - fluid).min())
    return (float(times[index]), "square" if square_clearance <= tank_clearance else "tank",
            float(diagnostics["minimum_clearance"][index] * 1000))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference = read_reference(CASE / "reference" / "Force_Re100.dat")
    tags = sorted(p.parent.name for p in RUNS.glob("*/native.toml"))
    metrics = []
    for tag in tags:
        replay = RUNS / "coefh2" / tag
        if not (replay / "native.toml").exists():
            continue
        old, new = [read_table(path / "native.csv") for path in (RUNS / tag, replay)]
        t = old["time"]
        np.testing.assert_array_equal(t, new["time"])
        for coefh, table, directory in ((1.2, old, RUNS / tag), (2.0, new, replay)):
            meta = tomllib.loads((directory / "native.toml").read_text())
            p, v = table["cd_pressure"], table["cd_viscous"]
            window = (t >= 0.7) & (t <= 0.9)
            i = np.flatnonzero(window)[np.argmin(p[window])]
            clearance_t, clearance_wall, clearance_mm = clearance_event(directory, t)
            metrics.append(dict(case=tag, coefh=coefh, n=meta["n"], cfl=meta["cfl"],
                                wall_layers=meta["wall_layers"], h=meta["h"], duration_s=t[-1],
                                steps=meta["steps"], startup_min_cd_pressure=p[i], startup_min_time_s=t[i],
                                startup_pressure_l2=relative_error(t[t <= 0.9], p[t <= 0.9], reference, 1),
                                pressure_l2=relative_error(t, p, reference, 1),
                                viscous_l2=relative_error(t, v, reference, 2),
                                minimum_clearance_mm=clearance_mm, minimum_clearance_time_s=clearance_t,
                                minimum_clearance_wall=clearance_wall))
    if not metrics:
        raise RuntimeError("No completed coefh replays")
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    before, after = metrics[::2], metrics[1::2]
    indices = np.arange(len(after))
    fig, axes = plt.subplots(1, 2, figsize=(13, 1.8 + 0.34 * len(after)), sharey=True,
                             layout="constrained")
    for ax, key, multiplier, title in zip(axes, ("startup_pressure_l2", "minimum_clearance_mm"),
                                         (100, 1), ("Startup pressure drag error [%]",
                                                    "Minimum saved wall clearance [mm]")):
        ax.barh(indices - 0.18, [multiplier * row[key] for row in before], 0.36, label="coefh 1.2")
        ax.barh(indices + 0.18, [multiplier * row[key] for row in after], 0.36, label="coefh 2.0")
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(indices, [row["case"] for row in after])
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / "startup_error_and_clearance.png", dpi=155)
    plt.close(fig)

    groups = (("startup", [r for r in after if r["duration_s"] < 1]),
              ("full", [r for r in after if r["duration_s"] >= 1]))
    for name, cases in groups:
        if not cases:
            continue
        rows = math.ceil(len(cases) / 2)
        fig, axes = plt.subplots(rows, 2, figsize=(12, 2.7 * rows), squeeze=False, layout="constrained")
        for ax, result in zip(axes.flat, cases):
            tag = result["case"]
            ax.plot(reference[:, 0], reference[:, 1], "k--", label="FD reference")
            for directory, label in ((RUNS / tag, "coefh 1.2"), (RUNS / "coefh2" / tag, "coefh 2.0")):
                history = read_table(directory / "native.csv")
                ax.plot(history["time"], history["cd_pressure"], label=label)
            ax.set(xlim=(0, result["duration_s"]), title=tag, xlabel="Time [s]", ylabel="Pressure drag Cd")
            ax.axhline(0, color="0.6", lw=0.8)
            ax.grid(alpha=0.25)
        axes.flat[0].legend(fontsize=8)
        for ax in list(axes.flat)[len(cases):]:
            ax.set_visible(False)
        fig.savefig(OUT / f"{name}_pressure_drag.png", dpi=155)
        plt.close(fig)


if __name__ == "__main__":
    main()
