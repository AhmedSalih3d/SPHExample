"""Measure crossings of the nominal moving square and outer tank walls."""
import csv
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_coefh_refinement import integral, read_reference, read_table


CASE = Path(__file__).resolve().parent
TAG = "wall_penetration_controls"
OUT = CASE / "results" / TAG


def clearances(fluid, wall, tolerance):
    relative = fluid - wall.mean(axis=0)
    square = np.max(np.abs(relative), axis=1) - 0.5
    tank = np.minimum(fluid, np.array([10., 5.]) - fluid).min(axis=1)
    return square.min(), (square < -tolerance).sum(), (tank < -tolerance).sum()


def crossings(path, times):
    with h5py.File(path) as handle:
        root = handle["VTKHDF/Particles"] if "VTKHDF/Particles" in handle else handle["VTKHDF"]
        count = len(times)
        np.testing.assert_allclose(times, root["Steps/Values"][:count], atol=1e-12, rtol=0)
        for offset, size in zip(root["Steps/PointOffsets"][:count], root["NumberOfPoints"][:count]):
            span = slice(int(offset), int(offset + size))
            points = root["Points"][span, :2]
            markers = root["PointData/GroupMarker"][span]
            yield clearances(points[markers == 2], points[markers == 3], 1e-10)


def dual_crossings(directory, n):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    stats = directory / "positions_stats.csv"
    reader = vtk.vtkPolyDataReader()
    reader.ReadAllScalarsOn()
    with stats.open() as stream:
        for row in csv.DictReader(list(stream)[3:], delimiter=";"):
            reader.SetFileName(str(directory / f"positions_{int(row['Part']):04}.vtk"))
            reader.Update()
            mesh = reader.GetOutput()
            points = vtk_to_numpy(mesh.GetPoints().GetData())[:, [0, 2]].astype(float)
            types = vtk_to_numpy(mesh.GetPointData().GetArray("Type"))
            fluid, wall = points[types == 3], points[types == 1]
            assert len(fluid) == 49*n*n and len(wall) == n*n
            yield float(row["Time [s]"]), *clearances(fluid, wall, 1e-6)


def write_dual_comparison():
    native = CASE / "runs" / "manual_coefh2_matrix" / "dbc" / "zero-gravity-linear"
    runs = CASE / "runs" / TAG
    n80 = runs / "dual-n080" / "vtk"
    times = np.linspace(0, 8, 401)
    entries = []
    for n in (40, 80):
        path = native / f"Re100_n{n:03}" / "MovingSquare.vtkhdf"
        rows = [(time, *measurements) for time, measurements in zip(times, crossings(path, times))]
        entries.append((f"Native DBC n{n}", rows))
    if (runs / "dual-pilot" / "positions_stats.csv").exists():
        entries.append(("DualSPHysics n40 coefh 1.2", list(dual_crossings(runs / "dual-pilot", 40))))
    if (n80 / "positions_stats.csv").exists():
        entries.append(("DualSPHysics n80 coefh 2.0 NoFixed", list(dual_crossings(n80, 80))))
    matched = runs / "dual-n040-matched" / "vtk"
    if (matched / "positions_stats.csv").exists():
        entries.append(("DualSPHysics n40 coefh 2.0 no shifting", list(dual_crossings(matched, 40))))
        short_t = np.linspace(0, 2, 101)
        path = runs / "NoMDBC" / "no-shifting" / "MovingSquare.vtkhdf"
        entries.append(("Native DBC n40 coefh 2.0 no shifting",
                        [(time, *data) for time, data in zip(short_t, crossings(path, short_t))]))
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    for label, data in entries:
        t, mm, square, tank = np.array(data).T
        expected_end = 2 if "no shifting" in label else 8
        assert len(t) == expected_end * 50 + 1 and t[0] == 0 and abs(t[-1] - expected_end) < 0.001
        mm *= 1000
        indices = np.flatnonzero(square)
        rows.append(dict(case=label, frames=len(t), first_crossing_s=t[indices[0]] if len(indices) else "",
                         minimum_square_clearance_mm=mm.min(), minimum_time_s=t[mm.argmin()],
                         peak_square_crossings=int(square.max()), peak_tank_crossings=int(tank.max())))
        axes[0].plot(t, mm, label=label)
        axes[1].plot(t, square, label=label)
    axes[0].set_ylabel("Moving square clearance [mm]")
    axes[0].axhline(0, color="0.5", ls=":")
    axes[1].set(xlabel="Time [s]", ylabel="Fluid centers inside square")
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / "dual_comparison.png", dpi=170)
    with (OUT / "dual_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def native_metrics(boundary, label, force, frame_metrics, reference, n=40):
    t = force["time"]
    clearance, square, tank = frame_metrics.T
    indices = np.flatnonzero(square)
    end = t[-1]
    ref_t = reference[:, 0]
    knots = np.unique(np.r_[t[t >= .1], ref_t[(ref_t > .1) & (ref_t < end)]])
    errors = []
    for column, key in ((1, "cd_pressure"), (2, "cd_viscous")):
        ref = np.interp(knots, ref_t, reference[:, column])
        errors.append(np.sqrt(integral(knots, np.interp(knots, t, force[key]) - ref, squared=True) /
                              integral(knots, ref, squared=True)))
    return dict(boundary=boundary, test=label, n=n, duration_s=end,
                first_crossing_s=t[indices[0]] if len(indices) else "",
                minimum_square_clearance_mm=1000 * clearance.min(), peak_square_crossings=int(square.max()),
                peak_tank_crossings=int(tank.max()), pressure_drag_l2=errors[0], viscous_drag_l2=errors[1])


def write_fine_comparison():
    reference = read_reference(CASE / "reference" / "Force_Re100.dat")
    clearance_fig, clearance_axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    drag_fig, drag_axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    baseline_root = Path("manual_coefh2_matrix")
    native_runs = [
        (label, baseline_root / wall / "zero-gravity-linear", "Re100_n080.csv", "Re100_n080/MovingSquare.vtkhdf")
        for wall, label in (("dbc", "DBC"), ("simple-mdbc", "SimpleMDBC"))
    ] + [
        (mode, Path(TAG) / f"{mode}_n080", "native.csv", "MovingSquare.vtkhdf")
        for mode in ("DBCPlusImpedance", "UpdatedMDBCWithImpedance")
    ]
    for label, directory, history, particles in native_runs:
        force = read_table(CASE / "results" / directory / history)
        force = force[force["time"] <= 2.0]
        t = force["time"]
        np.testing.assert_allclose(t, np.linspace(0, 2, 101), atol=1e-12, rtol=0)
        clearance, count, _ = np.array(list(crossings(CASE / "runs" / directory / particles, t))).T
        clearance_axes[0].plot(t, 1000 * clearance, label=label)
        clearance_axes[1].step(t, count, where="mid", label=label)
        for ax, key in zip(drag_axes, ("cd_pressure", "cd_viscous")):
            ax.plot(t, force[key], label=label)
    dual = CASE / "runs" / TAG / "dual-n080" / "vtk"
    if (dual / "positions_stats.csv").exists():
        data = [row for row in dual_crossings(dual, 80) if row[0] <= 2.0001]
        t, clearance, count, _ = np.asarray(data).T
        label = "DualSPHysics DBC (NoFixed shifting)"
        clearance_axes[0].plot(t, 1000 * clearance, "--", label=label)
        clearance_axes[1].step(t, count, where="mid", linestyle="--", label=label)
    for column, ax in enumerate(drag_axes, start=1):
        ax.plot(reference[:, 0], reference[:, column], "k--", label="FD reference")
    title = "SPHERIC Test 06 · Re = 100 · L/dp = 80 · coefh = 2.0"
    for fig, axes, labels, filename in (
        (clearance_fig, clearance_axes, ("Square clearance [mm]", "Fluid centers inside square"), "n080_clearance.png"),
        (drag_fig, drag_axes, ("Pressure drag Cd", "Viscous drag Cd"), "n080_drag.png"),
    ):
        for ax, label in zip(axes, labels):
            ax.set(xlim=(0, 2), ylabel=label)
            ax.grid(alpha=.22)
        axes[0].axhline(0, color="0.5", ls=":", lw=0.9)
        axes[0].legend(fontsize=8, ncol=2)
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(title)
        fig.savefig(OUT / filename, dpi=175)
        plt.close(fig)


def main():
    reference = read_reference(CASE / "reference" / "Force_Re100.dat")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, layout="constrained")
    full_fig, full_axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
    rows = []
    for wall, mode, (clearance_ax, drag_ax) in zip(("dbc", "simple-mdbc"), ("NoMDBC", "SimpleMDBC"), axes):
        baseline = Path("manual_coefh2_matrix") / wall / "zero-gravity-linear"
        baseline_results = CASE / "results" / baseline / "Re100_n040.csv"
        baseline_particles = CASE / "runs" / baseline / "Re100_n040" / "MovingSquare.vtkhdf"
        force = read_table(baseline_results)
        frames = np.array(list(crossings(baseline_particles, force["time"])))
        rows.append(native_metrics(wall, "baseline-full", force, frames, reference))
        for ax, values in zip(full_axes, (1000*frames[:, 0], force["cd_pressure"], force["cd_viscous"])):
            ax.plot(force["time"], values, label=mode)
        trials = [("baseline", baseline)] + [
            (p.parent.name, Path(TAG) / mode / p.parent.name)
            for p in sorted((OUT / mode).glob("*/Re100_n040.toml"))
        ]
        drag_ax.plot(reference[:, 0], reference[:, 1], "k--", label="FD reference")
        for label, directory in trials:
            f = read_table(CASE / "results" / directory / "Re100_n040.csv")
            f = f[f["time"] <= 2.0]
            t, p = f["time"], f["cd_pressure"]
            frames = np.array(list(crossings(
                CASE / "runs" / directory / "Re100_n040" / "MovingSquare.vtkhdf"
                if label == "baseline" else CASE / "runs" / directory / "MovingSquare.vtkhdf", t)))
            rows.append(native_metrics(wall, label, f, frames, reference))
            clearance_ax.plot(t, 1000 * frames[:, 0], label=label)
            drag_ax.plot(t, p, label=label)
        clearance_ax.axhline(0, ls=":", color="0.3")
        clearance_ax.set(ylabel=f"{mode} clearance [mm]", xlim=(0, 2))
        drag_ax.set(ylabel=f"{mode} pressure Cd", xlim=(0, 2))
        for ax in (clearance_ax, drag_ax):
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    for mode in ("UpdatedMDBC", "DBCPlusImpedance", "DBCPlusImpedance_full", "DBCPlusImpedance_n080",
                 "UpdatedMDBCWithImpedance", "UpdatedMDBCWithImpedance_full", "UpdatedMDBCWithImpedance_n080"):
        directory = OUT / mode
        if not (directory / "native.toml").exists():
            continue
        f = read_table(directory / "native.csv")
        t = f["time"]
        frames = np.array(list(crossings(
            CASE / "runs" / TAG / mode / "MovingSquare.vtkhdf", t)))
        rows.append(native_metrics(mode, "boundary-models", f, frames, reference,
                                   n=80 if mode.endswith("_n080") else 40))
        if mode.endswith("_full"):
            for ax, values in zip(full_axes, (1000*frames[:, 0], f["cd_pressure"], f["cd_viscous"])):
                ax.plot(t, values, label=mode.removesuffix("_full"))
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    fig.savefig(OUT / "comparison.png", dpi=170)
    for ax, column in zip(full_axes[1:], (1, 2)):
        ax.plot(reference[:, 0], reference[:, column], "k--", label="FD reference")
    full_axes[0].axhline(0, color="0.5", ls=":")
    for ax, label in zip(full_axes, ("Moving square clearance [mm]", "Pressure Cd", "Viscous Cd")):
        ax.set(ylabel=label, xlim=(0, 8))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    full_axes[-1].set_xlabel("Time [s]")
    full_fig.savefig(OUT / "full_comparison.png", dpi=170)
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if sys.argv[1:] == ["--fine"]:
        write_fine_comparison()
    else:
        main()
        write_dual_comparison()
