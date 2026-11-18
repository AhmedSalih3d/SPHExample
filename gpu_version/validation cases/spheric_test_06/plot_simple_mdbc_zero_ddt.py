"""Compare raw forces and pressure drift after the selected particle history closes."""
import argparse
import csv
import runpy
import tomllib
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE = Path(__file__).resolve().parent
STEM = "Re100_n080"


def mean_fluid_pressure(path, time):
    with h5py.File(path, "r") as file:
        root = file["VTKHDF/Particles"] if "VTKHDF/Particles" in file else file["VTKHDF"]
        np.testing.assert_array_equal(root["Steps/Values"][:], time)
        counts = root["NumberOfPoints"][:]
        assert len(counts) == len(time) and np.all(counts == 332100)
        pressure = root["PointData/Pressure"]
        markers = root["PointData/GroupMarker"]
        result = []
        for offset, count in zip(root["Steps/PointOffsets"][:], counts):
            frame = slice(offset, offset + count)
            fluid = markers[frame] == 2
            assert fluid.sum() == 313600
            result.append(pressure[frame][fluid].mean(dtype=np.float64))
        return np.array(result)


def main(startup):
    duration = 0.9 if startup else 8.0
    tag = "simple_mdbc_zero_ddt_startup" if startup else "simple_mdbc_zero_ddt"
    out = CASE / "results" / tag
    baseline = CASE / "results" / "zero_ddt_full"
    read = lambda path: np.genfromtxt(path, delimiter=",", names=True)
    control, simple = read(baseline / f"{STEM}.csv"), read(out / "native.csv")
    control, simple_state = control[control["time"] <= duration], read(out / "native_state.csv")
    dbc_state = read(baseline / f"{STEM}_state.csv")[:len(control)]
    m = tomllib.loads((out / "native.toml").read_text(encoding="utf-8"))
    b = tomllib.loads((baseline / f"{STEM}.toml").read_text(encoding="utf-8"))
    assert m["boundary_mode"] == "SimpleMDBC" and b["boundary_mode"] == "NoMDBC"
    assert m["diffusion"] == b["density_diffusion_model"] == "ZeroDensityDiffusion"
    assert m["motion"] == b["motion_kind"] == "integrated"
    for key in ("dx", "sound_speed", "h", "cfl", "background_pressure", "wall_layers", "reynolds"):
        assert m[key] == b[key], key
    time = simple["time"]
    assert len(time) == m["frames"] == round(duration / 0.02) + 1 and time[0] == 0 and time[-1] == duration
    for table in (control, dbc_state, simple_state):
        np.testing.assert_array_equal(table["time"], time)
    for table in (control, simple, dbc_state, simple_state):
        assert all(np.isfinite(table[field]).all() for field in table.dtype.names)
    for state in (simple_state, dbc_state):
        assert np.all(state["fluid_particles"] == 313600) and np.all(state["square_particles"] == 6400)
    p_simple = mean_fluid_pressure(CASE / "runs" / tag / STEM / "MovingSquare.vtkhdf", time)
    assert np.isfinite(p_simple).all()

    analysis = runpy.run_path(str(CASE / "plot_coefh_refinement.py"))
    ref = analysis["read_reference"](CASE / "reference" / "Force_Re100.dat")
    rt = ref[:, 0]
    knots = np.unique(np.r_[0.1, duration, time[(time > 0.1) & (time < duration)],
                            rt[(rt > 0.1) & (rt < duration)]])
    energy = analysis["integral"]
    target = lambda column: ref[:, column] if column else ref[:, 1] + ref[:, 2]

    fig, grid = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    axes = list(grid.flat)
    rows = []
    with (out / "mean_fluid_pressure.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("time", "simple_mdbc_mean_pressure_pa", "dbc_mean_pressure_pa"))
        writer.writerows(zip(time, p_simple, dbc_state["pressure_mean"]))
    cases = (
        ("DBC, no diffusion", "#0072b2", control, dbc_state, dbc_state["pressure_mean"]),
        ("SimpleMDBC, no diffusion", "#d55e00", simple, simple_state, p_simple),
    )
    for label, color, data, state, mean_pressure in cases:
        row = dict(boundary=label, duration_s=duration, minimum_clearance_mm=1000 * state["minimum_clearance"].min(),
                   fluid_mean_pressure_final_pa=mean_pressure[-1], initial_fluid_mean_pressure_pa=mean_pressure[0],
                   density_departure_percent=100 * state["density_relative_max"].max())
        forces = (data["cd_pressure"], data["cd_viscous"], data["cd_pressure"] + data["cd_viscous"])
        for ax, name, values, column in zip(axes, ("pressure", "viscous", "total"), forces, (1, 2, 0)):
            exact = np.interp(knots, rt, target(column))
            error = np.interp(knots, time, values) - exact
            row[f"{name}_error"] = float(np.sqrt(energy(knots, error, squared=True) / energy(knots, exact, squared=True)))
            row[f"{name}_cd_at_0_74"] = float(values[np.where(time == 0.74)[0][0]])
            row[f"{name}_cd_at_end"] = float(values[-1])
            row[f"{name}_cd_min"] = float(values.min())
            row[f"{name}_min_time_s"] = float(time[values.argmin()])
            row[f"{name}_cd_peak"] = float(values.max())
            ax.plot(time, values, lw=1.2, label=label, color=color)
        axes[3].plot(time, mean_pressure / 1000, lw=1.2, label=label, color=color)
        rows.append(row)

    for ax, column in zip(axes, (1, 2, 0)):
        samples = rt <= duration
        ax.plot(rt[samples], target(column)[samples], "k--", lw=1.1, label="Official FD reference")
    for ax, ylabel in zip(axes, ("Pressure drag Cd", "Viscous drag Cd", "Total drag Cd", "Mean fluid pressure [kPa]")):
        ax.axvline(0.74, color="grey", ls=":", lw=0.8)
        ax.set(xlim=(0, duration), ylabel=ylabel, xlabel="Time [s]")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    checkpoint = "startup checkpoint (separate matched replay)" if startup else "full run"
    fig.suptitle(f"Re=100, L/dp=80: {checkpoint}\n"
                 "c0=160 m/s, coefh=1.2, ZeroDensityDiffusion; raw forces")
    fig.tight_layout()
    fig.savefig(out / "force_comparison.png", dpi=180)
    plt.close(fig)
    with (out / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--startup", action="store_true", help="Plot the completed 0.9 s replay.")
    main(parser.parse_args().startup)
