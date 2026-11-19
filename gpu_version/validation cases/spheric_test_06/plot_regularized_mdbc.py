"""Compare matched RegularizedMDBC and SimpleMDBC runs using raw output samples."""
import argparse
import csv
import shutil
import tomllib
from pathlib import Path

from plot_coefh_refinement import integral, read_reference, read_table
from plot_simple_mdbc_zero_ddt import mean_fluid_pressure
import matplotlib.pyplot as plt
import numpy as np


CASE = Path(__file__).resolve().parent
RUNS = CASE / "runs" / "regularized_mdbc_experiment"


def compare(study):
    n = 40 if study == "coarse" else 80
    suffix = "" if study == "startup" else f"_{study}"
    root = CASE / "results" / "regularized_mdbc"
    out = root if study == "startup" else root / study
    out.mkdir(parents=True, exist_ok=True)
    duration = 8.0 if study == "full" else 0.9
    reference = read_reference(CASE / "reference" / "Force_Re100.dat")
    rt = reference[:, 0]
    labels = (("control", "SimpleMDBC", "#0072b2"), ("regularized", "RegularizedMDBC", "#d55e00"))
    if study == "half-step" and (RUNS / "impedance_n80_half-step" / "native.toml").is_file():
        labels += (("impedance", "DBCPlusImpedance", "#009e73"),)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    pressure_axis, viscous_axis, detail_axis, drift_axis, clearance_axis, density_axis = axes.flat
    rows, history = [], []
    detail_minimum, detail_maximum = 0, 0
    common = None
    for name, label, color in labels:
        directory = RUNS / f"{name}_n{n}{suffix}"
        info = tomllib.loads((directory / "native.toml").read_text(encoding="utf-8"))
        forces = read_table(directory / "native.csv")
        state = read_table(directory / "native_state.csv")
        if name == "control" and "cd_pressure_stored" not in state.dtype.names:
            state = read_table(directory / "stored_wall_state.csv")
        time = forces["time"]
        assert len(time) == info["frames"] == round(duration / 0.02) + 1
        np.testing.assert_allclose(time, np.linspace(0, duration, len(time)), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(state["time"], time)
        for table in (forces, state):
            assert all(np.isfinite(table[field]).all() for field in table.dtype.names)
        assert info["boundary_mode"] == label
        assert info["diffusion"] == "LinearDensityDiffusion"
        assert np.all(state["fluid_particles"] == info["fluid_particles"])
        assert np.all(state["square_particles"] == info["square_particles"])
        settings = tuple(info[key] for key in ("reynolds", "n", "dx", "sound_speed", "h", "cfl", "motion",
                         "wall_layers", "background_pressure", "divergence_cleaning", "diffusion",
                         "density_diffusion_coefficient", "shifting_mode", "particles", "fixed_particles",
                         "timestep_criterion", "gpu", "driver", "julia", "cuda_jl"))
        if common is None:
            common = settings
        assert settings == common, "Unmatched solver settings"

        knots = np.unique(np.r_[0.1, duration, time[(time > 0.1) & (time < duration)],
                               rt[(rt > 0.1) & (rt < duration)]])

        def error(values, target):
            exact = np.interp(knots, rt, target)
            residual = np.interp(knots, time, values) - exact
            return float(np.sqrt(integral(knots, residual, squared=True) / integral(knots, exact, squared=True)))

        pressure, viscous = forces["cd_pressure"], forces["cd_viscous"]
        if name == "impedance":
            mean_pressure = mean_fluid_pressure(directory / "state" / "MovingSquare.vtkhdf", time)
        else:
            mean_pressure = state["mean_fluid_pressure_pa"]
        assert np.isfinite(mean_pressure).all()
        drift = mean_pressure - mean_pressure[0]
        clearance = 1000 * state["minimum_clearance"]
        density_departure = 100 * state["density_relative_max"]
        window = np.flatnonzero((time >= 0.7) & (time <= 0.9))
        minimum = window[pressure[window].argmin()]
        detail = (time >= 0.65) & (time <= 0.9)
        detail_minimum = min(detail_minimum, pressure[detail].min())
        detail_maximum = max(detail_maximum, pressure[detail].max())
        rows.append(dict(boundary=label, steps=info["steps"], frames=info["frames"],
                         pressure_l2=error(pressure, reference[:, 1]), viscous_l2=error(viscous, reference[:, 2]),
                         startup_minimum_pressure_cd=pressure[minimum], startup_minimum_time_s=time[minimum],
                         maximum_pressure_cd=pressure.max(), minimum_clearance_mm=clearance.min(),
                         final_mean_pressure_drift_pa=drift[-1], maximum_mean_pressure_drift_pa=abs(drift).max(),
                         maximum_density_departure_percent=density_departure.max()))
        stored_pressure = stored_viscous = (None,) * len(time)
        if name != "impedance":
            stored_pressure, stored_viscous = state["cd_pressure_stored"], state["cd_viscous_stored"]
            stored_minimum = window[stored_pressure[window].argmin()]
            rows[-1].update(stored_pressure_l2=error(stored_pressure, reference[:, 1]),
                            stored_viscous_l2=error(stored_viscous, reference[:, 2]),
                            startup_minimum_stored_cd=stored_pressure[stored_minimum],
                            startup_minimum_stored_time_s=time[stored_minimum])
            detail_minimum = min(detail_minimum, stored_pressure[detail].min())
            detail_maximum = max(detail_maximum, stored_pressure[detail].max())
            detail_axis.plot(time, stored_pressure, lw=1, ls=":", color=color)
        if study == "full":
            late = np.flatnonzero(time > 0.9)
            late_minimum = late[pressure[late].argmin()]
            late_maximum = late[pressure[late].argmax()]
            rows[-1].update(late_pressure_cd_min=pressure[late_minimum], late_pressure_cd_min_time_s=time[late_minimum],
                            late_pressure_cd_max=pressure[late_maximum], late_pressure_cd_max_time_s=time[late_maximum])
        for values in zip(time, pressure, viscous, stored_pressure, stored_viscous, drift, clearance, density_departure):
            history.append(dict(zip(("boundary", "time_s", "pressure_cd", "viscous_cd", "stored_pressure_cd",
                                     "stored_viscous_cd", "mean_pressure_drift_pa", "clearance_mm",
                                     "density_departure_percent"), (label, *values))))
        for ax, values in ((pressure_axis, pressure), (viscous_axis, viscous), (detail_axis, pressure),
                           (drift_axis, drift), (clearance_axis, clearance), (density_axis, density_departure)):
            ax.plot(time, values, lw=1.25, color=color, label=label)

    for ax, column in ((pressure_axis, 1), (viscous_axis, 2), (detail_axis, 1)):
        ax.plot(rt, reference[:, column], "k--", lw=1, label="FD reference")
    for ax, ylabel in zip(axes.flat, ("Pressure drag Cd", "Viscous drag Cd", "Pressure drag Cd",
                                      "Mean fluid pressure drift [Pa]", "Minimum clearance [mm]",
                                      "Max. fluid density departure [%]")):
        ax.set(xlabel="Time [s]", ylabel=ylabel, xlim=(0, duration))
        ax.grid(alpha=0.2)
    detail_axis.set(xlim=(0.65, 0.9), title="Startup detail (dotted uses stored wall density)")
    detail_axis.set_ylim(detail_minimum - 0.5, detail_maximum + 0.5)
    clearance_axis.axhline(0, color="0.45", ls=":", lw=1)
    pressure_axis.legend(fontsize=8)
    fig.suptitle(f"SPHERIC Test 06, Re=100, L/dp={n}, CFL={info['cfl']:.2g}, fluid-only density diffusion")
    fig.savefig(out / "force_comparison.png", dpi=170)
    plt.close(fig)

    for filename, records in (("metrics.csv", rows), ("history.csv", history)):
        with (out / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    if study == "startup":
        for filename in ("frozen_forces.csv", "frozen_neighborhoods.csv"):
            shutil.copyfile(RUNS / filename, out / filename)
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", nargs="?", default="startup", choices=("startup", "full", "half-step", "coarse"))
    compare(parser.parse_args().study)
