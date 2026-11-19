"""Compare saved startup histories with DBC and PlanarShifting held fixed."""
import csv
import tomllib

from audit_rear_wall_contact import read_study
from plot_coefh_refinement import integral, read_reference, read_table
from plot_regularized_mdbc_fields import CASE, RUNS, plot_fields, read_frame
import matplotlib.pyplot as plt
import numpy as np


OUT = CASE / "results" / "regularized_mdbc" / "dbc_diffusion_sweep"
PREFIX = "dbc_n80_half-step-planar"
MODELS = (
    ("-zero", "Zero diffusion"),
    ("-zero-gravity", "Zero-gravity linear"),
    ("", "Fluid-only linear"),
    ("-complex", "Complex"),
    ("-fourtakas-inner", "Fourtakas inner (DDT2)"),
    ("-fourtakas-full", "Fourtakas full (DDT3)"),
)
MATCHED = ("n", "reynolds", "dx", "h", "cfl", "sound_speed", "motion", "steps", "frames",
           "wall_layers", "background_pressure", "density_diffusion_coefficient", "timestep_criterion",
           "boundary_mode", "shifting_mode", "divergence_cleaning", "particles")


def write_rows(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference = read_reference(CASE / "reference" / "Force_Re100.dat")
    fig, grid = plt.subplots(3, 2, figsize=(12, 10), sharex=True, layout="constrained")
    metrics, contact = [], []
    panels = tuple((PREFIX + suffix, label) for suffix, label in MODELS)
    matched = None
    for (tag, label), color in zip(panels, plt.get_cmap("tab10").colors):
        directory = RUNS / tag
        meta = tomllib.loads((directory / "native.toml").read_text())
        settings = tuple(meta[key] for key in MATCHED)
        if matched is None:
            matched = settings
        assert settings == matched and meta["boundary_mode"] == "NoMDBC" and meta["shifting_mode"] == "PlanarShifting"
        records = read_study(tag, label, meta, include_mean_pressure=True)
        contact.extend(records)
        state = read_table(directory / "native_state.csv")
        force = read_table(directory / "native.csv")
        t, p, v = (force[key] for key in ("time", "cd_pressure", "cd_viscous"))
        np.testing.assert_allclose(t, np.linspace(0, 0.9, 46), atol=1e-12, rtol=0)
        np.testing.assert_array_equal(t, state["time"])
        supported = 20 - np.array([r["middle_without_fluid_neighbors"] for r in records])
        fill = np.array([r["median_middle_kernel_support"] for r in records])
        rear_pressure = np.array([r["maximum_middle_fluid_pressure_kpa"] for r in records])
        clearance = 1000 * state["minimum_clearance"]
        mean_pressure = np.array([r["mean_fluid_pressure_kpa"] for r in records])
        drift = mean_pressure - mean_pressure[0]
        late = t >= 0.7
        worst = np.flatnonzero(late)[np.argmin(p[late])]
        peak = np.flatnonzero(late)[np.nanargmax(rear_pressure[late])]
        ref_t = reference[:, 0]
        knots = np.unique(np.r_[t[t >= 0.1], ref_t[(ref_t > 0.1) & (ref_t < 0.9)]])
        errors = []
        for values, column in ((p, 1), (v, 2)):
            target = np.interp(knots, ref_t, reference[:, column])
            errors.append(np.sqrt(integral(knots, np.interp(knots, t, values) - target, squared=True) /
                                  integral(knots, target, squared=True)))
        metrics.append(dict(model=label, minimum_pressure_cd=p[worst], minimum_pressure_cd_time_s=t[worst],
                            reference_cd_at_min=np.interp(t[worst], ref_t, reference[:, 1]),
                            pressure_l2=errors[0], viscous_l2=errors[1], minimum_clearance_mm=clearance.min(),
                            final_drift_kpa=drift[-1], minimum_supported_rear_owners=supported[late].min(),
                            minimum_rear_kernel_support=fill[late].min(), supported_at_cd_min=supported[worst],
                            peak_rear_pressure_kpa=rear_pressure[peak], peak_rear_pressure_time_s=t[peak]))
        style = ":" if label == "Complex" else "-"
        for ax, values in zip(grid.flat, (p, v, fill, rear_pressure, clearance, drift)):
            ax.plot(t, values, color=color, ls=style, label=label)
    for ax, column in zip(grid[0], (1, 2)):
        ax.plot(reference[:, 0], reference[:, column], "k--", label="FD reference")
    grid[0, 0].axhline(0, color="0.5", lw=0.8)
    grid[2, 0].axhline(0, color="0.5", lw=0.8)
    for ax, label in zip(grid.flat, ("Pressure drag Cd", "Viscous drag Cd", "Median rear kernel support",
                                    "Peak rear fluid pressure [kPa]", "Minimum clearance [mm]",
                                    "Mean pressure drift [kPa]")):
        ax.set(xlim=(0, 0.9), ylabel=label)
        ax.grid(alpha=0.25)
    for ax in grid[-1]:
        ax.set_xlabel("Time [s]")
    grid[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("SPHERIC Test 06 · DBC + PlanarShifting · Re = 100 · L/dp = 80 · CFL = 0.10")
    fig.savefig(OUT / "comparison.png", dpi=180)
    plt.close(fig)
    write_rows("metrics.csv", metrics)
    write_rows("rear_wall_history.csv", contact)
    for time in (0.76, 0.88):
        frames = [read_frame(tag, time) for tag, _ in panels]
        for field in ("pressure", "speed"):
            plot_fields(frames, field, models=panels, out=OUT, subject=" · DBC + PlanarShifting", time=time)


if __name__ == "__main__":
    main()
