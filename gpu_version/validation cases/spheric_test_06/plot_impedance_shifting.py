"""Compare matched DBCPlusImpedance startup runs with and without PlanarShifting."""
import csv
import tomllib

from audit_rear_wall_contact import CASE, RUNS, read_study
from plot_coefh_refinement import integral, read_reference, read_table
from plot_regularized_mdbc_fields import read_frame, plot_fields
from plot_simple_mdbc_zero_ddt import mean_fluid_pressure
import matplotlib.pyplot as plt
import numpy as np


OUT = CASE / "results" / "regularized_mdbc" / "impedance_shifting"
MODELS = (("impedance_n80_half-step", "NoShifting"),
          ("impedance_n80_half-step-planar", "PlanarShifting"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reference = read_reference(CASE / "reference" / "Force_Re100.dat")
    metas = [tomllib.loads((RUNS / tag / "native.toml").read_text()) for tag, _ in MODELS]
    differences = {key for key, value in metas[0].items() if metas[1][key] != value}
    outcomes = {"shifting_mode", "density_relative_max", "mach_max", "minimum_clearance",
                "motion_error_max", "velocity_error_max", "readbacks", "rebuilds", "solver_seconds", "wall_seconds"}
    assert differences <= outcomes, differences - outcomes

    fig, grid = plt.subplots(2, 3, figsize=(13.5, 6.7), sharex=True, layout="constrained")
    axs = grid.ravel()
    records, metrics = [], []
    for (tag, shift), meta, color in zip(MODELS, metas, ("#0072b2", "#d55e00")):
        assert meta["boundary_mode"] == "DBCPlusImpedance" and meta["shifting_mode"] == shift
        folder = RUNS / tag
        forces = read_table(folder / "native.csv")
        state = read_table(folder / "native_state.csv")
        time, cp, cv = forces["time"], forces["cd_pressure"], forces["cd_viscous"]
        np.testing.assert_array_equal(time, state["time"])
        np.testing.assert_allclose(time, np.linspace(0, 0.9, 46), atol=1e-12, rtol=0)
        assert np.all(state["fluid_particles"] == 313600) and np.all(state["square_particles"] == 6400)
        mean_pressure = mean_fluid_pressure(folder / "state" / "MovingSquare.vtkhdf", time) / 1000
        drift = mean_pressure - mean_pressure[0]
        clearance = state["minimum_clearance"] * 1000
        contact = read_study(tag, shift, meta)
        for row, pressure, gap in zip(contact, drift, clearance):
            row.update(mean_fluid_pressure_drift_kpa=pressure, minimum_clearance_mm=gap)
        records.extend(contact)
        distance = [row["median_middle_face_distance_dp"] for row in contact]
        unsupported = [row["middle_without_fluid_neighbors"] for row in contact]

        ref_t = reference[:, 0]
        knots = np.unique(np.r_[0.1, 0.9, time[(time > 0.1) & (time < 0.9)],
                               ref_t[(ref_t > 0.1) & (ref_t < 0.9)]])

        def relative_l2(samples, column):
            exact = np.interp(knots, ref_t, reference[:, column])
            error = np.interp(knots, time, samples) - exact
            return np.sqrt(integral(knots, error, squared=True) / integral(knots, exact, squared=True))

        startup = np.flatnonzero(time >= 0.7)
        lowest = startup[cp[startup].argmin()]
        pressure_088 = cp[np.flatnonzero(np.isclose(time, 0.88, rtol=0, atol=1e-12)).item()]
        metrics.append(dict(shifting=shift, pressure_l2=relative_l2(cp, 1), viscous_l2=relative_l2(cv, 2),
                            minimum_pressure_cd=cp[lowest], minimum_pressure_time_s=time[lowest],
                            pressure_cd_at_088=pressure_088, minimum_clearance_mm=clearance.min(),
                            final_pressure_drift_kpa=drift[-1], maximum_central_unsupported=max(unsupported),
                            peak_central_stencil_fluid_pressure_kpa=np.nanmax(
                                [row["maximum_middle_fluid_pressure_kpa"] for row in contact])))

        for ax, series in zip(axs, (cp, cv, drift, clearance, distance, unsupported)):
            ax.plot(time, series, color=color, label=shift, lw=1.4)

    for ax, column in ((axs[0], 1), (axs[1], 2)):
        ax.plot(reference[:, 0], reference[:, column], "k--", lw=1, label="FD reference")
    for ax, label in zip(axs, ("Pressure drag Cd", "Viscous drag Cd", "Mean fluid pressure drift [kPa]",
                              "Minimum clearance [mm]", "Median central rear gap [dp]",
                              "Central wall particles without fluid [of 20]")):
        ax.set(xlim=(0, 0.9), ylabel=label, xlabel="Time [s]")
        ax.grid(alpha=0.2)
        ax.axvline(0.88, color="0.5", ls=":", lw=0.8)
    axs[0].legend(fontsize=8)
    axs[3].axhline(0, color="0.5", lw=0.8)
    axs[5].set_ylim(-0.5, 20.5)
    axs[5].set_yticks([0, 5, 10, 15, 20])
    fig.suptitle("DBCPlusImpedance with and without shifting · Re = 100 · L/dp = 80 · CFL = 0.10")
    fig.savefig(OUT / "comparison.png", dpi=185)
    plt.close(fig)

    for name, rows in (("metrics.csv", metrics), ("history.csv", records)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    times = tuple(dict.fromkeys((metrics[1]["minimum_pressure_time_s"], 0.88)))
    panels = tuple((tag, f"{shift} · {time:.2f} s") for time in times for tag, shift in MODELS)
    snapshots = [read_frame(tag, time) for time in times for tag, _ in MODELS]
    for field in ("speed", "pressure"):
        plot_fields(snapshots, field, models=panels, out=OUT, subject=" · DBCPlusImpedance", time=None)


if __name__ == "__main__":
    main()
