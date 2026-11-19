"""Compare time and space refinements of the Re=100 moving-square case."""
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
TAG = "matched_timestep_refinement"
RESULTS = CASE / "results" / TAG
RUNS = CASE / "runs" / TAG
BASELINE = CASE / "results/manual_coefh2_matrix/dbc/zero-gravity-linear"
KEYS = ("cd_pressure", "cd_viscous")
PHYSICS = ("reynolds", "gravity", "rho0", "sound_speed", "nu", "viscosity_model",
           "boundary_mode", "motion_kind", "shifting", "density_diffusion",
           "density_diffusion_model", "time_stepping", "background_pressure", "wall_layers")


def read_matched():
    histories = {}
    grids = {}
    metadata = {}
    for n in (40, 80, 160):
        name = f"Re100_n{n:03}"
        record = RESULTS / f"{name}.toml"
        if not record.exists():
            continue
        meta = tomllib.loads(record.read_text())
        history = read_table(record.with_suffix(".csv"))
        np.testing.assert_allclose(history["time"], np.linspace(0, 8, 401), atol=1e-12, rtol=0)
        log = (RUNS / name / "SimulationOutput.log").read_text(encoding="utf-8", errors="replace")
        steps = [int(s) for s in re.findall(r"^Part_\d+\s+\d+\.\d+\s+(\d+)\s", log, re.M)]
        assert len(steps) == 401 and steps[0] == 0 and steps[-1] == meta["steps"]
        assert all(np.isfinite(history[key]).all() for key in KEYS)
        histories[n], grids[n], metadata[n] = history, steps, meta
    return histories, metadata, grids


def norm(time, values):
    return np.sqrt(integral(time, values, squared=True))


def main():
    histories, metadata, grids = read_matched()
    if not histories:
        raise RuntimeError("No completed matched runs")
    baseline = read_table(BASELINE / "Re100_n080.csv")
    baseline_meta = tomllib.loads((BASELINE / "Re100_n080.toml").read_text())
    for n, meta in metadata.items():
        assert all(meta[key] == baseline_meta[key] for key in PHYSICS), f"Physics settings differ at n={n}"
    reference = read_reference(CASE / "reference/Force_Re100.dat")
    fine_record = RESULTS / "fine_time/Re100_n160.toml"
    fine_time = None
    if fine_record.exists():
        fine_meta = tomllib.loads(fine_record.read_text())
        assert all(fine_meta[key] == baseline_meta[key] for key in PHYSICS)
        assert fine_meta["dx"] == 1/160 and np.isclose(fine_meta["cfl"], 0.1)
        fine_time = read_table(fine_record.with_suffix(".csv"))
        np.testing.assert_allclose(fine_time["time"], np.linspace(0, .9, 46), atol=1e-12, rtol=0)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for column, (ax, key) in enumerate(zip(axes, KEYS), start=1):
        ax.plot(reference[:, 0], reference[:, column], "k--", label="FD reference")
        ax.plot(baseline["time"], baseline[key], color="0.6", lw=1.0, label="80, CFL 0.20")
        for n, history in histories.items():
            ax.plot(history["time"], history[key], lw=1.1, label=f"{n}, CFL {metadata[n]['cfl']:.2f}")
        if fine_time is not None:
            ax.plot(fine_time["time"], fine_time[key], ":", lw=1.6, label="160, CFL 0.10")
        ax.set(ylabel=("Pressure drag Cd" if column == 1 else "Viscous drag Cd"), xlim=(0, 8))
        ax.grid(alpha=.22)
        ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("SPHERIC Test 06 · DBC · PlanarShifting · Re = 100 · coefh = 2.0")
    fig.savefig(RESULTS / "drag_histories.png", dpi=175)
    plt.close(fig)

    rows = []
    if 80 in histories:
        time = histories[80]["time"]
        np.testing.assert_array_equal(time, baseline["time"])
        for start, end in ((.1, .9), (.1, 8), (1., 8)):
            select = (time >= start) & (time <= end)
            t = time[select]
            for key in KEYS:
                fine80 = histories[80][key][select]
                denominator = norm(t, histories.get(160, histories[80])[key][select])
                temporal = norm(t, fine80 - baseline[key][select])
                coarse = norm(t, histories[40][key][select] - fine80) if 40 in histories else np.nan
                fine = norm(t, fine80 - histories[160][key][select]) if 160 in histories else np.nan
                temporal160 = ""
                if fine_time is not None and 160 in histories and end <= .9:
                    lower_cfl = fine_time["time"] >= start
                    np.testing.assert_array_equal(t, fine_time["time"][lower_cfl])
                    temporal160 = norm(t, histories[160][key][select] - fine_time[key][lower_cfl]) / denominator
                rows.append(dict(start_s=start, end_s=end, component=key,
                                 normalization_n=160 if 160 in histories else 80,
                                 temporal_n80=temporal / denominator, temporal_n160=temporal160,
                                 spatial_40_to_80=coarse / denominator,
                                 spatial_80_to_160=fine / denominator, apparent_order=np.log2(coarse / fine)))
        with (RESULTS / "differences.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for n, meta in metadata.items():
        interval_steps = np.diff(grids[n])
        print(f"L/dp={n}: {meta['steps']} steps, {interval_steps.min()}–{interval_steps.max()} steps per frame, "
              f"{8e6/meta['steps']:.3f} µs mean timestep, {1000*meta['minimum_clearance']:.3f} mm minimum clearance")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
