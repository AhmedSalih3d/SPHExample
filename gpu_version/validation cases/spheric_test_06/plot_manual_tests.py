"""Compare the saved manual Test runs with the archived symplectic controls."""
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
OUT = CASE / "results/manual_tests"
RUNS = CASE / "runs/coefh"
ORIGINAL = CASE / "results/coefh"
COMPONENTS = ("cd_pressure", "cd_viscous")


def run_history(path):
    n, test = map(int, re.fullmatch(r"Re100_n(\d+)_Test(\d+)", path.name).groups())
    log = (path / "SimulationOutput.log").read_text(encoding="utf-8")
    final = re.search(r"^Part_0400\s+8\.000000\s+(\d+)\s+\d+\s+([\d.]+)", log, re.M)
    steps, seconds = int(final[1]), float(final[2])
    c0 = float(re.search(r"^  c₀: \S+ ([\d.]+)", log, re.M)[1])
    cfl = float(re.search(r"^  CFL: \S+ ([\d.]+)", log, re.M)[1])
    scheme = re.search(r"time_stepping = (\w+)", log)[1]
    shift = re.search(r"shifting_mode = (\w+)", log)[1]
    criterion = re.search(r"timestep_criterion = :(\w+)", log)[1]
    interaction = re.search(r"interaction_path = :(\w+)", log)[1]
    stem = (OUT / path.name) if test <= 4 else (ORIGINAL / f"Re100_n{n:03}")
    if test > 4:
        meta = tomllib.loads(stem.with_suffix(".toml").read_text())
        assert (meta["steps"], meta["shifting"], meta["time_stepping"]) == (steps, shift, scheme)
    force = read_table(stem.with_suffix(".csv"))
    state = read_table(stem.with_name(stem.name + "_state.csv"))
    np.testing.assert_allclose(force["time"], np.linspace(0, 8, 401), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(force["time"], state["time"])
    return force, dict(run=path.name, n=n, test=test, sound_speed=c0, cfl=cfl, time_stepping=scheme,
                       shifting=shift, timestep_criterion=criterion, interaction_path=interaction,
                       steps=steps, elapsed_at_final_frame_s=seconds,
                       minimum_clearance_mm=1000*state["minimum_clearance"].min())


def norm(t, values):
    return np.sqrt(integral(t, values, squared=True))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    reference = read_reference(CASE / "reference/Force_Re100.dat")
    runs, rows = {}, []
    for path in sorted(RUNS.glob("Re100_n*_Test*")):
        force, info = run_history(path)
        n, test = info["n"], info["test"]
        runs[n, test] = force
        t = force["time"]
        compare = np.union1d(t[t >= .1], reference[(reference[:, 0] > .1) & (reference[:, 0] < 8), 0])
        for col, key in enumerate(COMPONENTS, 1):
            ref = np.interp(compare, reference[:, 0], reference[:, col])
            info[key + "_l2"] = norm(compare, np.interp(compare, t, force[key]) - ref) / norm(compare, ref)
        startup = (t >= .65) & (t <= .95)
        i = np.flatnonzero(startup)[force["cd_pressure"][startup].argmin()]
        info.update(startup_min_pressure_cd=force["cd_pressure"][i], startup_min_time_s=t[i])
        rows.append(info)
    assert len(rows) == 9
    write_csv("metrics.csv", rows)
    scenarios = (
        ("n040_tests", [(40, i) for i in range(1, 6)]),
        ("single_neighbor_refinement", [(10, 7), (20, 6), (40, 5), (80, 8), (160, 8)]),
    )
    for name, tags in scenarios:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        for i, key in enumerate(COMPONENTS):
            for ax, xlim in zip(axes[i], ((0, 8), (.35, 1.05))):
                ax.plot(reference[:, 0], reference[:, i+1], "k--", label="FD reference")
                for n, test in tags:
                    f = runs[n, test]
                    ax.plot(f["time"], f[key], lw=1.1, label=f"n{n} Test {test}")
                ax.set(xlim=xlim, xlabel="Time [s]", ylabel=("Pressure drag Cd" if i == 0 else "Viscous drag Cd"))
                ax.grid(alpha=.22)
        axes[0, 0].legend(fontsize=8, ncol=2)
        fig.suptitle("Manual moving-square tests · Re = 100 · coefh = 2.0")
        fig.savefig(OUT / (name + ".png"), dpi=170)
        plt.close(fig)
    rows = []
    for sequence in (((20,6), (40,5), (80,8)), ((40,5), (80,8), (160,8))):
        a,b,c = (runs[tag] for tag in sequence)
        for start, stop in ((.1,8.), (.1,.9), (1.,8.)):
            selected = (c["time"] >= start) & (c["time"] <= stop)
            t = c["time"][selected]
            for key in COMPONENTS:
                p,q,r = (data[key][selected] for data in (a,b,c))
                rows.append(dict(resolutions="/".join(str(tag[0]) for tag in sequence),
                                 start_s=start, end_s=stop, component=key,
                                 coarse_difference=norm(t,p-q)/norm(t,r), fine_difference=norm(t,q-r)/norm(t,r)))
    write_csv("resolution_differences.csv", rows)
    rows = []
    for n, test in ((40, 5), (80, 8), (160, 8)):
        folder = "manual_coefh2_matrix/dbc/zero-gravity-linear" if n < 160 else "matched_timestep_refinement"
        symplectic = read_table(CASE / "results" / folder / f"Re100_n{n:03}.csv")
        single = runs[n, test]
        np.testing.assert_array_equal(single["time"], symplectic["time"])
        select = single["time"] >= .1
        t = single["time"][select]
        for key in COMPONENTS:
            ref = symplectic[key][select]
            rows.append(dict(n=n, component=key, relative_difference=norm(t, single[key][select] - ref)/norm(t, ref)))
    write_csv("integrator_differences.csv", rows)


if __name__ == "__main__":
    main()
