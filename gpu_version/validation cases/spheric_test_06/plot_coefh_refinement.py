"""Verify and render raw coefh=1.2 refinement results without filtering either solver or reference."""
import csv
import hashlib
import math
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "coefh_refinement"
WORK = ROOT / "runs" / "coefh_refinement"
COLORS = {20: "#d55e00", 40: "#009e73", 80: "#0072b2"}


def read_table(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def read_reference(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0] not in "+-.0123456789":
            if rows:
                raise ValueError(f"Unexpected reference text in {path}")
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid reference sample in {path}")
        rows.append(values)
    result = np.asarray(rows)
    if len(result) < 2 or not np.all(np.diff(result[:, 0]) > 0):
        raise ValueError(f"Invalid reference time grid in {path}")
    return result


def integral(t, values, squared=False):
    if squared:
        return np.sum(np.diff(t) * (values[:-1] ** 2 + values[:-1] * values[1:] + values[1:] ** 2)) / 3
    return np.sum(np.diff(t) * (values[:-1] + values[1:])) / 2


def average(t, values, left=3.0, right=7.0):
    knots = np.r_[left, t[(t > left) & (t < right)], right]
    return integral(knots, np.interp(knots, t, values)) / (right - left)


def relative_error(t, values, rt, target, residual=True):
    knots = np.unique(np.r_[0.1, 8.0, t[(t > 0.1) & (t < 8)], rt[(rt > 0.1) & (rt < 8)]])
    prediction = np.interp(knots, t, values)
    reference = np.interp(knots, rt, target)
    numerator = prediction - reference if residual else prediction
    return np.sqrt(integral(knots, numerator, squared=True) / integral(knots, reference, squared=True))


def fields(data):
    return {"pressure": data["cd_pressure"], "viscous": data["cd_viscous"],
            "total": data["cd_pressure"] + data["cd_viscous"]}


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / (name + ".png"), dpi=180)
    fig.savefig(OUT / (name + ".pdf"))
    plt.close(fig)


def main():
    with (OUT / "provenance.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            source = ROOT / row["source"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Source history changed: {source}")
    summary = list(csv.DictReader((OUT / "summary.csv").open(newline="", encoding="utf-8")))
    changes = list(csv.DictReader((OUT / "resolution_changes.csv").open(newline="", encoding="utf-8")))
    data, states, metadata, references = {}, {}, {}, {}
    independent = []
    for re in (50, 100, 150):
        reference = read_reference(ROOT / "reference" / f"Force_Re{re:03d}.dat")
        references[re] = reference
        targets = {"pressure": reference[:, 1], "viscous": reference[:, 2],
                   "total": reference[:, 1] + reference[:, 2]}
        for n in (20, 40, 80):
            stem = OUT / f"re{re}" / f"Re{re:03d}_n{n:03d}"
            if not stem.with_suffix(".toml").is_file():
                print(f"Pending: Re={re}, L/dp={n}")
                continue
            d, s = read_table(stem.with_suffix(".csv")), read_table(Path(str(stem) + "_state.csv"))
            m = tomllib.loads(stem.with_suffix(".toml").read_text(encoding="utf-8"))
            assert len(d) == len(s) == m["frames"] == 401
            assert d["time"][0] == 0 and d["time"][-1] == 8
            assert np.array_equal(d["time"], s["time"])
            assert all(np.isfinite(d[key]).all() for key in d.dtype.names)
            assert all(np.isfinite(s[key]).all() for key in s.dtype.names)
            assert np.all(s["fluid_particles"] == 49 * n * n)
            assert np.all(s["square_particles"] == n * n)
            data[re, n], states[re, n], metadata[re, n] = d, s, m
            for component, values in fields(d).items():
                error = relative_error(d["time"], values, reference[:, 0], targets[component])
                row = next(r for r in summary if int(r["reynolds"]) == re and int(r["n"]) == n
                           and r["component"] == component)
                assert math.isclose(error, float(row["raw_nrmse"]), abs_tol=1e-10, rel_tol=0)
                assert math.isclose(average(d["time"], values), float(row["mean_3_7"]),
                                    abs_tol=1e-10, rel_tol=0)
                independent.append(dict(reynolds=re, n=n, component=component, raw_nrmse=error))
        for row in (r for r in changes if int(r["reynolds"]) == re):
            a, b = data[re, int(row["coarse"])], data[re, int(row["fine"])]
            difference = fields(a)[row["component"]] - fields(b)[row["component"]]
            value = relative_error(a["time"], difference, reference[:, 0],
                                   targets[row["component"]], residual=False)
            assert math.isclose(value, float(row["relative_to_reference"]), abs_tol=1e-10, rel_tol=0)

    fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
    for column, re in enumerate((50, 100, 150)):
        ref = references[re]
        for row, component in enumerate(("pressure", "viscous", "total")):
            ax = axes[row, column]
            for n in (20, 40, 80):
                if (re, n) not in data:
                    continue
                d = data[re, n]
                ax.plot(d["time"], fields(d)[component], color=COLORS[n], lw=1, label=f"L/dp={n}")
            target = ref[:, 1] + ref[:, 2] if component == "total" else ref[:, 1 if component == "pressure" else 2]
            ax.plot(ref[:, 0], target, color="black", lw=1.2, label="Official FD samples")
            ax.set(xlim=(0, 8), ylabel=f"{component.capitalize()} Cd", xlabel="Time [s]")
            ax.grid(alpha=0.2)
            if row == 0:
                available = ", ".join(str(n) for n in (20, 40, 80) if (re, n) in data)
                ax.set_title(f"Re = {re}; available L/dp = {available or 'none'}")
    for ax in axes[0]:
        ax.legend(fontsize=8)
    fig.suptitle("Raw resolution comparison: coefh=1.2, c0=160 m/s, CFL=0.2\n"
                 "Integrated motion, original closed tank, DBC; no force filtering")
    save_figure(fig, "raw_drag")

    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    for column, re in enumerate((50, 100, 150)):
        for n in (20, 40, 80):
            if (re, n) not in data:
                continue
            d, s = data[re, n], states[re, n]
            axes[0, column].plot(d["time"], d["cd_pressure"], color=COLORS[n], lw=1, label=f"L/dp={n}")
            axes[1, column].plot(s["time"], np.maximum(0, -s["minimum_clearance"]) * n,
                                 color=COLORS[n], lw=1)
        ref = references[re]
        axes[0, column].plot(ref[:, 0], ref[:, 1], color="black", lw=1.2, label="Official FD")
        axes[0, column].set(xlim=(1.5, 6.5), title=f"Re = {re}", ylabel="Raw pressure Cd")
        axes[1, column].set(xlim=(0, 8), ylabel="Maximum penetration / dp", xlabel="Time [s]")
        for ax in axes[:, column]:
            ax.grid(alpha=0.2)
    for ax in axes[0]:
        ax.legend(fontsize=8)
    fig.suptitle("Central pressure and full-history penetration; no smoothing")
    save_figure(fig, "pressure_and_penetration")

    cfl = read_table(OUT / "cfl01" / "Re100_n040.csv")
    a = data[100, 40]
    ref = references[100]
    cfl_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for component, ax, index in zip(("pressure", "viscous"), axes, (1, 2)):
        for label, d in (("CFL=0.2", a), ("CFL=0.1 (archived control)", cfl)):
            ax.plot(d["time"], fields(d)[component], lw=1, label=label)
        ax.plot(ref[:, 0], ref[:, index], color="black", lw=1, label="Official FD")
        ax.set(xlim=(0, 8), xlabel="Time [s]", ylabel=f"{component.capitalize()} Cd")
        ax.grid(alpha=0.2)
        cfl_rows.append(dict(component=component,
            cfl02_error=relative_error(a["time"], fields(a)[component], ref[:, 0], ref[:, index]),
            cfl01_error=relative_error(cfl["time"], fields(cfl)[component], ref[:, 0], ref[:, index]),
            relative_change=relative_error(a["time"], fields(a)[component] - fields(cfl)[component],
                                           ref[:, 0], ref[:, index], residual=False)))
    axes[0].legend(fontsize=8)
    fig.suptitle("Timestep sensitivity at Re=100, L/dp=40; raw forces")
    save_figure(fig, "cfl_control")
    for name, rows in (("independent_metrics.csv", independent), ("cfl_sensitivity.csv", cfl_rows)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    for re in (50, 100, 150):
        plot_directory = WORK / f"re{re}" / "paraview_force_plot"
        plot_directory.mkdir(parents=True, exist_ok=True)
        available_states = [
            WORK / f"re{re}" / f"n{n}" / "MovingSquare_SingleVTKHDFStateFile.py"
            for n in (80, 20) if (re, n) in data
        ]
        state = next((path for path in available_states if path.is_file()), None)
        if not any((re, n) in data for n in (20, 40, 80)):
            continue
        simulation_argument = f"Path({state.as_posix()!r})" if state else "None"
        script = f'''from pathlib import Path
from types import SimpleNamespace
import runpy

args = SimpleNamespace(
    results=Path({(OUT / f"re{re}").as_posix()!r}),
    output=Path({plot_directory.as_posix()!r}),
    n=None, component="both", simulation_state={simulation_argument},
)
plotting = runpy.run_path({(ROOT / "paraview_force_plot.py").as_posix()!r})
plotting["make_force_plot"]({re}, args)
'''
        (plot_directory / f"ForceComparison_Re{re:03d}.py").write_text(script, encoding="utf-8")
    complete = len(data) == 9
    report = [
        "# Raw coefh=1.2 refinement results", "",
        f"Status: {'all nine resolution histories complete' if complete else 'partial study; pending runs are not plotted'}.",
        "", "Original 10 by 5 m closed tank, c0=160 m/s, coefh=1.2, integrated motion,",
        "DBC, CFL=0.2, delta=0.1, five wall layers; no background pressure,",
        "cleaning, sponge, shifting or hold. All forces and supplied reference",
        "samples are plotted without acoustic filtering, scaling or time shifts.",
        "The reference authors' pre-publication processing is not established.",
        "", "## Completed resolution histories", "",
        "| Re | L/dp | Pressure raw error | Viscous raw error | Total raw error | Penetration [mm] | Penetration / dp |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for re in (50, 100, 150):
        for n in (20, 40, 80):
            selected = {row["component"]: row for row in summary
                        if int(row["reynolds"]) == re and int(row["n"]) == n}
            if not selected:
                report.append(f"| {re} | {n} | pending | pending | pending | - | - |")
                continue
            p, v, total = (selected[key] for key in ("pressure", "viscous", "total"))
            report.append(f'| {re} | {n} | {100*float(p["raw_nrmse"]):.2f}% | '
                          f'{100*float(v["raw_nrmse"]):.2f}% | {100*float(total["raw_nrmse"]):.2f}% | '
                          f'{float(p["penetration_mm"]):.3f} | {float(p["penetration_dp"]):.3f} |')
    report += ["", "## Changes between resolutions", "",
               "Raw L2 differences over 0.1-8 s, normalized by the official reference L2 norm.",
               "These are changes between simulations, not their reference errors.", "",
               "| Re | Coarse -> fine | Component | Relative change |",
               "|---:|:---|:---|---:|"]
    for row in changes:
        report.append(f'| {row["reynolds"]} | {row["coarse"]} -> {row["fine"]} | '
                      f'{row["component"]} | {100*float(row["relative_to_reference"]):.2f}% |')
    report += ["", "## Timestep sensitivity", "",
               "Archived Re=100, L/dp=40 control, same h and motion. Halving CFL",
               "does not establish temporal convergence from two settings alone.", "",
               "| Component | CFL=0.2 error | CFL=0.1 error | Change between histories |",
               "|:---|---:|---:|---:|"]
    for row in cfl_rows:
        report.append(f'| {row["component"]} | {100*row["cfl02_error"]:.2f}% | '
                      f'{100*row["cfl01_error"]:.2f}% | {100*row["relative_change"]:.2f}% |')
    if all((100, n) in data for n in (20, 40, 80)):
        report += ["", "## Re=100 assessment", "",
                   "The finest history has less central pressure noise, but it does not",
                   "improve the full raw force agreement. In particular, the viscous mean",
                   "crosses the reference as resolution increases. The n=40 result must",
                   "not be declared a converged optimum because it agrees best in one component.",
                   "", "| L/dp | Pressure mean bias, 3-7 s | Viscous mean bias, 3-7 s | Pressure noise RMS |",
                   "|---:|---:|---:|---:|"]
        for n in (20, 40, 80):
            selected = {row["component"]: row for row in summary
                        if int(row["reynolds"]) == 100 and int(row["n"]) == n}
            p, v = selected["pressure"], selected["viscous"]
            report.append(f'| {n} | {100*float(p["mean_bias_3_7"]):+.2f}% | '
                          f'{100*float(v["mean_bias_3_7"]):+.2f}% | {float(p["pressure_noise"]):.5f} |')
        report += ["", "Pressure changes between the two finest histories are larger than",
                   "between the two coarser histories in this dataset. Penetration decreases",
                   "in millimetres but is not monotonically decreasing as a fraction of dp.",
                   "Together with the CFL sensitivity, this does not establish an asymptotic",
                   "space/time convergence regime. Finish diagnosing the Re=100 startup,",
                   "wall traction and timestep effects before spending on the remaining",
                   "Re=50/150 fine runs. Those resolutions remain explicitly pending.", ""]
    report += ["", "## Interpretation and provenance", "",
               "Refinement changes the spatial stencil scale and the acoustic timestep together.",
               "Agreement between resolutions does not by itself establish accuracy against",
               "the incompressible reference. Fixed c0 and fixed h/dp do not demonstrate",
               "the incompressible or SPH consistency limits. No Richardson order or",
               "extrapolated solution is claimed without an asymptotic convergence regime.",
               "Saved-frame penetration remains a separate failure of exact impermeability.",
               "", "Raw errors integrate squared piecewise-linear residuals exactly over",
               "0.1-8 s. Julia and independent NumPy integrals agree within 1e-10.",
               "Startup peaks/times are sampled at the 0.02 s output interval, not continuous maxima.",
               "The central-window noise proxy is for diagnosis only; it is not applied",
               "to any displayed force curve. All completed state frames are finite and",
               "retain their expected fluid/body counts. Per-case metadata records density,",
               "motion, frozen-particle and Mach checks.",
               "", "The n=40 and CFL=0.1 controls are reused without changing their",
               "original metadata. `provenance.csv` records sources and force SHA-256",
               "checksums. Hardware differs between archived and fresh runs; no timing",
               "or speedup claim is made. Original user results are not overwritten.",
               "", "Load the generated Python states under",
               "`../../runs/coefh_refinement/re<Re>/paraview_force_plot/` in ParaView",
               "to see all completed raw resolutions alongside the reference and the",
               "completed fresh particle history (where available), with animated markers.",
               "", "![Raw forces](raw_drag.png)", "",
               "![Pressure and penetration](pressure_and_penetration.png)", "",
               "![CFL control](cfl_control.png)", ""]
    (OUT / "README.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Verified {len(independent)} component errors and {len(changes)} resolution differences independently.")
    print("Saved raw figures, CFL comparison and three all-resolution ParaView entry points.")
    print("CFL sensitivity:", cfl_rows)


if __name__ == "__main__":
    main()
