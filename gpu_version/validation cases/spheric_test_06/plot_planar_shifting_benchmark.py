"""Compare completed SPHERIC Test 06 force histories at three Reynolds numbers."""
from pathlib import Path
import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def read_history(path):
    rows = np.genfromtxt(path, delimiter=",", names=True)
    if len(rows) != 401 or rows["time"][0] != 0 or abs(rows["time"][-1] - 8) > 1e-3:
        raise ValueError(f"Incomplete benchmark history: {path}")
    for name in ("time", "cd_pressure", "cd_viscous"):
        if not np.isfinite(rows[name]).all():
            raise ValueError(f"Nonfinite {name}: {path}")
    return rows


def main(n, archived_dual=False, matched=False):
    tag = "planar_shifting_benchmark" + ("" if n == 160 else f"_n{n}")
    if matched:
        tag += "_ddt2"
    out = ROOT / "results" / tag
    full, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, layout="constrained")
    startup, zoom = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True, layout="constrained")
    for col, re in enumerate((50, 100, 150)):
        native = read_history(out / "native" / f"re{re}" / f"Re{re:03d}_n{n:03d}.csv")
        dual_dir = ROOT / "results" / "coefh_reynolds" / f"re{re}_dual" if archived_dual else out / "dual" / f"re{re}"
        dual = read_history(dual_dir / "dual.csv")
        lines = (ROOT / "reference" / f"Force_Re{re:03d}.dat").read_text().splitlines()
        ref = np.loadtxt([line for line in lines if line.lstrip()[:1].isdigit()])
        axes[0, col].set_title(f"Re = {re}")
        zoom[col].set_title(f"Re = {re}")
        for row, component in enumerate(("pressure", "viscous")):
            ax = axes[row, col]
            targets = (ax, zoom[col]) if row == 0 else (ax,)
            for panel in targets:
                for data, label, color in (
                    (native, "SPHExampleGPU", "#d55e00"),
                    (dual, "DualSPHysics 5.4 (archived)" if archived_dual else "DualSPHysics 5.4", "#0072b2"),
                ):
                    panel.plot(data["time"], data[f"cd_{component}"], color=color, label=label)
                panel.plot(ref[:, 0], ref[:, row + 1], color="0.2", ls="--", label="SPHERIC FD")
                panel.grid(alpha=.25)
            if col == 0:
                ax.set_ylabel(f"{component.capitalize()} drag Cd")
        axes[1, col].set(xlim=(0, 8), xlabel="Time (s)")
        zoom[col].set(xlim=(0, 1.2), xlabel="Time (s)")
        zoom[col].axhline(0, color="0.55", lw=.8)
    zoom[0].set_ylabel("Pressure drag Cd")
    axes[0, 0].legend(fontsize=8)
    zoom[0].legend(fontsize=8)
    settings = f"L/dp = {n} · CFL 0.20 · Julia shifting cap 0.10 dp per component"
    if matched:
        settings += " · Fourtakas DDT2"
    suffix = ""
    if archived_dual:
        settings += " · archived DualSPHysics coefh 1.2, shifting off"
        suffix = "_archived_dual"
    full.suptitle(f"SPHERIC Test 06 full-run drag\n{settings}")
    startup.suptitle(f"SPHERIC Test 06 startup pressure drag\n{settings}")
    full.savefig(out / f"drag_comparison{suffix}.png", dpi=180)
    startup.savefig(out / f"startup_pressure_drag{suffix}.png", dpi=180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, choices=(40, 80, 160), default=160)
    parser.add_argument("--archived-dual", action="store_true")
    parser.add_argument("--matched-numerics", action="store_true")
    args = parser.parse_args()
    if args.archived_dual and (args.resolution != 40 or args.matched_numerics):
        parser.error("The archived DualSPHysics baseline is at resolution 40")
    main(args.resolution, args.archived_dual, args.matched_numerics)
