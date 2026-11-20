"""Compare completed SPHERIC Test 06 force histories at three Reynolds numbers."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "planar_shifting_benchmark"


def read_history(path):
    rows = np.genfromtxt(path, delimiter=",", names=True)
    if len(rows) != 401 or rows["time"][0] != 0 or abs(rows["time"][-1] - 8) > 1e-3:
        raise ValueError(f"Incomplete benchmark history: {path}")
    for name in ("time", "cd_pressure", "cd_viscous"):
        if not np.isfinite(rows[name]).all():
            raise ValueError(f"Nonfinite {name}: {path}")
    return rows


def main():
    full, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, layout="constrained")
    startup, zoom = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True, layout="constrained")
    for col, re in enumerate((50, 100, 150)):
        native = read_history(OUT / "native" / f"re{re}" / f"Re{re:03d}_n160.csv")
        dual = read_history(OUT / "dual" / f"re{re}" / "dual.csv")
        ref = np.loadtxt(ROOT / "reference" / f"Force_Re{re:03d}.dat", skiprows=2)
        axes[0, col].set_title(f"Re = {re}")
        zoom[col].set_title(f"Re = {re}")
        for row, component in enumerate(("pressure", "viscous")):
            ax = axes[row, col]
            targets = (ax, zoom[col]) if row == 0 else (ax,)
            for panel in targets:
                for data, label, color in (
                    (native, "SPHExampleGPU", "#d55e00"),
                    (dual, "DualSPHysics 5.4", "#0072b2"),
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
    settings = "L/dp = 160 · CFL 0.20 · shifting cap 0.10 dp per component"
    full.suptitle(f"SPHERIC Test 06 full-run drag\n{settings}")
    startup.suptitle(f"SPHERIC Test 06 startup pressure drag\n{settings}")
    full.savefig(OUT / "drag_comparison.png", dpi=180)
    startup.savefig(OUT / "startup_pressure_drag.png", dpi=180)


if __name__ == "__main__":
    main()
