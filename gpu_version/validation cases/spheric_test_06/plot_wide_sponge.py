"""Plot raw DBC/integrated-motion controls; expanded tanks change the benchmark."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/wide_sponge"
reference = np.loadtxt(ROOT / "reference/Force_Re100.dat", skiprows=2)
variants = [
    ("original_tank", "Original tank · no sponge", "#757575", "--"),
    ("expanded_tank", "Expanded tank · no sponge", "#0072b2", "-"),
    ("expanded_sponge", "Expanded tank · 2 m sponge, 100 s⁻¹", "#d55e00", "-"),
]
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for name, label, color, style in variants:
    data = np.genfromtxt(OUT / name / "Re100_n040.csv", delimiter=",", names=True)
    for column, component in enumerate(("pressure", "viscous")):
        for ax in axes[:, column]:
            ax.plot(data["time"], data[f"cd_{component}"], color=color, ls=style, lw=1.2, label=label)
for column, component in enumerate(("pressure", "viscous")):
    for ax in axes[:, column]:
        ax.plot(reference[:, 0], reference[:, column+1], color="black", lw=1.4,
                label="Official FD · original tank only")
        ax.set(xlabel="Time [s]", ylabel=f"{component.capitalize()} drag coefficient")
        ax.grid(alpha=.2)
    axes[0, column].set(xlim=(0, 8), title=f"{component.capitalize()} drag · full history")
    axes[1, column].set(xlim=(3, 7), title=f"{component.capitalize()} drag · later interval")
    # Choose limits from this interval so startup extrema do not flatten the zoom.
    values = [reference[(reference[:,0]>=3) & (reference[:,0]<=7),column+1]]
    for name, *_ in variants:
        d = np.genfromtxt(OUT/name/"Re100_n040.csv",delimiter=",",names=True)
        values.append(d[f"cd_{component}"][(d["time"]>=3)&(d["time"]<=7)])
    lo, hi = min(np.min(v) for v in values), max(np.max(v) for v in values)
    margin = .08*(hi-lo)
    axes[1,column].set_ylim(lo-margin,hi+margin)
handles, labels = axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc="lower center",ncol=2,fontsize=9,bbox_to_anchor=(.5,.005))
fig.suptitle("Wider sponge diagnostic · DBC · integrated-velocity motion · Re = 100\n"
             "c₀ = 40 m/s · Δp = 0.025 m · cleaning off · expanded domain changes the physical problem",fontsize=12)
fig.tight_layout(rect=(0,.09,1,.92))
fig.savefig(OUT/"drag_comparison.png",dpi=180)
fig.savefig(OUT/"drag_comparison.pdf")
print(OUT/"drag_comparison.png")
