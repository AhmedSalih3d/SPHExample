"""Plot paired whole-run timings; run after still_wedge_timestep.jl."""
import csv
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = Path(__file__).resolve().parents[1] / "validation cases/still_wedge_timestep/results"
with (results / "timings.csv").open(newline="") as source:
    rows = list(csv.DictReader(source))
pairs = {}
for row in rows:
    pairs.setdefault(int(row["repetition"]), {})[row["criterion"]] = float(row["wall_seconds"])
ratios = [pair["pairwise"] / pair["legacy"] for pair in pairs.values()]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout="constrained")
for repetition, pair in pairs.items():
    axes[0].plot([0, 1], [pair["legacy"], pair["pairwise"]], "o-", label=f"Pair {repetition}")
axes[0].set(xticks=[0, 1], xticklabels=["Legacy", "New default"], ylabel="Complete run wall time (s)")
axes[0].legend(frameon=False, fontsize=8)
axes[0].grid(axis="y", alpha=0.25)
axes[1].bar(list(pairs), [100 * (ratio - 1) for ratio in ratios], color="#4477aa")
axes[1].set(xlabel="Alternating pair", ylabel="Runtime increase (%)", xticks=list(pairs))
axes[1].axhline(100 * (statistics.median(ratios) - 1), color="#cc6677", linestyle="--", label="Median paired increase")
axes[1].legend(frameon=False, fontsize=8)
fig.suptitle("StillWedgeMDBC.jl: 4 s simulated, 3,027 particles, RTX 5080\nActual example boundary: NoMDBC; output enabled; kernels warmed")
fig.savefig(results / "timings.png", dpi=180)
for criterion in ("legacy", "pairwise"):
    data = [row for row in rows if row["criterion"] == criterion]
    print(criterion, "wall median", statistics.median(float(r["wall_seconds"]) for r in data),
          "solver median", statistics.median(float(r["solver_seconds"]) for r in data))
print("paired ratio median/min/max", statistics.median(ratios), min(ratios), max(ratios))
