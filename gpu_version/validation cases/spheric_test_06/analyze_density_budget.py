"""Verify read-only replay parity and summarize per-ID continuity/DDT/floor budgets."""
import csv
import gzip
import shutil
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "density_budget"
MODELS = {"original": "coefh_refinement/re100", "fluid_only": "fluid_only_ddt", "zero": "zero_ddt_full"}


def main():
    summaries, parity = [], []
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    for column, (model, baseline) in enumerate(MODELS.items()):
        tag = f"{model}_replay_n80"
        directory = OUT / tag
        forces = np.genfromtxt(directory / "Re100_n080.csv", delimiter=",", names=True)
        reference = np.genfromtxt(ROOT / "results" / baseline / "Re100_n080.csv", delimiter=",", names=True)[:46]
        for field in forces.dtype.names:
            assert np.array_equal(forces[field], reference[field]), (model, field)
        source_hdf = (ROOT / "runs" / "coefh_refinement" / "re100" / "n80" / "MovingSquare.vtkhdf"
                      if model == "original" else ROOT / "runs" / baseline / "Re100_n080" / "MovingSquare.vtkhdf")
        # Compare actual particle state at every frame, not just force proxies.
        with h5py.File(ROOT / "runs" / "density_budget" / tag / "MovingSquare.vtkhdf") as a, h5py.File(source_hdf) as b:
            ar = a["VTKHDF/Particles"] if "VTKHDF/Particles" in a else a["VTKHDF"]
            br = b["VTKHDF/Particles"] if "VTKHDF/Particles" in b else b["VTKHDF"]
            for frame in range(46):
                sa, sb = int(ar["Steps/PointOffsets"][frame]), int(br["Steps/PointOffsets"][frame])
                count = int(ar["NumberOfPoints"][frame])
                assert count == int(br["NumberOfPoints"][frame])
                for field in ("Points", "PointData/Velocity", "PointData/Density", "PointData/GroupMarker"):
                    assert np.array_equal(ar[field][sa:sa+count], br[field][sb:sb+count]), (model, frame, field)
        parity.append(dict(model=model, frames=46, state_and_force_exact=True))
        trace = directory / "density_budget.csv"
        archived = ROOT / "runs" / "density_budget" / tag / "density_budget.csv.gz"
        if trace.is_file():
            with trace.open("rb") as source, gzip.open(archived, "wb") as target:
                shutil.copyfileobj(source, target)
        with gzip.open(archived, "rt") as source:
            d = np.genfromtxt(source, delimiter=",", names=True, dtype=None, encoding="utf-8")
        assert len(d) > 0
        assert np.all(np.abs(d["continuity_rate"] + d["diffusion_rate"] - d["consumed_rate"])
                      <= np.maximum(1e-5, 64*np.finfo(np.float32).eps*d["absolute_pair_rate"]))
        for stage in ("predictor", "corrector"):
            for ident in np.unique(d["id"]):
                particle = d[(d["id"] == ident) & (d["stage"] == stage)]
                assert len(np.unique(particle["iteration"])) == len(particle)
                for left, right in ((.3, .9), (.68, .76)):
                    p = particle[(particle["time"] >= left) & (particle["time"] < right)]
                    weight = p["dt"] / (2 if stage == "predictor" else 1)
                    summaries.append(dict(model=model, id=ident, type=int(p["type"][0]), stage=stage,
                        start=left, stop=right, samples=len(p),
                        density_start=float(p["base_density"][0]),
                        density_end=float(p["stored_density"][-1]),
                        continuity_rate_time_integral=float(np.sum(p["continuity_rate"]*weight)),
                        diffusion_rate_time_integral=float(np.sum(p["diffusion_rate"]*weight)),
                        floor_increment_sum=float(p["floor_increment"].sum()),
                        floor_events=int((p["floor_increment"] > 0).sum())))
                if particle["type"][0] == 3 and stage == "corrector":
                    label = f'ID {ident}'
                    axes[0, column].plot(particle["time"], particle["stored_density"] - 1000, lw=.7, label=label)
                    axes[1, column].plot(particle["time"], particle["continuity_rate"], lw=.6)
                    axes[2, column].plot(particle["time"], particle["diffusion_rate"], lw=.6)
        for row, title in enumerate(("Stored rear-wall density minus rho0 [kg/m3]",
                                     "Rear-wall continuity rate [kg/m3/s]", "Rear-wall DDT rate [kg/m3/s]")):
            axes[row, column].set(title=model if row == 0 else "", ylabel=title, xlabel="Time [s]")
            axes[row, column].axvline(.74 if model == "original" else .72, color="grey", ls=":", lw=1)
            axes[row, column].grid(alpha=.2)
        axes[0, column].legend(fontsize=6)
        if trace.is_file():
            trace.unlink()  # Named generated trace is archived under ignored runs/, not committed.
        print(f"Verified {model}: {len(d)} stage rows, exact force AND particle-state parity", flush=True)
    fig.suptitle("Read-only density-budget replay: fixed particle IDs, corrector stage\n"
                 "Continuity and DDT rates can strongly cancel; floor corrections are recorded separately")
    fig.tight_layout()
    fig.savefig(OUT / "rear_wall_budget.png", dpi=180)
    plt.close(fig)
    for name, rows in (("summary.csv", summaries), ("parity.csv", parity)):
        with (OUT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
