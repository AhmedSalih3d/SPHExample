"""Compare the two Float 1 down studies and count fluid centers inside the float."""
import csv
import tomllib
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results/no_penetration_study"
CASE = "float1_down"
VARIANTS = (
    ("UpdatedMDBC", "floating_updated_study/UpdatedMDBC", "#168b86"),
    ("UpdatedMDBCNoPenetration", "no_penetration_study", "#d55e00"),
)


def crossings(folder, run):
    body = np.genfromtxt(folder / f"{CASE}_body.csv", delimiter=",", names=True,
                        deletechars="", replace_space="")
    with h5py.File(run / "FreeHeave.vtkhdf") as file:
        particles = file["VTKHDF/Particles"]
        times = particles["Steps/Values"][:]
        np.testing.assert_allclose(times, body["Time"], atol=1e-9, rtol=0)
        for frame in range(0, len(times), 10):
            start = int(particles["Steps/PointOffsets"][frame])
            stop = start + int(particles["NumberOfPoints"][frame])
            center = np.array([body[f"Center:{i}"][frame] for i in range(3)])
            points = particles["Points"][start:stop] - center
            # Cull the bulk water before transforming near-body points.
            near = np.flatnonzero(np.max(np.abs(points), axis=1) < 0.23)
            types = particles["PointData/Type"][start:stop]
            relative = points[near[types[near] == 1]]
            scalar, *axis = [body[f"Orientation:{i}"][frame] for i in range(4)]
            axis = -np.array(axis)
            turn = 2 * np.cross(axis, relative)
            local = relative + scalar * turn + np.cross(axis, turn)
            radius = np.hypot(local[:, 0], local[:, 1])
            height = local[:, 2] + 0.090
            inset = np.minimum(0.29 - height, 0.15 - np.hypot(radius, np.maximum(0.15 - height, 0)))
            yield times[frame], int(np.sum(inset > 1e-6)), max(0., float(np.max(inset, initial=0.))) * 1000


def main():
    experiment = np.loadtxt(ROOT / "reference/Fl1-0p074.txt")
    fig, (heave, intrusion) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                            constrained_layout=True)
    heave.plot(*experiment.T, "--", color="black", label="Experiment")
    with (OUTPUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("mode", "heave_rmse_mm", "max_inside_count_at_20hz", "max_inset_mm_at_20hz"))
        for mode, tag, color in VARIANTS:
            folder = ROOT / "results" / tag
            run = ROOT / "runs" / tag / CASE
            config = tomllib.loads((folder / f"{CASE}.toml").read_text())
            assert config["complete"] and config["frozen_particles"] == 0
            assert config["frames"] == 901 and config["dx"] == .025 and config["sound_speed_m_s"] == 25
            data = np.genfromtxt(folder / f"{CASE}.csv", names=True, delimiter=",")
            target = experiment[(experiment[:, 0] >= data["time"][0]) & (experiment[:, 0] <= data["time"][-1])]
            rmse = np.sqrt(np.mean((np.interp(target[:, 0], data["time"], data["heave_mm"]) - target[:, 1])**2))
            times, counts, inset = map(np.array, zip(*crossings(folder, run)))
            writer.writerow((mode, rmse, max(counts), max(inset)))
            heave.plot(data["time"], data["heave_mm"], color=color, label=mode)
            intrusion.plot(times, counts, color=color, marker=".", markersize=3, label=mode)
    heave.set(ylabel="Heave (mm)", xlim=(0, 4.5))
    intrusion.set(xlabel="Time since release (s)", ylabel="Fluid centers inside float",
                  title="Sampled at 20 Hz")
    for ax in (heave, intrusion):
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
    fig.suptitle("Test 14 · Float 1 down · 25 mm particle spacing")
    fig.savefig(OUTPUT / "comparison.png", dpi=160)


if __name__ == "__main__":
    main()
