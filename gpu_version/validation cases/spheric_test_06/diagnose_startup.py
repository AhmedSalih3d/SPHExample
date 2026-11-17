"""Reconstruct local square/fluid pressure reactions around saved-frame divergence."""
import csv
import tomllib
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "coefh_refinement" / "startup_diagnosis"
TIMES = np.arange(14, 46) * 0.02
SNAPSHOTS = (0.34, 0.38, 0.42, 0.46, 0.70, 0.72, 0.74, 0.76)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, images = [], {}
    for n in (20, 40, 80):
        source = ROOT / "results" / "coefh_refinement" / "re100" / f"Re100_n{n:03d}"
        m = tomllib.loads(source.with_suffix(".toml").read_text())
        forces = np.genfromtxt(source.with_suffix(".csv"), delimiter=",", names=True)
        path = (ROOT / "runs" / "coefh" / "Re100_n040" / "MovingSquare.vtkhdf"
                if n == 40 else ROOT / "runs" / "coefh_refinement" / "re100" / f"n{n}" / "MovingSquare.vtkhdf")
        h = m["h"]
        alpha = float(np.float32(7 / (4 * np.pi * float(np.float32(h)**2))))
        mass = float(np.float32(1000 / n**2))
        with h5py.File(path) as file:
            root = file["VTKHDF/Particles"] if "VTKHDF/Particles" in file else file["VTKHDF"]
            times = root["Steps/Values"][:]
            for requested in TIMES:
                frame = int(np.argmin(abs(times - requested)))
                assert abs(times[frame] - requested) < 1e-9
                start = int(root["Steps/PointOffsets"][frame])
                stop = start + int(root["NumberOfPoints"][frame])
                x = root["Points"][start:stop, :2]
                rho = root["PointData/Density"][start:stop].astype(float)
                marker = root["PointData/GroupMarker"][start:stop]
                fluid, body = marker == 2, marker == 3
                p = 1000 * 160**2 / 7 * ((rho / 1000)**7 - 1)
                center = x[body].mean(axis=0)
                xf, xb = x[fluid], x[body]
                pf, pb = p[fluid], p[body]
                rf, rb = rho[fluid], rho[body]
                pairs = cKDTree(xb).sparse_distance_matrix(cKDTree(xf), 2*h, output_type="coo_matrix")
                i, j, distance = pairs.row, pairs.col, pairs.data
                valid = distance > 0
                i, j, distance = i[valid], j[valid], distance[valid]
                displacement = xb[i] - xf[j]
                gradient = alpha * 5 / 8 * (distance / h - 2)**3 / h**2
                coefficient = -mass**2 / (rb[i] * rf[j]) * gradient * displacement[:, 0]
                pair_force = coefficient * (pb[i] + pf[j])
                mean = pf.mean()
                uniform_force = coefficient * 2 * mean
                relative = xb[i] - center
                face = np.where(abs(relative[:, 0]) >= abs(relative[:, 1]),
                                np.where(relative[:, 0] > 0, 0, 1),
                                np.where(relative[:, 1] > 0, 2, 3))
                clearance = np.max(abs(xf - center) - 0.5, axis=1)
                near = np.max(abs(xf - center), axis=1) < 0.5 + 2*h
                row = dict(n=n, time=float(times[frame]),
                    cd_reconstructed=-2*pair_force.sum()/1000,
                    cd_saved=float(forces["cd_pressure"][frame]),
                    cd_uniform=-2*uniform_force.sum()/1000,
                    cd_deviation=-2*(pair_force-uniform_force).sum()/1000,
                    fluid_mean_pa=float(mean), body_mean_pa=float(pb.mean()),
                    fluid_pressure_min=float(pf.min()), fluid_pressure_max=float(pf.max()),
                    near_min_density=float(rf[near].min()), near_max_density=float(rf[near].max()),
                    min_body_clearance=float(clearance.min()), inside_body=int((clearance < 0).sum()),
                    accepted_pairs=len(i), nearest_pair_dp=float(distance.min()*n))
                for k, name in enumerate(("front", "rear", "top", "bottom")):
                    selected = face == k
                    row[name+"_cd"] = float(-2*pair_force[selected].sum()/1000)
                    row[name+"_wall_cd"] = float(-2*(coefficient[selected]*pb[i[selected]]).sum()/1000)
                    row[name+"_fluid_cd"] = float(-2*(coefficient[selected]*pf[j[selected]]).sum()/1000)
                    row[name+"_uniform_cd"] = float(-2*uniform_force[selected].sum()/1000)
                    row[name+"_pairs"] = int(selected.sum())
                    row[name+"_wall_pressure"] = float(np.mean(pb[np.unique(i[selected])]))
                    row[name+"_fluid_pressure"] = float(np.mean(pf[np.unique(j[selected])]))
                if not np.isclose(row["cd_reconstructed"], row["cd_saved"], atol=2e-5, rtol=2e-5):
                    raise ValueError(f"Reaction reconstruction mismatch: {row}")
                records.append(row)
                if n in (40, 80) and any(abs(requested - t) < 1e-9 for t in SNAPSHOTS):
                    close = np.max(abs(xf-center), axis=1) < 0.8
                    images[n, round(requested, 2)] = (xf[close]-center, pf[close]-mean, xb-center, pb-mean)
        print(f"Verified 32 exact-frame pressure decompositions at n={n}", flush=True)
    with (OUT / "pressure_force_decomposition.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    for n, color in ((20, "#d55e00"), (40, "#009e73"), (80, "#0072b2")):
        rows = [r for r in records if r["n"] == n]
        t = [r["time"] for r in rows]
        for ax, key, label in zip(axes.flat,
            ("cd_saved", "cd_uniform", "cd_deviation", "fluid_mean_pa", "front_cd", "rear_cd",
             "inside_body", "nearest_pair_dp", "min_body_clearance"),
            ("Total pressure Cd", "Uniform-mean contribution Cd", "Pressure-deviation contribution Cd",
             "Fluid spatial mean [Pa]", "Front contribution Cd", "Rear contribution Cd",
             "Fluid centres inside square", "Closest body/fluid distance / dp", "Minimum square clearance [m]")):
            ax.plot(t, [r[key] for r in rows], color=color, label=f"L/dp={n}")
            ax.set(ylabel=label, xlabel="Time [s]")
            ax.axvline(0.36, ls=":", color="grey")
            ax.axvline(0.74, ls="--", color="grey")
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("Re=100 saved-frame diagnosis: onset at 0.36 s; finest sharp drop at 0.74 s\n"
                 "Uniform/deviation decomposition is an algebraic diagnostic, not corrected forces")
    fig.tight_layout()
    fig.savefig(OUT / "force_decomposition.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 8, figsize=(24, 6), sharex=True, sharey=True, constrained_layout=True)
    limit = max(max(abs(p).max(), abs(wall_p).max()) for _, p, _, wall_p in images.values())
    for row, n in enumerate((40, 80)):
        for col, time in enumerate(SNAPSHOTS):
            x, p, wall_x, wall_p = images[n, time]
            ax = axes[row, col]
            image = ax.scatter(x[:, 0], x[:, 1], c=p, s=3, cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.add_patch(Rectangle((-.5, -.5), 1, 1, facecolor="#cccccc", edgecolor="black"))
            ax.scatter(wall_x[:, 0], wall_x[:, 1], c=wall_p, s=3, cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.set(aspect="equal", xlim=(-.8,.8), ylim=(-.8,.8), title=f"n={n}, t={time:g} s")
    fig.colorbar(image, ax=axes, label="Fluid AND wall pressure minus fluid spatial mean [Pa]", shrink=.8)
    fig.savefig(OUT / "near_square_startup.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
