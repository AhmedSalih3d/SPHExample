"""Freeze saved wall neighborhoods to measure MLS extrapolation and density amplification."""
import csv
import tomllib
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

CASE = Path(__file__).resolve().parent
OUT = CASE / "results" / "simple_mdbc_reconstruction_audit"


def reconstruct(wall, ghost, fluid, rho, h, mass, alpha):
    """CPU Float64 equivalent of SimpleMDBC, with fixed-volume sensitivity weights."""
    relative = fluid - ghost
    q = np.linalg.norm(relative, axis=1) / h
    weight = alpha * (1 - q / 2)**4 * (2*q + 1)
    gradient = alpha * (5/8) * (q - 2)[:, None]**3 * (-relative) / h**2
    columns = np.column_stack((weight, gradient)).T * (mass / rho)
    matrix = columns @ np.column_stack((np.ones(len(rho)), relative))
    rhs = columns @ rho
    det = float(np.linalg.det(matrix))
    shepard = float(rhs[0] / matrix[0, 0])
    displacement = wall - ghost
    evaluation = np.r_[1.0, displacement]
    if abs(det) >= 1e-3:
        solution = np.linalg.solve(matrix, rhs)
        sensitivity = np.linalg.solve(matrix.T, evaluation) @ columns
        density = float(evaluation @ solution)
        ghost_density, extrapolation = float(solution[0]), float(displacement @ solution[1:])
        branch = "MLS"
    else:
        sensitivity = columns[0] / matrix[0, 0]
        density, ghost_density, extrapolation, branch = shepard, shepard, 0.0, "Shepard"
    scaled = np.diag([1, h, h]) @ matrix @ np.diag([1, 1/h, 1/h])
    # Volume factors are frozen for this sensitivity test; full simulations update them.
    perturbation = 0.001 * np.sign(sensitivity)
    perturbed_rhs = columns @ (rho + perturbation)
    perturbed = (evaluation @ np.linalg.solve(matrix, perturbed_rhs) if branch == "MLS"
                 else perturbed_rhs[0] / matrix[0, 0])
    np.testing.assert_allclose(perturbed - density, 0.001*np.abs(sensitivity).sum(), atol=2e-8, rtol=1e-6)
    return dict(density=density, shepard=shepard, ghost_density=ghost_density,
                extrapolation=extrapolation, determinant=det, scaled_condition=np.linalg.cond(scaled),
                amplification=np.abs(sensitivity).sum(), branch=branch,
                fluid_min=float(rho.min()), fluid_max=float(rho.max()),
                perturbation_gain=float((perturbed-density)/0.001))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, forces = [], []
    setups = (
        ("no_ddt", "simple_mdbc_zero_ddt", (0.78, 0.80, 0.84, 0.86, 0.88, 7.50, 7.54, 7.60, 8.00)),
        ("fluid_ddt", "simple_mdbc_fluid_ddt_startup", (0.76, 0.78, 0.80, 0.82, 0.86)),
    )
    for label, tag, requested in setups:
        metadata = tomllib.loads((CASE / "results" / tag / "native.toml").read_text())
        saved = np.genfromtxt(CASE / "results" / tag / "native.csv", delimiter=",", names=True)
        h = metadata["h"]
        mass = float(np.float32(1000 * metadata["dx"]**2))
        alpha = float(np.float32(7 / (4*np.pi*float(np.float32(h)**2))))
        with h5py.File(CASE / "runs" / tag / "Re100_n080" / "MovingSquare.vtkhdf") as file:
            root = file["VTKHDF/Particles"] if "VTKHDF/Particles" in file else file["VTKHDF"]
            times = root["Steps/Values"][:]
            for time in requested:
                frame = int(np.argmin(abs(times-time)))
                assert abs(times[frame]-time) < 1e-10
                offset, count = int(root["Steps/PointOffsets"][frame]), int(root["NumberOfPoints"][frame])
                selection = slice(offset, offset+count)
                xy = root["Points"][selection, :2]
                markers = root["PointData/GroupMarker"][selection]
                density = root["PointData/Density"][selection].astype(float)
                water, body = xy[markers == 2], xy[markers == 3]
                rf, rb = density[markers == 2], density[markers == 3]
                tree = cKDTree(water)
                center = body.mean(axis=0)
                relative = body-center
                gap = .5-abs(relative)
                displacement = np.where(np.isclose(gap, gap.min(axis=1)[:, None], atol=1e-12, rtol=0),
                                        2*np.sign(relative)*gap, 0)
                ghosts = body+displacement
                close = tree.query_ball_point(body, 2*h)
                reconstructed = rb.copy()
                shepard_density = rb.copy()
                ghost_density = rb.copy()
                for owner in (i for i, neighbors in enumerate(close) if neighbors):
                    ids = tree.query_ball_point(ghosts[owner], 2*h)
                    if not ids:
                        raise ValueError(f"Empty ghost at {tag}, {time}, {owner}")
                    data = reconstruct(body[owner], ghosts[owner], water[ids], rf[ids], h, mass, alpha)
                    reconstructed[owner], shepard_density[owner], ghost_density[owner] = (
                        data[name] for name in ("density", "shepard", "ghost_density"))
                    face = ("front" if relative[owner, 0] > 0 else "rear") if abs(relative[owner, 0]) >= abs(relative[owner, 1]) else "side"
                    rows.append(dict(model=label, time=time, owner=owner, face=face,
                                     wall_x=relative[owner, 0], wall_y=relative[owner, 1],
                                     stored_density=rb[owner], neighbors=len(ids), **data))
                pairs = cKDTree(body).sparse_distance_matrix(tree, 2*h, output_type="coo_matrix")
                i, j, radius = pairs.row, pairs.col, pairs.data
                valid = radius > 0
                i, j, radius = i[valid], j[valid], radius[valid]
                grad_x = alpha*(5/8)*(radius/h - 2)**3*(body[i, 0]-water[j, 0])/h**2
                fp = 1000*160**2/7*((rf/1000)**7-1)
                face = np.where(abs(relative[:, 0]) >= abs(relative[:, 1]),
                                np.where(relative[:, 0] > 0, "front", "rear"), "side")
                for variant, wall_density in (("stored", rb), ("MLS", reconstructed),
                                              ("ghost_only", ghost_density), ("Shepard", shepard_density)):
                    wp = 1000*160**2/7*((wall_density/1000)**7-1)
                    pair_cd = 2*mass**2/1000 * (wp[i]+fp[j])/(wall_density[i]*rf[j])*grad_x
                    if variant == "MLS":
                        np.testing.assert_allclose(pair_cd.sum(), saved["cd_pressure"][frame],
                                                   atol=2e-4, rtol=2e-5)
                    for side in ("front", "rear", "side", "all"):
                        value = pair_cd.sum() if side == "all" else pair_cd[face[i] == side].sum()
                        forces.append(dict(model=label, time=time, reconstruction=variant, face=side,
                                           pressure_cd=float(value)))
        print(f"Audited {label} frozen neighborhoods", flush=True)
    with (OUT / "wall_reconstruction.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for label, _, times in setups:
        for time in times:
            for face in ("front", "rear", "side"):
                selected = [r for r in rows if r["model"] == label and r["time"] == time and r["face"] == face]
                summary.append(dict(model=label, time=time, face=face, walls=len(selected),
                    mls_walls=sum(r["branch"] == "MLS" for r in selected),
                    condition_max=max(r["scaled_condition"] for r in selected),
                    amplification_max=max(r["amplification"] for r in selected),
                    gradient_density_max=max(abs(r["extrapolation"]) for r in selected),
                    reconstructed_min=min(r["density"] for r in selected),
                    reconstructed_max=max(r["density"] for r in selected),
                    shepard_min=min(r["shepard"] for r in selected),
                    shepard_max=max(r["shepard"] for r in selected),
                    stored_min=min(r["stored_density"] for r in selected),
                    stored_max=max(r["stored_density"] for r in selected)))
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (OUT / "frozen_forces.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(forces[0]))
        writer.writeheader()
        writer.writerows(forces)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, time in zip(axes.flat, (7.50, 7.54, 7.60, 8.00)):
        selected = sorted((r for r in rows if r["model"] == "no_ddt" and r["time"] == time and r["face"] == "front"
                           and r["wall_x"] > .49), key=lambda r:r["wall_y"])
        for field, caption in (("stored_density", "Stored wall"), ("density", "Frozen MLS"), ("shepard", "Frozen Shepard")):
            ax.plot([r["wall_y"] for r in selected], [r[field]-1000 for r in selected], ".-", label=caption)
        ax.set(title=f"Front outer layer, t={time:g} s", xlabel="y relative to square [m]", ylabel="Density minus rho0 [kg/m3]")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Frozen input comparison: SimpleMDBC without diffusion\nShepard shown as diagnostic only, not a validated replacement")
    fig.savefig(OUT / "front_reconstruction.png", dpi=170)
    for row in summary:
        if row["face"] == "front" and row["time"] >= 7:
            print(row)


if __name__ == "__main__":
    main()
