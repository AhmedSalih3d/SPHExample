"""Compare matched DDT-on/off n80 startup forces and rear-wall pressure."""
import csv
import json
import tomllib
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "zero_ddt_startup"


def main():
    histories, diagnostics, metadata, initial = {}, [], {}, {}
    for mode in ("on", "off"):
        directory = ROOT / "results" / ("coefh_refinement/re100" if mode == "on" else "zero_ddt_startup")
        stem = directory / "Re100_n080"
        m = tomllib.loads(stem.with_suffix(".toml").read_text())
        metadata[mode] = m
        data = np.genfromtxt(stem.with_suffix(".csv"), delimiter=",", names=True)
        histories[mode] = data[data["time"] <= 0.9 + 1e-9]
        assert len(histories[mode]) == 46
        path = ROOT / "runs" / ("coefh_refinement/re100/n80" if mode == "on" else "zero_ddt_startup/Re100_n080") / "MovingSquare.vtkhdf"
        h = m["h"]
        alpha = float(np.float32(7 / (4 * np.pi * float(np.float32(h)**2))))
        mass = float(np.float32(1000 / 80**2))
        with h5py.File(path) as file:
            root = file["VTKHDF/Particles"] if "VTKHDF/Particles" in file else file["VTKHDF"]
            for frame, time in enumerate(histories[mode]["time"]):
                start = int(root["Steps/PointOffsets"][frame])
                stop = start + int(root["NumberOfPoints"][frame])
                x = root["Points"][start:stop, :2]
                rho = root["PointData/Density"][start:stop].astype(float)
                markers = root["PointData/GroupMarker"][start:stop]
                if frame == 0:
                    initial[mode] = (x.copy(), rho.copy(), markers.copy(),
                                     root["PointData/Velocity"][start:stop])
                fluid, body = markers == 2, markers == 3
                assert fluid.sum() == 313600 and body.sum() == 6400
                p = 1000*160**2/7*((rho/1000)**7-1)
                xf, xb, rf, rb, pf, pb = x[fluid], x[body], rho[fluid], rho[body], p[fluid], p[body]
                center = xb.mean(axis=0)
                pairs = cKDTree(xb).sparse_distance_matrix(cKDTree(xf), 2*h, output_type="coo_matrix")
                i, j, distance = pairs.row, pairs.col, pairs.data
                valid = distance > 0
                i, j, distance = i[valid], j[valid], distance[valid]
                gradient = alpha*5/8*(distance/h-2)**3/h**2
                coeff = -mass**2/(rb[i]*rf[j])*gradient*(xb[i, 0]-xf[j, 0])
                force = coeff*(pb[i]+pf[j])
                relative = xb[i]-center
                rear = (abs(relative[:, 0]) >= abs(relative[:, 1])) & (relative[:, 0] < 0)
                reconstructed = -2*force.sum()/1000
                assert np.isclose(reconstructed, histories[mode]["cd_pressure"][frame], atol=2e-5, rtol=2e-5)
                diagnostics.append(dict(mode=mode, time=float(time), cd_pressure=float(reconstructed),
                    rear_cd=float(-2*force[rear].sum()/1000),
                    rear_wall_cd=float(-2*(coeff[rear]*pb[i[rear]]).sum()/1000),
                    rear_fluid_cd=float(-2*(coeff[rear]*pf[j[rear]]).sum()/1000),
                    rear_wall_mean_pa=float(pb[np.unique(i[rear])].mean()),
                    rear_fluid_mean_pa=float(pf[np.unique(j[rear])].mean()),
                    fluid_mean_pa=float(pf.mean()), fluid_std_pa=float(pf.std()),
                    inside_body=int((np.max(abs(xf-center)-.5, axis=1) < 0).sum())))
        print(f"Verified 46 pressure reactions for DDT {mode}", flush=True)
    for key in ("dx", "sound_speed", "h", "cfl", "nu", "rho0", "gravity", "background_pressure",
                "motion_kind", "wall_layers", "start_delay", "divergence_cleaning", "edge_sponge"):
        assert metadata["on"][key] == metadata["off"][key], key
    assert metadata["off"]["density_diffusion_model"].endswith("ZeroDensityDiffusion")
    assert metadata["on"]["density_diffusion_model"].endswith("ZeroGravityLinearDensityDiffusion")
    assert np.array_equal(histories["on"]["time"], histories["off"]["time"])
    assert all(np.array_equal(a, b) for a, b in zip(initial["on"], initial["off"]))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for mode, color in (("on", "#0072b2"), ("off", "#d55e00")):
        rows = [r for r in diagnostics if r["mode"] == mode]
        time = [r["time"] for r in rows]
        for ax, key, label in zip(axes.flat,
            ("cd_pressure", None, "rear_cd", "rear_wall_mean_pa", "rear_fluid_mean_pa", "fluid_mean_pa"),
            ("Pressure Cd", "Viscous Cd", "Rear pressure contribution Cd",
             "Participating rear-wall pressure [Pa]", "Participating rear-fluid pressure [Pa]", "Fluid spatial mean [Pa]")):
            values = histories[mode]["cd_viscous"] if key is None else [r[key] for r in rows]
            ax.plot(time, values, color=color, label=f"DDT {mode}")
            ax.set(xlabel="Time [s]", ylabel=label, xlim=(0, .9))
            ax.axvline(.74, color="grey", ls=":", lw=1)
            ax.grid(alpha=.2)
    reference = np.loadtxt(ROOT / "reference" / "Force_Re100.dat", skiprows=2)
    for ax, column in ((axes[0, 0], 1), (axes[0, 1], 2)):
        selected = reference[:, 0] <= .9
        ax.plot(reference[selected, 0], reference[selected, column], color="black", ls="--",
                lw=1, label="Official FD reference")
        ax.legend(fontsize=8)
    fig.suptitle("Re=100, L/dp=80: matched zero-DDT startup control\n"
                 "c0=160, coefh=1.2, CFL=0.2, original tank, integrated motion; all forces raw")
    fig.tight_layout()
    fig.savefig(OUT / "ddt_comparison.png", dpi=180)
    fig.savefig(OUT / "ddt_comparison.pdf")
    with (OUT / "pressure_diagnosis.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    for mode in ("on", "off"):
        d = histories[mode]
        print(mode, "pressure minimum", float(d["cd_pressure"].min()), "time",
              float(d["time"][d["cd_pressure"].argmin()]), "pressure maximum", float(d["cd_pressure"].max()))
        print(json.dumps(next(r for r in diagnostics if r["mode"] == mode and abs(r["time"]-.74)<1e-9)))


if __name__ == "__main__":
    main()
