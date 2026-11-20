"""Compare central rear kernel support during the DDT2 refinement runs."""
import argparse
import tomllib

import h5py

from audit_rear_wall_contact import snapshot_metrics
from plot_planar_shifting_convergence import (
    OUT, ROOT, REYNOLDS, RESOLUTIONS, history_path, np, plt, read_history, write_csv,
)


def sample_startup(path, force, meta):
    with h5py.File(path) as handle:
        root = handle["VTKHDF"]
        if "Particles" in root:
            root = root["Particles"]
        times = root["Steps/Values"][:]
        np.testing.assert_array_equal(times, force["time"])
        offsets = root["Steps/PointOffsets"][:]
        counts = root["NumberOfPoints"][:]
        for frame in np.flatnonzero((times >= 0.4) & (times <= 1.2)):
            span = slice(offsets[frame], offsets[frame] + counts[frame])
            density = root["PointData/Density"][span].astype(float)
            rho0 = meta["rho0"]
            pressure = rho0 * meta["sound_speed"]**2 / 7 * ((density / rho0)**7 - 1)
            metrics = snapshot_metrics(
                root["Points"][span, :2], root["PointData/Velocity"][span, :2],
                density, pressure, root["PointData/GroupMarker"][span], meta["dx"], meta["h"],
            )
            yield dict(time=times[frame], pressure_cd=force["cd_pressure"][frame], **metrics)


def main(partial):
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, layout="constrained")
    for col, re in enumerate(REYNOLDS):
        axes[0, col].set_title(f"Re = {re}")
        for n in RESOLUTIONS:
            force_path = history_path(n, "native", re)
            output = ROOT / "runs" / force_path.parents[2].name / "native" / f"re{re}" / "MovingSquare.vtkhdf"
            meta_path = force_path.with_suffix(".toml")
            if partial and (not meta_path.exists() or not output.exists()):
                continue
            force = read_history(force_path)
            meta = tomllib.loads(meta_path.read_text())
            frames = list(sample_startup(output, force, meta))
            rows.extend(dict(reynolds=re, resolution=n, **frame) for frame in frames)
            time = [row["time"] for row in frames]
            axes[0, col].plot(time, [row["pressure_cd"] for row in frames], label=f"L/dp = {n}")
            unsupported = [100 * row["middle_without_fluid_neighbors"] / (n // 4) for row in frames]
            axes[1, col].plot(time, unsupported)
        axes[1, col].set(xlabel="Time (s)", ylim=(-3, 103))
        for ax in axes[:, col]:
            ax.grid(alpha=.25)
    axes[0, 0].set_ylabel("Pressure drag Cd")
    axes[1, 0].set_ylabel("Central probes without support (%)")
    handles, labels = max((ax.get_legend_handles_labels() for ax in axes[0]), key=lambda item: len(item[0]))
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle("SPHExampleGPU · Rear-wall kernel support during startup · Fourtakas DDT2")
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "rear_contact.csv", rows)
    fig.savefig(OUT / "rear_contact.png", dpi=180)
    print(f"Measured rear kernel support at {len(rows)} saved frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true")
    main(parser.parse_args().partial)
