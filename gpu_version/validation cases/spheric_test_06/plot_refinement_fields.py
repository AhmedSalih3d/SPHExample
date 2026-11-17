"""Saved-frame fields and temporal pressure spectra; no simulation or force filtering."""
import csv
import json
import tomllib
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "coefh_refinement" / "fields"
TIMES = (0.56, 1.0, 4.0, 8.0)
PROBES = {"upstream": (0.5, 2.5), "upper_tank": (5.0, 4.0)}
RADIUS = 0.1
COLORS = {20: "#d55e00", 40: "#009e73", 80: "#0072b2"}


def spectrum(time, values):
    """One-sided Hann coherent-gain amplitude spectrum after linear detrending."""
    dt = np.diff(time)
    if len(time) < 4 or not np.allclose(dt, dt[0], rtol=0, atol=1e-10):
        raise ValueError("FFT requires at least four uniformly spaced samples")
    trend = np.column_stack((np.ones(len(time)), time - np.mean(time)))
    residual = values - trend @ np.linalg.lstsq(trend, values, rcond=None)[0]
    window = np.hanning(len(time))
    amplitude = np.abs(np.fft.rfft(residual * window)) / window.sum()
    amplitude[1:] *= 2
    if len(time) % 2 == 0:
        amplitude[-1] /= 2
    return np.fft.rfftfreq(len(time), dt[0]), amplitude


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / (name + ".png"), dpi=180)
    fig.savefig(OUT / (name + ".pdf"))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snapshots, signals, diagnostics = {}, {}, []
    for n in (20, 40, 80):
        stem = ROOT / "results" / "coefh_refinement" / "re100" / f"Re100_n{n:03d}"
        m = tomllib.loads(stem.with_suffix(".toml").read_text())
        path = (ROOT / "runs" / "coefh" / "Re100_n040" / "MovingSquare.vtkhdf"
                if n == 40 else ROOT / "runs" / "coefh_refinement" / "re100" / f"n{n}" / "MovingSquare.vtkhdf")
        rows = []
        with h5py.File(path, "r") as file:
            root = file["VTKHDF/Particles"] if "VTKHDF/Particles" in file else file["VTKHDF"]
            times = root["Steps/Values"][:]
            assert len(times) == 401 and times[0] == 0 and times[-1] == 8
            offsets, counts = root["Steps/PointOffsets"][:], root["NumberOfPoints"][:]
            for frame, time in enumerate(times):
                start = int(offsets[frame])
                stop = start + int(counts[frame])
                x = root["Points"][start:stop, :2]
                marker = root["PointData/GroupMarker"][start:stop]
                fluid, body = marker == 2, marker == 3
                assert fluid.sum() == 49 * n**2 and body.sum() == n**2
                center = x[body].mean(axis=0)
                rho = root["PointData/Density"][start:stop][fluid].astype(float)
                p = m["rho0"] * m["sound_speed"]**2 / 7 * ((rho / m["rho0"])**7 - 1) + m["background_pressure"]
                xf = x[fluid]
                if not np.isfinite(p).all():
                    raise ValueError(f"Nonfinite pressure at n={n}, time={time}")
                row = dict(n=n, time=float(time), pressure_mean=float(p.mean()),
                           pressure_std=float(p.std()), pressure_min=float(p.min()), pressure_max=float(p.max()))
                for name, point in PROBES.items():
                    selected = np.sum((xf - point)**2, axis=1) <= RADIUS**2
                    if not selected.any():
                        raise ValueError(f"Empty pressure probe {name} at n={n}, time={time}")
                    weights = 1 / rho[selected]
                    row[name] = float(np.average(p[selected], weights=weights))
                rows.append(row)
                for requested in TIMES:
                    if abs(time - requested) < 1e-9:
                        velocity = root["PointData/Velocity"][start:stop, :2][fluid].astype(float)
                        snapshots[n, requested] = (xf, velocity, p, center)
                        diagnostics.append(dict(n=n, time=float(time),
                            pressure_min=float(p.min()), pressure_max=float(p.max()),
                            speed_max=float(np.linalg.norm(velocity, axis=1).max())))
        signals[n] = rows
        print(f"Read all 401 pressure frames at L/dp={n}", flush=True)

    p_limit = max(np.max(np.abs(s[2])) for s in snapshots.values())
    speed_limit = max(np.linalg.norm(s[1], axis=1).max() for s in snapshots.values())
    deviation_limit = max(np.max(np.abs(s[2] - s[2].mean())) for s in snapshots.values())
    for field in ("speed", "pressure", "pressure_deviation"):
        fig, axes = plt.subplots(3, 4, figsize=(18, 8), sharex=True, sharey=True, constrained_layout=True)
        for i, n in enumerate((20, 40, 80)):
            for j, time in enumerate(TIMES):
                x, velocity, p, center = snapshots[n, time]
                values = np.linalg.norm(velocity, axis=1) if field == "speed" else p
                limit = p_limit
                if field == "pressure_deviation":
                    values = p - p.mean()
                    limit = deviation_limit
                stride = max(1, int(np.ceil(len(x) / 100000)))
                ax = axes[i, j]
                image = ax.scatter(x[::stride, 0], x[::stride, 1], c=values[::stride],
                    s=0.5, rasterized=True, cmap="viridis" if field == "speed" else "RdBu_r",
                    vmin=0 if field == "speed" else -limit,
                    vmax=speed_limit if field == "speed" else limit)
                if field == "speed":
                    edges = (np.linspace(0, 10, 31), np.linspace(0, 5, 16))
                    count, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=edges)
                    vx, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=edges, weights=velocity[:, 0])
                    vy, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=edges, weights=velocity[:, 1])
                    mask = count > 0
                    gx, gy = np.meshgrid((edges[0][1:] + edges[0][:-1]) / 2,
                                         (edges[1][1:] + edges[1][:-1]) / 2, indexing="ij")
                    ax.quiver(gx[mask], gy[mask], vx[mask] / count[mask], vy[mask] / count[mask],
                              color="white", scale=12, width=0.003)
                ax.add_patch(Rectangle(center - 0.5, 1, 1, facecolor="#cccccc", edgecolor="black"))
                for name, point in PROBES.items():
                    ax.plot(*point, marker="+", color="black", ms=5)
                ax.set(xlim=(0, 10), ylim=(0, 5), aspect="equal",
                       title=f"L/dp={n}, t={time:g} s")
        label = "Speed [m/s]" if field == "speed" else (
            "Pressure minus frame spatial mean [Pa]" if field == "pressure_deviation" else "Gauge pressure [Pa]")
        fig.colorbar(image, ax=axes, shrink=0.8, label=label)
        fig.suptitle(f"Re=100 {field} fields; common full-range color scale, all resolutions/times\n"
                     "Particles subsampled only for display; crosses mark fixed pressure probes")
        fig.savefig(OUT / (field + "_fields.png"), dpi=180)
        fig.savefig(OUT / (field + "_fields.pdf"))
        plt.close(fig)

    spectra = []
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for n, rows in signals.items():
        time = np.array([r["time"] for r in rows])
        drag = np.genfromtxt(ROOT / "results" / "coefh_refinement" / "re100" / f"Re100_n{n:03d}.csv",
                             delimiter=",", names=True)
        for col, name in enumerate(("upstream", "upper_tank", "pressure_drag")):
            values = drag["cd_pressure"] if name == "pressure_drag" else np.array([r[name] for r in rows])
            axes[0, col].plot(time, values, color=COLORS[n], lw=1, label=f"L/dp={n}")
            selected = (time >= 1.5) & (time <= 6.5)
            for window_name, chosen in (("full", np.ones(len(time), dtype=bool)), ("central", selected)):
                frequency, amplitude = spectrum(time[chosen], values[chosen])
                for f, a in zip(frequency, amplitude):
                    spectra.append(dict(n=n, signal=name, window=window_name,
                                        frequency_hz=float(f), amplitude=float(a)))
                spectral_row = 2 if window_name == "central" else 1
                axes[spectral_row, col].semilogy(frequency[1:], np.maximum(amplitude[1:], 1e-12),
                                                color=COLORS[n], lw=1, label=f"L/dp={n}")
            axes[0, col].set(xlabel="Time [s]", ylabel="Cd" if name == "pressure_drag" else "Gauge pressure [Pa]",
                             title=name.replace("_", " "))
            for spectral_row, title in ((1, "Full window: 0-8 s"), (2, "Central window: 1.5-6.5 s")):
                axes[spectral_row, col].set(xlabel="Frequency [Hz]",
                    ylabel="Amplitude [Cd]" if name == "pressure_drag" else "Amplitude [Pa]",
                    xlim=(0, 25), title=title)
    for ax in axes[1:].flat:
        for f in (8, 16, 24):
            ax.axvline(f, color="grey", ls="--", lw=0.8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Pressure signals and full/central-window FFT amplitudes\n"
                 "Linear detrend + Hann for spectral estimation only; dashed lines: nominal 8 Hz and harmonics")
    save(fig, "pressure_spectra")
    for name, records in (("pressure_signals.csv", [r for rows in signals.values() for r in rows]),
                          ("pressure_spectra.csv", spectra), ("field_extrema.csv", diagnostics)):
        with (OUT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (OUT / "README.md").write_text(
        "# Re=100 flow fields and pressure Fourier spectra\n\n"
        "Saved full histories at L/dp=20/40/80; no new simulation. Speed and gauge-pressure\n"
        "images use identical full-range color limits across resolutions and snapshots.\n"
        "At most 100,000 fluid particles per panel are displayed; numerical extrema and\n"
        "pressure signals use all fluid particles. Arrows are spatial-bin mean velocities.\n\n"
        "Pressure is reconstructed in Float64 with the run's Tait EOS, c0=160 m/s,\n"
        "rho0=1000 kg/m3 and zero background offset. Crosses mark fixed pressure probes\n"
        "at (0.5,2.5) and (5,4) m: volume-weighted fluid means inside radius 0.1 m.\n"
        "These spatial averages are not individual material-particle histories.\n\n"
        "FFT files include full 0-8 s and central 1.5-6.5 s windows. Each signal is\n"
        "linearly detrended and Hann-windowed before rFFT. Amplitudes use coherent-gain\n"
        "normalization and the one-sided factor (excluding DC/Nyquist). The displayed\n"
        "time histories remain raw. This is amplitude, not PSD. Full-window spectra\n"
        "contain the physical startup transient and must not be interpreted as pure noise.\n\n"
        "At output dt=0.02 s, Nyquist is 25 Hz; higher-frequency content can alias.\n"
        "Central-window bin spacing is about 0.199 Hz; Hann broadens spectral peaks.\n"
        "Dashed 8/16/24 Hz lines are longitudinal empty-tank estimates c0/(2Lx)\n"
        "and harmonics, not identified eigenmodes of the tank containing a square.\n"
        "Transverse and obstacle-modified modes can also occur. Neither a spectral\n"
        "peak nor its absence alone proves the origin of the startup force spike.\n\n"
        "![Velocity fields](speed_fields.png)\n\n![Pressure fields](pressure_fields.png)\n\n"
        "Pressure-deviation panels subtract each frame's fluid spatial mean for visibility;\n"
        "they include physical pressure gradients and are not an acoustic decomposition.\n\n"
        "![Pressure deviations](pressure_deviation_fields.png)\n\n"
        "![Pressure spectra](pressure_spectra.png)\n", encoding="utf-8")
    print(json.dumps({"snapshot_pressure_limit_pa": float(p_limit), "snapshot_speed_limit_ms": float(speed_limit)}))


if __name__ == "__main__":
    main()
