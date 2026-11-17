"""Time-localized pressure/viscous drag spectra from raw saved force histories."""
import csv
import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "coefh_refinement" / "fields"
spectrum = runpy.run_path(str(ROOT / "plot_refinement_fields.py"))["spectrum"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, data, transforms = [], {}, {}
    for n in (20, 40, 80):
        d = np.genfromtxt(ROOT / "results" / "coefh_refinement" / "re100" /
                          f"Re100_n{n:03d}.csv", delimiter=",", names=True)
        data[n] = d
        t = d["time"]
        for component in ("pressure", "viscous"):
            centers, columns = [], []
            for first in range(len(t) - 24):
                last = first + 25
                if t[last - 1] > 2.0 + 1e-9:
                    break
                f, a = spectrum(t[first:last], d[f"cd_{component}"][first:last])
                center = (t[first] + t[last - 1]) / 2
                centers.append(center)
                columns.append(a)
                records.extend(dict(n=n, component=component, center_time=center,
                                    window_start=t[first], window_end=t[last - 1],
                                    frequency_hz=frequency, amplitude=amplitude)
                               for frequency, amplitude in zip(f, a))
            transforms[n, component] = (np.array(centers), f, np.array(columns).T)

    fig, axes = plt.subplots(3, 3, figsize=(16, 10), constrained_layout=True)
    for col, n in enumerate((20, 40, 80)):
        d = data[n]
        ax = axes[0, col]
        ax.plot(d["time"], d["cd_pressure"], color="#0072b2", label="Pressure Cd")
        twin = ax.twinx()
        twin.plot(d["time"], d["cd_viscous"], color="#d55e00", label="Viscous Cd")
        ax.set(xlim=(0, 2), ylabel="Pressure Cd", title=f"L/dp={n}")
        twin.set_ylabel("Viscous Cd")
        ax.legend(loc="upper left", fontsize=8)
        twin.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)
    for row, component in ((1, "pressure"), (2, "viscous")):
        maximum = max(transforms[n, component][2][1:].max() for n in (20, 40, 80))
        for col, n in enumerate((20, 40, 80)):
            centers, f, amplitudes = transforms[n, component]
            image = axes[row, col].pcolormesh(centers, f[1:], np.maximum(amplitudes[1:], 1e-6),
                shading="nearest", cmap="magma", norm=LogNorm(vmin=maximum * 1e-4, vmax=maximum))
            peak_time = data[n]["time"][np.argmax(data[n][f"cd_{component}"])]
            axes[row, col].axvline(peak_time, color="cyan", ls="--", lw=1,
                                   label=f"Peak t={peak_time:.2f} s")
            axes[row, col].axhline(8, color="white", ls=":", lw=0.8)
            axes[row, col].set(xlim=(0, 2), ylim=(2, 24), xlabel="Window center [s]",
                               ylabel=f"{component.capitalize()} frequency [Hz]")
            axes[row, col].legend(fontsize=8, loc="upper right")
        fig.colorbar(image, ax=axes[row, :], label=f"{component.capitalize()} amplitude [Cd]")
    fig.suptitle("Re=100 startup: raw forces and sliding-window Fourier amplitudes\n"
                 "25 samples/window (0.48 s span), 0.02 s hop, linear detrend + Hann; nominal 8 Hz dotted")
    fig.savefig(OUT / "startup_time_frequency.png", dpi=180)
    fig.savefig(OUT / "startup_time_frequency.pdf")
    plt.close(fig)
    with (OUT / "startup_time_frequency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    assert len(records) == 3 * 2 * 77 * 13
    print("Verified 6006 startup spectral amplitudes; frequency bins are 2 Hz, Nyquist 25 Hz.")


if __name__ == "__main__":
    main()
