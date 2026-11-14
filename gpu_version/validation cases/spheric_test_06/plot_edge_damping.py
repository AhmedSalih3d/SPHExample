"""Render saved damping experiments; numerical metrics are computed in Julia."""
import csv
from pathlib import Path
import tomllib

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

CASE = Path(__file__).resolve().parent
RESULTS = CASE / "results/edge_damping"
BASELINE = CASE / "results/comparison"
WORK = CASE / "runs/edge_damping"
LABELS = {"native": "Julia GPU", "dual": "DualSPHysics 5.4"}
COLORS = {0: "#969ea7", 10: "#168b86", 100: "#d47d24"}


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def reference(reynolds):
    values = []
    for line in (CASE / f"reference/Force_Re{reynolds:03d}.dat").read_text().splitlines():
        try:
            value = [float(x) for x in line.split()]
            if len(value) == 3:
                values.append(value)
        except ValueError:
            pass
    return np.asarray(values)


def history(reynolds, solver, rate):
    path = BASELINE / f"re{reynolds}_c40" if rate == 0 else RESULTS / f"re{reynolds}_beta{rate}"
    data = rows(path / f"{solver}.csv")
    return {key: np.asarray([float(row[key]) for row in data])
            for key in ("time", "cd_pressure", "cd_viscous")}


def layout():
    x, y = np.meshgrid(np.linspace(0, 10, 1001), np.linspace(0, 5, 501))
    profile = np.maximum.reduce((0*x, (0.5-x)/0.5, (x-9.5)/0.5, (0.5-y)/0.5, (y-4.5)/0.5))
    profile = np.clip(profile, 0, 1)**2
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    shading = ax.pcolormesh(x, y, profile, vmin=0, vmax=1, cmap="YlOrBr", shading="auto", rasterized=True)
    ax.add_patch(Rectangle((0.5, 0.5), 9, 4, fill=False, edgecolor="#303e4b", ls="--", lw=1.2))
    ax.add_patch(Rectangle((1, 2), 1, 1, facecolor="#303e4b", edgecolor="white"))
    with (RESULTS / "geometry.toml").open("rb") as stream:
        final_center = tomllib.load(stream)["square_final_center_x"]
    ax.add_patch(Rectangle((final_center-0.5, 2), 1, 1, fill=False, edgecolor="#303e4b", lw=1.5))
    ax.annotate("Start", (1.5, 2.5), color="white", ha="center", va="center")
    ax.annotate("t = 8 s", (final_center, 2.5), color="#303e4b", ha="center", va="center")
    ax.annotate("Prescribed square motion", xy=(7.6, 2.5), xytext=(3.2, 2.5),
                arrowprops={"arrowstyle": "->", "color": "#303e4b"}, va="center", color="#303e4b")
    ax.text(5, 4.23, "Undamped core: [0.5, 9.5] × [0.5, 4.5] m", ha="center", color="#303e4b")
    ax.text(5, 0.92, "All four edges: 0.5 m quadratic velocity sponge\nβ = 10 or 100 s⁻¹; both velocity components; fluid particles only",
            ha="center", va="center", color="#303e4b", fontsize=10)
    ax.set(xlim=(0, 10), ylim=(0, 5), aspect="equal", xlabel="x [m]", ylabel="Physical y [m] (Dual: z)")
    fig.colorbar(shading, ax=ax, shrink=0.75, label="Damping multiplier s² (maximum at corners)")
    ax.set_title("Four-edge damping test in both solvers\nThe square remains outside the sponge; final clearance is only 26.46 mm", fontsize=13)
    fig.savefig(RESULTS / "zone_layout.png", dpi=170)
    plt.close(fig)


def drag(component):
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
    for r, reynolds in enumerate((50, 100, 150)):
        ref = reference(reynolds)
        for c, solver in enumerate(("native", "dual")):
            ax = axes[r, c]
            for rate in (0, 10, 100):
                data = history(reynolds, solver, rate)
                ax.plot(data["time"], data[f"cd_{component}"], color=COLORS[rate], lw=0.85,
                        label="Undamped" if rate == 0 else f"β = {rate} s⁻¹")
            ax.plot(ref[:, 0], ref[:, 1 if component == "pressure" else 2], color="#263441",
                    lw=1.25, ls="--", label="Original FD reference")
            ax.set(xlim=(0, 8), ylabel="$C_{D,p}$" if component == "pressure" else "$C_{D,v}$",
                   title=f"Re = {reynolds} — {LABELS[solver]}")
            ax.grid(alpha=0.2)
            if r == 2:
                ax.set_xlabel("Time [s]")
        limits = [ax.get_ylim() for ax in axes[r]]
        for ax in axes[r]:
            ax.set_ylim(min(x[0] for x in limits), max(x[1] for x in limits))
    axes[0, 0].legend(fontsize=8, frameon=False, ncols=2)
    fig.suptitle(f"Four-edge damping: raw {component} drag\nc0 = 40 m/s; L/dp = 40; damping changes the benchmark physics", fontsize=13)
    fig.savefig(RESULTS / f"{component}_drag.png", dpi=180)
    plt.close(fig)


def metrics():
    data = rows(RESULTS / "summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey="row", constrained_layout=True)
    for c, solver in enumerate(("native", "dual")):
        for rate, offset in ((10, -0.18), (100, 0.18)):
            selected = [next(row for row in data if row["solver"] == solver and int(row["reynolds"]) == re
                             and int(row["rate"]) == rate) for re in (50, 100, 150)]
            for r, (key, title) in enumerate((("pressure_noise_reduction", "Pressure-noise reduction"),
                                             ("pressure_mean_change_central", "Mean pressure-drag change, 2–6.5 s"))):
                axes[r, c].bar(np.arange(3)+offset, [100*float(row[key]) for row in selected], width=0.34,
                               color=COLORS[rate], label=f"β = {rate} s⁻¹")
                axes[r, c].set(xticks=np.arange(3), xticklabels=(50, 100, 150), xlabel="Re", ylabel="Change [%]",
                               title=f"{LABELS[solver]}: {title}")
                axes[r, c].axhline(0, color="#263441", lw=0.7)
                axes[r, c].grid(axis="y", alpha=0.2)
        axes[0, c].legend(fontsize=9, frameon=False)
    axes[0, 0].set_ylim(0, 125*max(float(row["pressure_noise_reduction"]) for row in data))
    axes[1, 0].set_ylim(0, 118*max(float(row["pressure_mean_change_central"]) for row in data))
    fig.suptitle("Noise reduction must be weighed against changes in the physical load\nNoise = RMS of reference error after subtracting a 0.5 s moving average, 1.5–6.5 s", fontsize=12)
    fig.savefig(RESULTS / "noise_and_bias.png", dpi=180)
    plt.close(fig)


def spectra():
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
    for r, reynolds in enumerate((50, 100, 150)):
        for c, solver in enumerate(("native", "dual")):
            ax = axes[r, c]
            for rate in (0, 10, 100):
                directory = BASELINE / f"re{reynolds}_c40" if rate == 0 else RESULTS / f"re{reynolds}_beta{rate}"
                data = [row for row in rows(directory / "spectra.csv") if row["solver"] == solver and row["component"] == "pressure"]
                ax.plot([float(row["frequency"]) for row in data], [float(row["amplitude"]) for row in data],
                        color=COLORS[rate], lw=1.2, label="Undamped" if rate == 0 else f"β = {rate} s⁻¹")
            ax.axvline(2, color="#263441", ls=":", lw=1, label="c0 / (2Lx) = 2 Hz")
            ax.set(xlim=(0, 12), title=f"Re = {reynolds} — {LABELS[solver]}", ylabel="Error amplitude [$C_D$]")
            ax.grid(alpha=0.2)
            if r == 2:
                ax.set_xlabel("Frequency [Hz]")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Pressure-drag error spectra: all four edges damped\n1.5–6.5 s; linear drift removed; Hann window", fontsize=13)
    fig.savefig(RESULTS / "pressure_spectra.png", dpi=180)
    plt.close(fig)


def native_state(path):
    with h5py.File(path, "r") as file:
        root = file["VTKHDF/Particles"] if "Particles" in file["VTKHDF"] else file["VTKHDF"]
        offset, count = int(root["Steps/PointOffsets"][-1]), int(root["NumberOfPoints"][-1])
        part = slice(offset, offset+count)
        points = root["Points"][part, :2]
        velocity = root["PointData/Velocity"][part, :2]
        marker = root["PointData/GroupMarker"][part]
    return points, velocity, marker == 2, marker == 3


def dual_state(path):
    lines = path.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("Pos.x"))
    data = list(csv.DictReader(lines[header:], delimiter=";"))
    points = np.asarray([(float(row["Pos.x [m]"]), float(row["Pos.z [m]"])) for row in data])
    velocity = np.asarray([(float(row["Vel.x [m/s]"]), float(row["Vel.z [m/s]"])) for row in data])
    types = np.asarray([int(row["Type"]) for row in data])
    return points, velocity, types == 3, types == 1


def draw_field(ax, state, maximum_speed):
    p, v, fluid, moving = state
    center = p[moving].mean(axis=0)
    dots = ax.scatter(p[fluid, 0], p[fluid, 1], c=np.linalg.norm(v[fluid], axis=1), s=0.45,
                      vmin=0, vmax=maximum_speed, cmap="viridis", rasterized=True)
    ax.add_patch(Rectangle(center-0.5, 1, 1, color="#263441"))
    ax.add_patch(Rectangle((0, 0), 10, 5, fill=False, edgecolor="#263441", lw=1))
    ax.add_patch(Rectangle((0.5, 0.5), 9, 4, fill=False, edgecolor="#e5a23d", ls="--", lw=0.8))
    ax.set(xlim=(-0.1, 10.1), ylim=(-0.1, 5.1), aspect="equal", xlabel="x [m]", ylabel="Physical y [m]")
    return dots


def fields():
    states = {}
    for reynolds in (50, 100, 150):
        directory = WORK / f"re{reynolds}_beta100"
        states[reynolds, "native"] = native_state(directory / "native/MovingSquare.vtkhdf")
        states[reynolds, "dual"] = dual_state(directory / "dual/fields_0400.csv")
    maximum_speed = np.ceil(10*max(np.linalg.norm(v[fluid], axis=1).max()
                                   for _, v, fluid, _ in states.values()))/10
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for r, reynolds in enumerate((50, 100, 150)):
        for c, solver in enumerate(("native", "dual")):
            dots = draw_field(axes[r, c], states[reynolds, solver], maximum_speed)
            axes[r, c].set_title(f"Re = {reynolds} — {LABELS[solver]}")
    fig.colorbar(dots, ax=axes, shrink=0.75, label="Fluid speed [m/s]")
    fig.suptitle("Four-edge damping: final fields at t ≈ 8 s, β = 100 s⁻¹\nSame speed scale in every panel; dashed line marks the undamped core", fontsize=13)
    fig.savefig(RESULTS / "fields_beta100.png", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    layout()
    drag("pressure")
    drag("viscous")
    metrics()
    spectra()
    fields()
