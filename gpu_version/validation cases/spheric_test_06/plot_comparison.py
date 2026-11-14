"""Plot saved Julia/DualSPHysics results; numerical analysis lives in Julia."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import h5py

CASE = Path(__file__).resolve().parent
RESULTS = CASE / "results/comparison"
WORK = CASE / "runs/comparison"
COLORS = {"native": "#6651b8", "dual": "#168b86"}
LABELS = {"native": "Julia GPU", "dual": "DualSPHysics 5.4"}


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def reference(re):
    values = []
    for line in (CASE / f"reference/Force_Re{re:03d}.dat").read_text().splitlines():
        try:
            data = [float(x) for x in line.split()]
            if len(data) == 3:
                values.append(data)
        except ValueError:
            pass
    return np.asarray(values)


def drag():
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
    for r, re in enumerate((50, 100, 150)):
        ref = reference(re)
        for c, component in enumerate(("pressure", "viscous")):
            ax = axes[r, c]
            for solver in ("native", "dual"):
                data = rows(RESULTS / f"re{re}_c40/{solver}.csv")
                ax.plot([float(x["time"]) for x in data], [float(x[f"cd_{component}"]) for x in data],
                        color=COLORS[solver], lw=0.85, label=LABELS[solver])
            ax.plot(ref[:, 0], ref[:, c+1], color="#243341", lw=1.4, ls="--", label="Finite difference reference")
            ax.set(xlim=(0, 8), ylabel=f"$C_{{D,{component[0] if c == 0 else 'v'}}}$",
                   title=f"Re = {re}: {component} drag")
            ax.grid(alpha=0.2)
            if r == 2:
                ax.set_xlabel("Time [s]")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Moving-square comparison — raw output, c0 = 40 m/s, L/dp = 40\nIdentical initial lattice; DBC and laminar viscosity; formulation differences documented", fontsize=13)
    fig.savefig(RESULTS / "solver_drag.png", dpi=170)
    plt.close(fig)


def spectra():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True, constrained_layout=True)
    for ax, re in zip(axes, (50, 100, 150)):
        data = rows(RESULTS / f"re{re}_c40/spectra.csv")
        for solver in ("native", "dual"):
            selected = [x for x in data if x["solver"] == solver and x["component"] == "pressure"]
            ax.plot([float(x["frequency"]) for x in selected], [float(x["amplitude"]) for x in selected],
                    color=COLORS[solver], lw=1.4, label=LABELS[solver])
        ax.axvline(2, color="#a85920", ls=":", label="c0 / (2 tank length) = 2 Hz")
        ax.set(xlim=(0, 20), xlabel="Frequency [Hz]", title=f"Re = {re}")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Pressure-drag error amplitude [$C_D$]")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Error spectra, 1.5–6.5 s; linear drift removed, Hann window\nThe common dominant peak is near the tank's acoustic frequency", fontsize=12)
    fig.savefig(RESULTS / "noise_spectra.png", dpi=170)
    plt.close(fig)


def fields():
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for row_index, re in enumerate((50, 100, 150)):
        with h5py.File(WORK / f"re{re}_c40/native/MovingSquare.vtkhdf", "r") as file:
            root = file["VTKHDF"]
            offset, count = int(root["Steps/PointOffsets"][-1]), int(root["NumberOfPoints"][-1])
            columns = slice(offset, offset+count)
            points = root["Points"][columns, :2]
            velocity = root["PointData/Velocity"][columns, :2]
            marker = root["PointData/GroupMarker"][columns]
            native = (points, velocity, marker == 2, marker == 3)
        lines = (WORK / f"re{re}_c40/dual/fields_0400.csv").read_text().splitlines()
        header = next(i for i, line in enumerate(lines) if line.startswith("Pos.x"))
        data = list(csv.DictReader(lines[header:], delimiter=";"))
        points = np.asarray([(float(x["Pos.x [m]"]), float(x["Pos.z [m]"])) for x in data])
        velocity = np.asarray([(float(x["Vel.x [m/s]"]), float(x["Vel.z [m/s]"])) for x in data])
        types = np.asarray([int(x["Type"]) for x in data])
        dual = (points, velocity, types == 3, types == 1)
        for c, (solver, state) in enumerate((("native", native), ("dual", dual))):
            p, v, fluid, moving = state
            center = p[moving].mean(axis=0)
            ax = axes[row_index, c]
            scatter = ax.scatter(p[fluid, 0], p[fluid, 1], c=np.linalg.norm(v[fluid], axis=1),
                                 s=0.45, vmin=0, vmax=2.5, cmap="viridis", rasterized=True)
            ax.add_patch(Rectangle(center-0.5, 1, 1, color="#263441"))
            ax.add_patch(Rectangle((0, 0), 10, 5, fill=False, edgecolor="#263441", lw=1))
            ax.set(xlim=(-0.1, 10.1), ylim=(-0.1, 5.1), aspect="equal", xlabel="x [m]", ylabel="y [m]",
                   title=f"Re = {re} — {LABELS[solver]}")
    fig.colorbar(scatter, ax=axes, shrink=0.75, label="Fluid speed [m/s]")
    fig.suptitle("Current moving-square fields near t = 8 s\nc0 = 40 m/s, L/dp = 40; same speed scale and initial particle lattice", fontsize=13)
    fig.savefig(RESULTS / "solver_fields.png", dpi=190)
    plt.close(fig)


def motion():
    data = rows(RESULTS / "motion.csv")
    t = np.asarray([float(x["time"]) for x in data])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    near = (t >= 3) & (t <= 3.1)
    ax = axes[0, 0]
    for key, label, color in (("acceleration", "Original position + velocity Hermite", COLORS["native"]),
                              ("integrated_acceleration", "Integrated tabulated velocity", "#bc6c25")):
        ax.plot(t[near], np.asarray([float(x[key]) for x in data])[near], lw=1.1, label=label, color=color)
    ax.set(xlabel="Time [s]", ylabel="Prescribed acceleration [m/s²]", title="Constant-speed segment: spurious acceleration")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[0, 1]
    ax.plot(t, 1e6*np.asarray([float(x["integrated_x"])-float(x["x"]) for x in data]), color="#bc6c25", lw=1)
    ax.set(xlabel="Time [s]", ylabel="Position difference [µm]", title="Position change from consistent velocity integration")
    for ax, component in zip(axes[1], ("pressure", "viscous")):
        for variant, label, color in (("re100_c40", "Original motion", COLORS["native"]),
                                       ("re100_integrated", "Integrated-velocity motion", "#bc6c25")):
            file = RESULTS / variant / "native.csv"
            if file.exists():
                values = rows(file)
                ax.plot([float(x["time"]) for x in values], [float(x[f"cd_{component}"]) for x in values],
                        lw=0.8, label=label, color=color)
        ref = reference(100)
        ax.plot(ref[:, 0], ref[:, 1 if component == "pressure" else 2], "--", color="#243341", lw=1.1, label="FD reference")
        ax.set(xlim=(0, 8), xlabel="Time [s]", ylabel="$C_D$", title=f"Re = 100: {component} drag")
        ax.legend(fontsize=8, frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Motion-input diagnostic — a small position correction can remove acceleration jitter", fontsize=13)
    fig.savefig(RESULTS / "motion_diagnostic.png", dpi=170)
    plt.close(fig)


def sound_speed():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    ref = reference(100)
    for column, speed in enumerate((40, 80, 160)):
        for solver in ("native", "dual"):
            data = rows(RESULTS / f"re100_c{speed}/{solver}.csv")
            axes[0, column].plot([float(x["time"]) for x in data], [float(x["cd_pressure"]) for x in data],
                                 lw=0.75, color=COLORS[solver], label=LABELS[solver])
            data = rows(RESULTS / f"re100_c{speed}/spectra.csv")
            selected = [x for x in data if x["solver"] == solver and x["component"] == "pressure"]
            axes[1, column].plot([float(x["frequency"]) for x in selected], [float(x["amplitude"]) for x in selected],
                                 color=COLORS[solver], lw=1.2)
        axes[0, column].plot(ref[:, 0], ref[:, 1], "--", color="#243341", lw=1.1, label="FD reference")
        axes[0, column].set(xlim=(0, 8), xlabel="Time [s]", ylabel="$C_{D,p}$", title=f"c0 = {speed} m/s — raw pressure drag")
        axes[1, column].axvline(speed/20, color="#bc6c25", ls=":", label="Simple box acoustic estimate")
        axes[1, column].set(xlim=(0, 20), xlabel="Frequency [Hz]", ylabel="Error amplitude [$C_D$]", title="Error spectrum, 1.5–6.5 s")
        for ax in axes[:, column]:
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, frameon=False)
    axes[1, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Sound speed changes the pressure-wave problem — Re = 100, L/dp = 40\nHigher sound speed does not guarantee monotonically quieter raw loads", fontsize=13)
    fig.savefig(RESULTS / "sound_speed_noise.png", dpi=170)
    plt.close(fig)


def timing():
    file = RESULTS / "timings.csv"
    if not file.exists():
        return
    data = rows(file)
    aligned_file = RESULTS / "timings_aligned.csv"
    groups = [("Re 50",50,data),("Re 100",100,data),("Re 150",150,data)]
    if aligned_file.exists():
        groups.append(("Re 100\naligned",100,rows(aligned_file)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for solver, offset in (("native", -0.18), ("dual", 0.18)):
        for i, (_, re, source) in enumerate(groups):
            selected = [float(x["wall_seconds"]) for x in source if x["solver"] == solver and int(x["reynolds"]) == re]
            if selected:
                axes[0].bar(i+offset, np.median(selected), width=0.34, color=COLORS[solver], label=LABELS[solver] if i == 0 else None)
                axes[0].scatter(np.full(len(selected), i+offset), selected, s=14, facecolor="white", edgecolor="#263441", zorder=3)
    for i, (_, re, source) in enumerate(groups):
        ratios = []
        for repetition in (1, 2, 3):
            pair = {x["solver"]: float(x["wall_seconds"]) for x in source if int(x["reynolds"]) == re and int(x["repetition"]) == repetition}
            if len(pair) == 2:
                ratios.append(pair["dual"] / pair["native"])
        if ratios:
            axes[1].bar(i, np.median(ratios), color="#536878", width=0.6)
            axes[1].scatter(np.full(len(ratios), i), ratios, s=18, color="#bc6c25", zorder=3)
    for ax in axes:
        ax.set_xticks(range(len(groups)), [label for label,_,_ in groups])
        ax.grid(axis="y", alpha=0.2)
    axes[0].set(ylabel="Whole-solver wall time [s]", title="Warm runs, three alternating pairs")
    axes[0].legend(frameon=False)
    axes[1].set(ylabel="DualSPHysics / Julia wall time", title="Paired elapsed-time ratios")
    fig.suptitle("Same 8 s problem, 84,864 particles and 401 frames; step counts/formulation differences reported")
    fig.savefig(RESULTS / "timing_comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    drag()
    spectra()
    fields()
    motion()
    sound_speed()
    timing()
