"""Plot SPHERIC Test 06 body-following variable-resolution SimpleMDBC results.

All displayed force histories are strictly unfiltered.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE = Path(__file__).resolve().parent
VRES_DIR = CASE / "results/vresolution_simple_mdbc"
UNIF_DIR = CASE / "results/mdbc_validation"


def read_csv_rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_history(base_dir: Path, variant_id: str):
    rows = read_csv_rows(base_dir / variant_id / "native.csv")
    data = {k: np.asarray([float(r[k]) for r in rows]) for k in ("time", "cd_pressure", "cd_viscous")}
    data["cd_total"] = data["cd_pressure"] + data["cd_viscous"]
    return data


def load_reference(reynolds: int):
    rows = []
    for line in (CASE / f"reference/Force_Re{reynolds:03d}.dat").read_text().splitlines():
        try:
            vals = [float(x) for x in line.split()]
            if len(vals) == 3:
                rows.append(vals)
        except ValueError:
            pass
    arr = np.asarray(rows)
    return {"time": arr[:, 0], "cd_pressure": arr[:, 1], "cd_viscous": arr[:, 2], "cd_total": arr[:, 1] + arr[:, 2]}


def bin_series(time, values, dt=0.10):
    edges = np.arange(0.0, 8.0 + 1e-9, dt)
    centers, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (time >= lo) & (time < hi + (1e-9 if hi >= 8.0 - 1e-9 else 0.0))
        if np.any(mask):
            centers.append(0.5 * (lo + hi))
            means.append(float(np.mean(values[mask])))
    return np.asarray(centers), np.asarray(means)


def load_ablation():
    rows = read_csv_rows(VRES_DIR / "diagnostic_ablation_points.csv")
    out = {}
    for r in rows:
        vid = r["variant"]
        out.setdefault(vid, []).append(r)
    parsed = {}
    for vid, items in out.items():
        parsed[vid] = {
            k: np.asarray([float(x[k]) for x in items])
            for k in ("time", "c_rho", "f_rho", "cd_pressure", "cd_viscous", "top_bl_count", "top_bl_ux_max")
        }
    return parsed


def plot_vres_drag_histories():
    ref100 = load_reference(100)
    u40 = load_history(UNIF_DIR, "re100_n40_c40_simple_mdbc_integrated")
    u60 = load_history(UNIF_DIR, "re100_n60_c40_simple_mdbc_integrated")
    v20_40 = load_history(VRES_DIR, "re100_c20_r2_c40_simple_mdbc")
    v20_80 = load_history(VRES_DIR, "re100_c20_r4_c40_simple_mdbc")
    abl = load_ablation()

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), constrained_layout=True)

    # Panel A: Unfiltered Pressure Drag (Re = 100, c0 = 40 m/s)
    ax = axes[0, 0]
    ax.plot(u40["time"], u40["cd_pressure"], color="#999999", lw=0.85, alpha=0.8, label="Uniform SimpleMDBC (L/dp = 40)")
    ax.plot(v20_40["time"], v20_40["cd_pressure"], color="#4477aa", lw=0.9, alpha=0.88, label="VRes SimpleMDBC (20 -> 40, NoShifting)")
    ax.plot(v20_80["time"], v20_80["cd_pressure"], color="#228833", lw=1.05, alpha=0.95, label="VRes SimpleMDBC (20 -> 80, NoShifting)")
    s_a2 = abl["vres_20_40_c40_shift_A2"]
    s_a03 = abl["vres_20_40_c40_shift_A03"]
    ax.plot(s_a2["time"], s_a2["cd_pressure"], color="#cc6677", marker="o", ms=4.5, lw=1.2, ls="-.", label="VRes 20->40 + PlanarShifting (A=2.0)")
    ax.plot(s_a03["time"], s_a03["cd_pressure"], color="#ee7733", marker="s", ms=4.5, lw=1.2, ls=":", label="VRes 20->40 + Shifting (A_eff=0.30)")
    ax.plot(ref100["time"], ref100["cd_pressure"], color="#111111", lw=1.45, ls="--", label="Official FD reference")
    ax.set(xlim=(0, 8), ylim=(-1.0, 10.5), title="(A) Re = 100: Unfiltered Pressure Drag Cd_p(t)", xlabel="Time (s)", ylabel="Cd (pressure)")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=7.6)

    # Panel B: Unfiltered Viscous Drag (Re = 100, c0 = 40 m/s)
    ax = axes[0, 1]
    ax.axhline(0.0, color="#444444", lw=0.8, ls=":")
    ax.plot(u40["time"], u40["cd_viscous"], color="#999999", lw=0.95, alpha=0.85, label="Uniform SimpleMDBC (L/dp = 40)")
    ax.plot(u60["time"], u60["cd_viscous"], color="#666666", lw=0.95, ls="-.", alpha=0.85, label="Uniform SimpleMDBC (L/dp = 60)")
    ax.plot(v20_40["time"], v20_40["cd_viscous"], color="#4477aa", lw=1.05, label="VRes SimpleMDBC (20 -> 40, NoShifting)")
    ax.plot(v20_80["time"], v20_80["cd_viscous"], color="#228833", lw=1.15, label="VRes SimpleMDBC (20 -> 80, NoShifting)")
    ax.plot(s_a2["time"], s_a2["cd_viscous"], color="#cc6677", marker="o", ms=5.0, lw=1.5, label="VRes 20->40 + PlanarShifting (A=2.0)")
    ax.plot(s_a03["time"], s_a03["cd_viscous"], color="#ee7733", marker="s", ms=5.0, lw=1.5, ls="--", label="VRes 20->40 + Shifting (A_eff=0.30)")
    ax.plot(ref100["time"], ref100["cd_viscous"], color="#111111", lw=1.5, ls="--", label="Official FD reference")
    ax.set(xlim=(0, 8), ylim=(-0.08, 1.05), title="(B) Re = 100: Unfiltered Viscous Drag Cd_v(t) — Curing Late Decay", xlabel="Time (s)", ylabel="Cd (viscous)")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=7.6)

    # Panel C: 0.10 s Bin-Averaged Cruise Comparison (t in [1.2, 8.0] s)
    ax = axes[1, 0]
    for data, color, lw, label in (
        (u40, "#888888", 1.2, "Uniform SimpleMDBC (40): Cd_p"),
        (v20_40, "#4477aa", 1.3, "VRes SimpleMDBC (20 -> 40): Cd_p"),
        (v20_80, "#228833", 1.5, "VRes SimpleMDBC (20 -> 80): Cd_p"),
    ):
        tb, pb = bin_series(data["time"], data["cd_pressure"], 0.12)
        ax.plot(tb, pb, color=color, lw=lw, label=label)
    ax.plot(ref100["time"], ref100["cd_pressure"], color="#111111", lw=1.5, ls="--", label="Official FD reference: Cd_p")
    ax.axvspan(1.5, 4.2, color="#228833", alpha=0.08, label="Pre-cavitation window (P_inf > 1.2 kPa)")
    ax.set(xlim=(1.2, 8.0), ylim=(0.4, 4.4), title="(C) Re = 100: 0.12 s Bin-Averaged Cruise Pressure Drag", xlabel="Time (s)", ylabel="Cd (pressure)")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=7.8)

    # Panel D: c0 = 160 m/s Uniform vs VRes Checkpoints (Re = 100)
    ax = axes[1, 1]
    u160_s = load_history(UNIF_DIR, "re100_n40_c160_simple_mdbc_integrated")
    u160_d = load_history(UNIF_DIR, "re100_n40_c160_dbc_integrated")
    v160 = abl["vres_20_40_c160_noshift"]
    ax.plot(u160_d["time"], u160_d["cd_pressure"], color="#4477aa", lw=0.9, alpha=0.82, label="Uniform DBC (40, c0=160): Cd_p")
    ax.plot(u160_s["time"], u160_s["cd_pressure"], color="#228833", lw=0.95, alpha=0.88, label="Uniform SimpleMDBC (40, c0=160): Cd_p")
    ax.plot(v160["time"], v160["cd_pressure"], color="#aa3377", marker="D", ms=5.0, lw=1.4, label="VRes SimpleMDBC (20->40, c0=160): Cd_p")
    ax.plot(ref100["time"], ref100["cd_pressure"], color="#111111", lw=1.45, ls="--", label="Official FD reference: Cd_p")
    ax_r = ax.twinx()
    ax_r.plot(u160_s["time"], u160_s["cd_viscous"], color="#228833", lw=0.9, ls=":", alpha=0.75)
    ax_r.plot(v160["time"], v160["cd_viscous"], color="#aa3377", marker="^", ms=4.8, lw=1.2, ls=":", label="VRes 20->40 (c0=160): Cd_v")
    ax_r.plot(ref100["time"], ref100["cd_viscous"], color="#111111", lw=1.1, ls=":")
    ax_r.set_ylabel("Cd (viscous, dotted)", fontsize=9)
    ax_r.set_ylim(-0.05, 1.05)
    ax.set(xlim=(0, 8), ylim=(0.0, 9.5), title="(D) Re = 100 at c0 = 160 m/s: Acoustic Wave Removal vs Cavitation", xlabel="Time (s)", ylabel="Cd (pressure, solid)")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=7.4, loc="upper right")

    fig.suptitle("SPHERIC Test 06: Body-Following Variable-Resolution SimpleMDBC vs Uniform SPH & FD Reference", fontsize=12)
    fig.savefig(VRES_DIR / "vres_drag_histories.png", dpi=175)
    plt.close(fig)


def plot_vres_physics_ablation():
    abl = load_ablation()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5), constrained_layout=True)

    # Panel A: Closed-tank ambient pressure P_inf(t) = Cb * ((c_rho/1000)^7 - 1)
    ax = axes[0]
    for vid, c0, color, marker, label in (
        ("uni20_c40_noshift", 40.0, "#666666", "o", "Uniform SimpleMDBC (c0=40)"),
        ("vres_20_40_c40_noshift", 40.0, "#4477aa", "s", "VRes NoShifting (c0=40)"),
        ("vres_20_40_c160_noshift", 160.0, "#aa3377", "D", "VRes NoShifting (c0=160)"),
        ("vres_20_40_c40_shift_A2", 40.0, "#cc6677", "^", "VRes + PlanarShifting (A=2.0)"),
        ("vres_20_40_c40_shift_A03", 40.0, "#ee7733", "v", "VRes + Shifting (A_eff=0.30)"),
    ):
        d = abl[vid]
        cb = 1000.0 * c0 * c0 / 7.0
        p_inf = cb * ((d["c_rho"] / 1000.0) ** 7 - 1.0) / 1000.0
        ax.plot(d["time"], p_inf, color=color, marker=marker, ms=4.8, lw=1.45, label=label)
    ax.axhline(1.20, color="#bb0000", ls="--", lw=1.2, label="Corner suction threshold (~1.2 kPa)")
    ax.axhline(0.75, color="#222222", ls=":", lw=1.0, label="Rear-wall limit_density floor (~0.75 kPa)")
    ax.set(xlim=(0.8, 8.2), title="(A) Closed-Tank Ambient Pressure P_inf(t)", xlabel="Time (s)", ylabel="Ambient tank pressure (kPa)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.2)

    # Panel B: Top boundary-layer particle count & max ux
    ax = axes[1]
    for vid, color, marker, label in (
        ("vres_20_40_c40_noshift", "#4477aa", "s", "VRes NoShifting (c0=40)"),
        ("vres_20_40_c160_noshift", "#aa3377", "D", "VRes NoShifting (c0=160)"),
        ("vres_20_40_c40_shift_A2", "#cc6677", "^", "VRes + PlanarShifting (A=2.0)"),
        ("vres_20_40_c40_shift_A03", "#ee7733", "v", "VRes + Shifting (A_eff=0.30)"),
    ):
        d = abl[vid]
        ax.plot(d["time"], d["top_bl_count"], color=color, marker=marker, ms=4.8, lw=1.45, label=f"{label}: count")
    ax.set(xlim=(0.8, 8.2), ylim=(35, 78), title="(B) Top-Face Boundary-Layer Particle Count", xlabel="Time (s)", ylabel="Particles in y in (3.00, 3.04] m")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.4, loc="lower left")

    # Panel C: Mass drift & fine particle count from 401-frame histories
    ax = axes[2]
    for vid, color, label in (
        ("re100_c20_r2_c40_simple_mdbc", "#4477aa", "Post-fix VRes 20 -> 40 (max 0.32%)"),
        ("re100_c20_r4_c40_simple_mdbc", "#228833", "Post-fix VRes 20 -> 80 (max 0.45%)"),
    ):
        rhist = read_csv_rows(VRES_DIR / vid / "resolution_history.csv")
        t = np.asarray([float(r["time"]) for r in rhist])
        m = np.asarray([float(r["represented_fluid_mass"]) for r in rhist])
        ax.plot(t, 100.0 * (m - m[0]) / m[0], color=color, lw=1.35, label=label)
    # Pre-fix 20->80 benchmark points logged before buffer-kinematics fix (0.923% drift):
    t_pre = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    m_pre = np.array([48912.50, 49003.75, 49074.69, 49092.66, 49130.94, 49185.94, 49210.62, 49286.72, 49359.38])
    ax.plot(t_pre, 100.0 * (m_pre - m_pre[0]) / m_pre[0], color="#bb2222", marker="x", ms=5.5, lw=1.3, ls="--", label="Pre-fix VRes 20 -> 80 (0.92% drift)")
    ax.axhline(0.0, color="#222222", ls=":", lw=0.9)
    ax.set(xlim=(0, 8), title="(C) Represented Fluid Mass Conservation", xlabel="Time (s)", ylabel="Mass drift vs t = 0 (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.6)

    fig.suptitle("Root-Cause Physics Diagnosis: Ambient Pressure Relaxation, Corner Cavitation & Buffer Kinematics Fix", fontsize=11.8)
    fig.savefig(VRES_DIR / "vres_physics_ablation.png", dpi=175)
    plt.close(fig)


def plot_re150_c160_comparison():
    ref150 = load_reference(150)
    unif_rows = read_csv_rows(UNIF_DIR / "summary.csv")
    d40 = load_history(UNIF_DIR, "re150_n40_c40_dbc_integrated")
    s40 = load_history(UNIF_DIR, "re150_n40_c40_simple_mdbc_integrated")
    d160 = load_history(UNIF_DIR, "re150_n40_c160_dbc_integrated")
    s160 = load_history(UNIF_DIR, "re150_n40_c160_simple_mdbc_integrated")

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(s40["time"], s40["cd_pressure"], color="#bbbbbb", lw=0.85, label="SimpleMDBC (c0 = 40 m/s)")
    ax.plot(d160["time"], d160["cd_pressure"], color="#4477aa", lw=0.95, label="DBC (c0 = 160 m/s, NRMSE 18.2%)")
    ax.plot(s160["time"], s160["cd_pressure"], color="#228833", lw=0.95, alpha=0.9, label="SimpleMDBC (c0 = 160 m/s, NRMSE 24.5%)")
    ax.plot(ref150["time"], ref150["cd_pressure"], color="#111111", lw=1.45, ls="--", label="Official FD reference (Re = 150)")
    ax.set(xlim=(0, 8), ylim=(-1.0, 10.5), title="(A) Re = 150 Pressure Drag: c0 = 40 vs 160 m/s (L/dp = 40)", xlabel="Time (s)", ylabel="Cd (pressure)")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=7.8)

    ax = axes[0, 1]
    ax.plot(s40["time"], s40["cd_viscous"], color="#bbbbbb", lw=0.9, label="SimpleMDBC (c0 = 40 m/s, NRMSE 22.8%)")
    ax.plot(d160["time"], d160["cd_viscous"], color="#4477aa", lw=1.05, label="DBC (c0 = 160 m/s, NRMSE 22.3%)")
    ax.plot(s160["time"], s160["cd_viscous"], color="#228833", lw=1.05, label="SimpleMDBC (c0 = 160 m/s, NRMSE 30.6%)")
    ax.plot(ref150["time"], ref150["cd_viscous"], color="#111111", lw=1.45, ls="--", label="Official FD reference (Re = 150)")
    ax.set(xlim=(0, 8), ylim=(0.0, 0.95), title="(B) Re = 150 Viscous Drag: Thin Boundary Layer (delta/dp ~ 2.6)", xlabel="Time (s)", ylabel="Cd (viscous)")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=7.8)

    # Panel C: Summary bar chart of Pressure Raw NRMSE across Re = 100 & Re = 150 (c0=40 vs c0=160)
    ax = axes[1, 0]
    labels = ["Re100\nDBC c40", "Re100\nSMDBC c40", "Re100\nDBC c160", "Re100\nSMDBC c160",
              "Re150\nDBC c40", "Re150\nSMDBC c40", "Re150\nDBC c160", "Re150\nSMDBC c160"]
    ids = [
        "re100_n40_c40_dbc_integrated", "re100_n40_c40_simple_mdbc_integrated",
        "re100_n40_c160_dbc_integrated", "re100_n40_c160_simple_mdbc_integrated",
        "re150_n40_c40_dbc_integrated", "re150_n40_c40_simple_mdbc_integrated",
        "re150_n40_c160_dbc_integrated", "re150_n40_c160_simple_mdbc_integrated",
    ]
    rmap = {r["id"]: r for r in unif_rows}
    p_err = [100.0 * float(rmap[k]["pressure_raw_nrmse"]) for k in ids]
    v_err = [100.0 * float(rmap[k]["viscous_raw_nrmse"]) for k in ids]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, p_err, width=w, color="#4477aa", label="Pressure Raw NRMSE (%)")
    ax.bar(x + w / 2, v_err, width=w, color="#ee7733", label="Viscous Raw NRMSE (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set(title="(C) Uniform L/dp = 40 Raw NRMSE: c0 = 40 vs 160 m/s", ylabel="Raw NRMSE (%)")
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False, fontsize=8)

    # Panel D: Late-time viscous bias vs boundary-layer resolution (L/dp = 20, 40, 60)
    ax = axes[1, 1]
    u_conv_s = sorted([r for r in unif_rows if int(r["reynolds"]) == 100 and int(r["sound_speed"]) == 40
                       and r["boundary_mode"] == "SimpleMDBC" and r["shifting_mode"] == "NoShifting"
                       and r["motion"] == "integrated"], key=lambda r: int(r["n"]))
    u_conv_d = sorted([r for r in unif_rows if int(r["reynolds"]) == 100 and int(r["sound_speed"]) == 40
                       and r["boundary_mode"] == "NoMDBC" and r["shifting_mode"] == "NoShifting"
                       and r["motion"] == "integrated"], key=lambda r: int(r["n"]))
    ax.axhline(0.0, color="#222222", ls=":", lw=1.0)
    ax.plot([int(r["n"]) for r in u_conv_d], [100.0 * float(r["viscous_late_bias"]) for r in u_conv_d],
            color="#4477aa", marker="o", lw=1.5, label="Uniform DBC (Re=100, c0=40)")
    ax.plot([int(r["n"]) for r in u_conv_s], [100.0 * float(r["viscous_late_bias"]) for r in u_conv_s],
            color="#228833", marker="^", lw=1.5, label="Uniform SimpleMDBC (Re=100, c0=40)")
    ax.scatter([40], [100.0 * float(rmap["re150_n40_c160_dbc_integrated"]["viscous_late_bias"])],
               color="#4477aa", marker="s", s=65, zorder=5, label="Re=150 DBC (L/dp=40, c0=160: +35.8%)")
    ax.scatter([40], [100.0 * float(rmap["re150_n40_c160_simple_mdbc_integrated"]["viscous_late_bias"])],
               color="#228833", marker="D", s=65, zorder=5, label="Re=150 SimpleMDBC (L/dp=40, c0=160: +48.5%)")
    ax.set(title="(D) Late-Time Viscous Drag Bias vs Wall Resolution L/dp", xlabel="Near-square resolution L/dp", ylabel="Late-time viscous bias (4–8 s, %)", xticks=[20, 40, 60])
    ax.grid(alpha=0.24)
    ax.legend(frameon=False, fontsize=7.5)

    fig.suptitle("SPHERIC Test 06 at Re = 150 & c0 = 160 m/s: Acoustic Convergence vs Boundary-Layer Resolution", fontsize=11.8)
    fig.savefig(VRES_DIR / "re150_c160_comparison.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    plot_vres_drag_histories()
    plot_vres_physics_ablation()
    plot_re150_c160_comparison()
