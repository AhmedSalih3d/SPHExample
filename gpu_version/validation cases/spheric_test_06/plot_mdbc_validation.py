"""Plot boundary-traction verification and SPHERIC Test 06 mDBC/shifting/resolution results.

All displayed force histories are strictly unfiltered.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASE = Path(__file__).resolve().parent
RESULTS = CASE / "results/mdbc_validation"


def read_csv_rows(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_history(variant_id: str):
    rows = read_csv_rows(RESULTS / variant_id / "native.csv")
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


def plot_traction_verification():
    p_rows = read_csv_rows(RESULTS / "traction_pressure.csv")
    c_dyn = read_csv_rows(RESULTS / "traction_couette_dynamic.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)

    # Panel A: Hydrostatic positive buoyancy
    ax = axes[0, 0]
    for mode, label, color, marker in (
        ("NoMDBC_rho0_walls", "DBC (rho0 walls)", "#999999", "o"),
        ("NoMDBC_hydro_walls", "DBC (hydro walls)", "#4477aa", "s"),
        ("SimpleMDBC", "SimpleMDBC", "#228833", "^"),
        ("UpdatedMDBC", "UpdatedMDBC", "#cc6677", "D"),
    ):
        sel = sorted([r for r in p_rows if r["regime"] == "hydrostatic_positive" and r["mode"] == mode], key=lambda r: int(r["n"]))
        ax.plot([int(r["n"]) for r in sel], [100.0 * float(r["abs_relative_error"]) for r in sel],
                marker=marker, lw=1.5, color=color, label=label)
    ax.set(title="1A. Hydrostatic buoyancy |error| on submerged square", xlabel="Resolution L/dp", ylabel="Relative force error (%)", xticks=[20, 40, 60])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # Panel B: Zero-ambient suction-compression linear pressure field
    ax = axes[0, 1]
    for mode, label, color, marker in (
        ("NoMDBC_rho0_walls", "DBC (clamped >= rho0)", "#4477aa", "o"),
        ("SimpleMDBC_predictor_clamped", "SimpleMDBC (predictor clamped)", "#228833", "^"),
        ("UpdatedMDBC", "UpdatedMDBC (negative p + a_w cloned)", "#cc6677", "D"),
    ):
        sel = sorted([r for r in p_rows if r["regime"] == "zero_ambient_suction_acceleration" and r["mode"] == mode], key=lambda r: int(r["n"]))
        ax.plot([int(r["n"]) for r in sel], [100.0 * float(r["abs_relative_error"]) for r in sel],
                marker=marker, lw=1.5, color=color, label=label)
    ax.set(title="1B. Zero-ambient suction/acceleration |error| (Fx = 1000 N/m)", xlabel="Resolution L/dp", ylabel="Relative force error (%)", xticks=[20, 40, 60])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # Panel C: Dynamic Couette wall shear error
    ax = axes[1, 0]
    for mode, label, color, marker in (
        ("NoMDBC", "DBC (NoMDBC)", "#4477aa", "o"),
        ("SimpleMDBC", "SimpleMDBC", "#228833", "^"),
        ("UpdatedMDBC", "UpdatedMDBC (mirrored no-slip)", "#cc6677", "D"),
    ):
        sel = sorted([r for r in c_dyn if r["mode"] == mode], key=lambda r: int(r["n"]))
        ax.plot([int(r["n"]) for r in sel], [100.0 * float(r["abs_tau_relative_error"]) for r in sel],
                marker=marker, lw=1.5, color=color, label=label)
    ax.set(title="2A. Dynamic Couette wall shear |error| at t = 2.5 s", xlabel="Channel resolution H/dp", ylabel="Relative wall-shear error (%)", xticks=[20, 40])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # Panel D: Dynamic Couette interior velocity-profile NRMSE
    ax = axes[1, 1]
    for mode, label, color, marker in (
        ("NoMDBC", "DBC (NoMDBC)", "#4477aa", "o"),
        ("SimpleMDBC", "SimpleMDBC", "#228833", "^"),
        ("UpdatedMDBC", "UpdatedMDBC (mirrored no-slip)", "#cc6677", "D"),
    ):
        sel = sorted([r for r in c_dyn if r["mode"] == mode], key=lambda r: int(r["n"]))
        ax.plot([int(r["n"]) for r in sel], [100.0 * float(r["velocity_profile_nrmse"]) for r in sel],
                marker=marker, lw=1.5, color=color, label=label)
    ax.set(title="2B. Dynamic Couette velocity-profile NRMSE at t = 2.5 s", xlabel="Channel resolution H/dp", ylabel="Profile NRMSE (% of U_top)", xticks=[20, 40])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Foundational boundary pressure and viscous wall-traction verification", fontsize=12)
    fig.savefig(RESULTS / "traction_verification.png", dpi=170)
    plt.close(fig)


def plot_boundary_drag(component: str):
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 9.2), sharex=True, constrained_layout=True)
    styles = (
        ("dbc", "#4477aa", "DBC (NoMDBC)"),
        ("simple_mdbc", "#228833", "SimpleMDBC"),
        ("updated_mdbc", "#cc6677", "UpdatedMDBC"),
    )
    row_ylims = ((-0.05, 1.40), (-0.05, 1.15), (-0.05, 0.95))
    for r_idx, re in enumerate((50, 100, 150)):
        ref = load_reference(re)
        for c_idx, (shift_suffix, shift_title) in enumerate((("", "NoShifting"), ("_shift", "PlanarShifting"))):
            ax = axes[r_idx, c_idx]
            for tag, color, label in styles:
                vid = f"re{re}_n40_c40_{tag}{shift_suffix}_integrated"
                data = load_history(vid)
                ax.plot(data["time"], data[f"cd_{component}"], color=color, lw=0.9, alpha=0.9, label=label)
            ax.plot(ref["time"], ref[f"cd_{component}"], color="#111111", lw=1.35, ls="--", label="Official FD reference")
            ax.set(xlim=(0, 8), title=f"Re = {re}, {shift_title}", ylabel=f"Cd ({component})")
            if component == "viscous":
                ax.set_ylim(*row_ylims[r_idx])
            ax.grid(alpha=0.2)
            if r_idx == 2:
                ax.set_xlabel("Time (s)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    note = " (unshifted UpdatedMDBC exceeds axis due to interior wall penetration)" if component == "viscous" else ""
    fig.suptitle(f"SPHERIC Test 06: raw {component} drag across boundary and shifting modes{note}\nL/dp = 40, c0 = 40 m/s, CFL = 0.2, pairwise default timestep, integrated velocity", fontsize=11)
    fig.savefig(RESULTS / f"{component}_drag_boundaries.png", dpi=170)
    plt.close(fig)


def plot_clearance_and_errors():
    rows = read_csv_rows(RESULTS / "summary.csv")
    base = [r for r in rows if int(r["n"]) == 40 and int(r["sound_speed"]) == 40 and r["motion"] == "integrated" and float(r["p_bg"]) == 0.0]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    configs = (
        ("NoMDBC", "NoShifting", "#4477aa", "--", "o", "DBC, NoShifting"),
        ("SimpleMDBC", "NoShifting", "#228833", "--", "^", "SimpleMDBC, NoShifting"),
        ("UpdatedMDBC", "NoShifting", "#cc6677", "--", "D", "UpdatedMDBC, NoShifting"),
        ("NoMDBC", "PlanarShifting", "#4477aa", "-", "o", "DBC, PlanarShifting"),
        ("SimpleMDBC", "PlanarShifting", "#228833", "-", "^", "SimpleMDBC, PlanarShifting"),
        ("UpdatedMDBC", "PlanarShifting", "#cc6677", "-", "D", "UpdatedMDBC, PlanarShifting"),
    )
    for bmode, smode, color, ls, marker, label in configs:
        sel = sorted([r for r in base if r["boundary_mode"] == bmode and r["shifting_mode"] == smode], key=lambda r: int(r["reynolds"]))
        res = [int(r["reynolds"]) for r in sel]
        axes[0].plot(res, [100.0 * float(r["pressure_raw_nrmse"]) for r in sel], color=color, ls=ls, marker=marker, lw=1.4, label=label)
        axes[1].plot(res, [100.0 * float(r["viscous_raw_nrmse"]) for r in sel], color=color, ls=ls, marker=marker, lw=1.4, label=label)
        axes[2].plot(res, [1000.0 * float(r["minimum_clearance"]) for r in sel], color=color, ls=ls, marker=marker, lw=1.4, label=label)

    axes[0].set(title="Raw pressure NRMSE (0.1–8 s)", xlabel="Reynolds number", ylabel="Error (%)", xticks=[50, 100, 150])
    axes[1].set(title="Raw viscous NRMSE (0.1–8 s)", xlabel="Reynolds number", ylabel="Error (%, log scale)", xticks=[50, 100, 150], yscale="log")
    axes[2].axhline(0.0, color="#222222", ls=":", lw=1.0)
    axes[2].set(title="Worst fluid-to-solid clearance", xlabel="Reynolds number", ylabel="Minimum clearance (mm)", xticks=[50, 100, 150])
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Boundary & shifting comparison across Reynolds numbers (L/dp = 40, c0 = 40 m/s)", fontsize=11.5)
    fig.savefig(RESULTS / "clearance_and_errors.png", dpi=170)
    plt.close(fig)


def plot_resolution_and_c160():
    rows = read_csv_rows(RESULTS / "summary.csv")
    ref = load_reference(100)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)

    # Top-left: Spatial resolution convergence of raw pressure & viscous NRMSE at Re=100, c0=40
    ax = axes[0, 0]
    for bmode, smode, color, marker, label in (
        ("NoMDBC", "NoShifting", "#4477aa", "o", "DBC (NoShifting)"),
        ("SimpleMDBC", "NoShifting", "#228833", "^", "SimpleMDBC (NoShifting)"),
        ("UpdatedMDBC", "PlanarShifting", "#cc6677", "D", "UpdatedMDBC + PlanarShifting"),
    ):
        sel = sorted([r for r in rows if int(r["reynolds"]) == 100 and int(r["sound_speed"]) == 40
                      and r["boundary_mode"] == bmode and r["shifting_mode"] == smode
                      and r["motion"] == "integrated" and float(r["p_bg"]) == 0.0], key=lambda r: int(r["n"]))
        ns = [int(r["n"]) for r in sel]
        ax.plot(ns, [100.0 * float(r["pressure_raw_nrmse"]) for r in sel], color=color, marker=marker, ls="-", lw=1.5, label=f"{label}: pressure")
        ax.plot(ns, [100.0 * float(r["viscous_raw_nrmse"]) for r in sel], color=color, marker=marker, ls="--", lw=1.2, label=f"{label}: viscous")
    ax.set(title="Spatial refinement at Re = 100, c0 = 40 m/s", xlabel="Resolution L/dp", ylabel="Raw NRMSE (%)", xticks=[20, 40, 60])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7.2, ncol=2)

    # Top-right: UpdatedMDBC + PlanarShifting raw pressure history across L/dp = 20, 40, 60
    ax = axes[0, 1]
    for n, color in ((20, "#88ccee"), (40, "#cc6677"), (60, "#882255")):
        data = load_history(f"re100_n{n}_c40_updated_mdbc_shift_integrated")
        ax.plot(data["time"], data["cd_pressure"], color=color, lw=0.95, label=f"UpdatedMDBC + Shift, L/dp = {n}")
    ax.plot(ref["time"], ref["cd_pressure"], color="#111111", lw=1.35, ls="--", label="Official FD reference")
    ax.set(xlim=(0, 8), title="UpdatedMDBC + PlanarShifting pressure drag vs L/dp (c0 = 40)", xlabel="Time (s)", ylabel="Cd (pressure)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # Bottom-left: c0 = 160 m/s raw pressure drag at Re = 100, L/dp = 40
    ax = axes[1, 0]
    for tag, color, label in (
        ("dbc", "#4477aa", "DBC (NoShifting)"),
        ("simple_mdbc", "#228833", "SimpleMDBC (NoShifting)"),
        ("updated_mdbc_shift", "#cc6677", "UpdatedMDBC + PlanarShifting"),
    ):
        data = load_history(f"re100_n40_c160_{tag}_integrated")
        ax.plot(data["time"], data["cd_pressure"], color=color, lw=0.9, label=label)
    ax.plot(ref["time"], ref["cd_pressure"], color="#111111", lw=1.35, ls="--", label="Official FD reference")
    ax.set(xlim=(0, 8), title="Re = 100, c0 = 160 m/s: raw pressure drag (L/dp = 40)", xlabel="Time (s)", ylabel="Cd (pressure)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # Bottom-right: c0 = 160 m/s raw viscous drag at Re = 100, L/dp = 40
    ax = axes[1, 1]
    for tag, color, label in (
        ("dbc", "#4477aa", "DBC (NoShifting)"),
        ("simple_mdbc", "#228833", "SimpleMDBC (NoShifting)"),
        ("updated_mdbc_shift", "#cc6677", "UpdatedMDBC + PlanarShifting"),
    ):
        data = load_history(f"re100_n40_c160_{tag}_integrated")
        ax.plot(data["time"], data["cd_viscous"], color=color, lw=0.95, label=label)
    ax.plot(ref["time"], ref["cd_viscous"], color="#111111", lw=1.35, ls="--", label="Official FD reference")
    ax.set(xlim=(0, 8), ylim=(-0.05, 1.25), title="Re = 100, c0 = 160 m/s: raw viscous drag (L/dp = 40)", xlabel="Time (s)", ylabel="Cd (viscous)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Spatial convergence and c0 = 160 m/s boundary sensitivity (Re = 100)", fontsize=11.5)
    fig.savefig(RESULTS / "resolution_and_c160.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    plot_traction_verification()
    plot_boundary_drag("pressure")
    plot_boundary_drag("viscous")
    plot_clearance_and_errors()
    plot_resolution_and_c160()
