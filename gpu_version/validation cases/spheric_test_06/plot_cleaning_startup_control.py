"""Compare raw startup controls without extrapolating short runs to eight seconds."""
from pathlib import Path
import csv
import tomllib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/cleaning_startup_c160"
reference = np.loadtxt(ROOT / "reference/Force_Re100.dat", skiprows=2)
fig, axes = plt.subplots(3, 2, figsize=(14, 11))
rows = []

def integral_square(t, z):
    return np.sum(np.diff(t) * (z[:-1]**2 + z[:-1]*z[1:] + z[1:]**2) / 3)

def add(directory, name, label, color, column, style="-", draw=True):
    d = np.genfromtxt(directory / "native.csv", delimiter=",", names=True)
    s = np.genfromtxt(directory / "native_state.csv", delimiter=",", names=True)
    meta = tomllib.loads((directory / "native.toml").read_text())
    assert meta["sound_speed"] == 160 and meta["particles"] == 84864
    assert all(np.isfinite(d[n]).all() for n in d.dtype.names)
    assert d["time"][-1] >= 1.6
    t = np.unique(np.r_[.1, 1.6, d["time"][(d["time"]>.1)&(d["time"]<1.6)],
                         reference[(reference[:,0]>.1)&(reference[:,0]<1.6),0]])
    row = dict(name=name, cfl=meta["cfl"], cleaning_kappa=meta.get("cleaning_kappa",0),
               cleaning_courant=meta.get("cleaning_courant",0),
               density_diffusion=meta.get("density_diffusion_coefficient",.1))
    for j, component in enumerate(("pressure", "viscous"), 1):
        actual = np.interp(t, d["time"], d[f"cd_{component}"])
        ref = np.interp(t, reference[:,0], reference[:,j])
        row[f"{component}_error"] = np.sqrt(integral_square(t,actual-ref)/integral_square(t,ref))
        if draw:
            axes[j-1,column].plot(d["time"], d[f"cd_{component}"], color=color, ls=style, lw=1, label=label)
    mask = d["time"] <= 1.6
    row.update(pressure_min=d["cd_pressure"][mask].min(), pressure_max=d["cd_pressure"][mask].max(),
               maximum_penetration_mm=1000*max(0,-s["minimum_clearance"][s["time"]<=1.6].min()))
    rows.append(row)
    if draw:
        axes[2,column].plot(s["time"],1000*np.maximum(0,-s["minimum_clearance"]), color=color, ls=style, lw=1)

for column, (boundary, short) in enumerate((("dbc", "dbc"), ("updated_shifting", "shifted"))):
    add(ROOT/"results/updated_mdbc_paper_audit_c160"/boundary, short+"_off_original",
        "Cleaning off · CFL .2", "#0072b2", column)
    add(ROOT/"results/updated_mdbc_paper_audit_c160_cleaning"/boundary, short+"_k10_original",
        "κ=10 · CFL .2 / clean .1", "#d55e00", column)
    add(OUT/(short+"_k10_small"), short+"_k10_small", "κ=10 · CFL .05 / clean .025", "#cc79a7", column)
    add(OUT/(short+"_k160_small"), short+"_k160_small", "κ=160 · CFL .05 / clean .025", "#009e73", column)
    if column == 0:
        add(OUT/"dbc_off_tiny", "dbc_off_tiny", "Cleaning off · CFL .025", "#666666", column, "--")
    axes[0,column].set_title("DBC" if column == 0 else "UpdatedMDBC + planar shifting")
    for j in range(2):
        axes[j,column].plot(reference[:,0],reference[:,j+1],color="black",lw=1.3,label="Official FD reference")
    axes[0,column].legend(fontsize=7)
for row, ylabel in enumerate(("Pressure drag coefficient", "Viscous drag coefficient", "Maximum penetration [mm]")):
    for ax in axes[row,:]:
        ax.set(xlim=(0,1.6),xlabel="Time [s]",ylabel=ylabel)
        ax.grid(alpha=.2)
fig.suptitle("Startup diagnosis · c₀=160 m/s · original tank and spacing · integrated motion\n"
             "Raw histories · smaller timestep and scalar-decay controls · no sponge",fontsize=12)
fig.tight_layout(rect=(0,0,1,.94))
fig.savefig(OUT/"startup_comparison.png",dpi=180)
fig.savefig(OUT/"startup_comparison.pdf")
for name in ("dbc_off_small", "dbc_delta2", "dbc_delta4"):
    add(OUT/name,name,name,"black",0,draw=False)
with (OUT/"startup_metrics.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for row in rows:
    print(row)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for directory, label, color in ((ROOT/"results/updated_mdbc_paper_audit_c160/dbc", "δ=0.1", "#0072b2"),
                               (OUT/"dbc_delta2", "δ=0.2", "#009e73"),
                               (OUT/"dbc_delta4", "δ=0.4", "#d55e00")):
    d=np.genfromtxt(directory/"native.csv",delimiter=",",names=True)
    s=np.genfromtxt(directory/"native_state.csv",delimiter=",",names=True)
    for j,component in enumerate(("pressure","viscous")):
        axes[j].plot(d["time"],d[f"cd_{component}"],lw=1.1,color=color,label=label)
    axes[2].plot(s["time"],1000*np.maximum(0,-s["minimum_clearance"]),lw=1.1,color=color,label=label)
for j,component in enumerate(("pressure","viscous")):
    axes[j].plot(reference[:,0],reference[:,j+1],color="black",lw=1.3,label="Official FD reference")
    axes[j].set_ylabel(f"{component.capitalize()} drag coefficient")
axes[2].set_ylabel("Maximum penetration [mm]")
for ax in axes:
    ax.set(xlim=(0,1.6),xlabel="Time [s]");ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle("Density diffusion controls · DBC · c₀=160 m/s · CFL=0.2 · cleaning off\n"
             "Original tank and spacing · integrated motion · raw startup forces",fontsize=12)
fig.tight_layout(rect=(0,0,1,.9))
fig.savefig(OUT/"diffusion_startup.png",dpi=180)
fig.savefig(OUT/"diffusion_startup.pdf")

if (OUT/"full_delta2/native.toml").exists():
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for directory,label,color in ((ROOT/"results/updated_mdbc_paper_audit_c160/dbc","δ=0.1","#0072b2"),
                                  (OUT/"full_delta2","δ=0.2","#d55e00")):
        d=np.genfromtxt(directory/"native.csv",delimiter=",",names=True)
        s=np.genfromtxt(directory/"native_state.csv",delimiter=",",names=True)
        for ax in axes[0,:]:
            ax.plot(d["time"],d["cd_pressure"],color=color,lw=1.1,label=label)
        axes[1,0].plot(d["time"],d["cd_viscous"],color=color,lw=1.1,label=label)
        axes[1,1].plot(s["time"],1000*np.maximum(0,-s["minimum_clearance"]),color=color,lw=1.1,label=label)
    for ax in axes[0,:]:
        ax.plot(reference[:,0],reference[:,1],color="black",lw=1.3,label="Official FD reference")
        ax.set_ylabel("Pressure drag coefficient")
    axes[1,0].plot(reference[:,0],reference[:,2],color="black",lw=1.3,label="Official FD reference")
    axes[1,0].set_ylabel("Viscous drag coefficient")
    axes[1,1].set_ylabel("Maximum penetration [mm]")
    for ax in axes.flat:
        ax.set(xlim=(0,8),xlabel="Time [s]");ax.grid(alpha=.2)
    axes[0,1].set(xlim=(1.5,6.5),ylim=(1.3,2.8),title="Central pressure window · raw curves")
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Full-history diffusion control · DBC · c₀=160 m/s · CFL=0.2\n"
                 "Original tank and spacing · integrated motion · divergence cleaning off",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92))
    fig.savefig(OUT/"diffusion_full.png",dpi=180)
    fig.savefig(OUT/"diffusion_full.pdf")
