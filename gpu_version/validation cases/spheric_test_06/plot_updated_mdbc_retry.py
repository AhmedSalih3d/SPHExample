"""Raw loads, penetration checks, and final near-body fluid positions."""
from pathlib import Path
import argparse
import csv
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("--paper-audit",action="store_true")
tag="updated_mdbc_paper_audit_c160" if parser.parse_args().paper_audit else "updated_mdbc_retry_c160"
OUT=ROOT/"results"/tag
WORK=ROOT/"runs"/tag
variants=[("dbc","DBC · no shifting","#757575"),
          ("updated","UpdatedMDBC · no shifting","#d55e00"),
          ("updated_shifting","UpdatedMDBC · planar shifting","#0072b2")]
reference=np.loadtxt(ROOT/"reference/Force_Re100.dat",skiprows=2)
fig,axes=plt.subplots(3,2,figsize=(13,11))
for name,label,color in variants:
    d=np.genfromtxt(OUT/name/"native.csv",delimiter=",",names=True)
    s=np.genfromtxt(OUT/name/"native_state.csv",delimiter=",",names=True)
    for column,component in enumerate(("pressure","viscous")):
        rows=(0,1) if name=="dbc" else (0,) if name=="updated_shifting" else (1,)
        for row in rows:
            axes[row,column].plot(d["time"],d[f"cd_{component}"],color=color,lw=1.15,label=label)
    axes[2,0].plot(s["time"],1000*np.maximum(0,-s["minimum_clearance"]),color=color,label=label)
    axes[2,1].plot(s["time"],100*s["density_relative_max"],color=color,label=label)
for column,component in enumerate(("pressure","viscous")):
    for row in (0,1):
        axes[row,column].plot(reference[:,0],reference[:,column+1],color="black",lw=1.3,label="Official FD reference")
        axes[row,column].set(ylabel=f"{component.capitalize()} drag coefficient")
        axes[row,column].legend(fontsize=8)
    axes[0,column].set_title(f"{component.capitalize()} · DBC versus UpdatedMDBC + shifting")
    axes[1,column].set_title(f"{component.capitalize()} · UpdatedMDBC without shifting (separate scale)")
axes[2,0].set(ylabel="Maximum solid penetration [mm]",title="Impermeability check · symlog scale",yscale="symlog")
axes[2,1].set(ylabel="Maximum |ρ/ρ₀ − 1| [%]",title="Fluid density departure")
axes[2,0].legend(fontsize=8);axes[2,1].legend(fontsize=8)
for ax in axes.flat:
    ax.set(xlim=(0,8),xlabel="Time [s]");ax.grid(alpha=.2)
fig.suptitle("Fresh UpdatedMDBC retry · c₀ = 160 m/s · integrated motion · original tank\n"
             "Δp = 0.025 m · Re = 100 · matched legacy timestep · no sponge or cleaning · raw forces",fontsize=12)
fig.tight_layout(rect=(0,0,1,.94))
fig.savefig(OUT/"drag_and_penetration.png",dpi=180)
fig.savefig(OUT/"drag_and_penetration.pdf")

fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharex=True,sharey=True)
counts=[]
for ax,(name,label,color) in zip(axes,variants):
    with h5py.File(WORK/name/"MovingSquare.vtkhdf","r") as f:
        root=f["VTKHDF/Particles"] if "VTKHDF/Particles" in f else f["VTKHDF"]
        time=float(np.asarray(root["Steps/Values"])[-1])
        offset=int(np.asarray(root["Steps/PointOffsets"])[-1])
        count=int(np.asarray(root["NumberOfPoints"])[-1])
        points=np.asarray(root["Points"][offset:offset+count,:])[:,:2]
        markers=np.asarray(root["PointData/GroupMarker"][offset:offset+count])
    center=points[markers==3].mean(axis=0)
    fluid=points[markers==2]-center
    inside=np.all(np.abs(fluid)<.5,axis=1)
    near=np.all(np.abs(fluid)<1.0,axis=1)
    ax.add_patch(Rectangle((-.5,-.5),1,1,facecolor="#eeeeee",edgecolor="black",lw=1,zorder=0))
    ax.scatter(fluid[near & ~inside,0],fluid[near & ~inside,1],s=2,color="#0072b2",label="Fluid outside body")
    ax.scatter(fluid[inside,0],fluid[inside,1],s=3,color="#d55e00",label="Fluid inside nominal body")
    ax.set(title=f"{label}\n{inside.sum()} fluid centres inside body",xlabel="x − body centre [m]",
           xlim=(-1,1),ylim=(-1,1),aspect="equal")
    counts.append(dict(variant=name,time=time,fluid_centres_inside_body=int(inside.sum())))
axes[0].set_ylabel("y − body centre [m]")
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc="lower center",ncol=2,fontsize=9)
fig.suptitle("Final fluid positions in body coordinates · shaded square is the nominal solid",fontsize=12)
fig.tight_layout(rect=(0,.1,1,.92))
fig.savefig(OUT/"final_body_particles.png",dpi=180)
with (OUT/"final_body_particles.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(counts[0]));writer.writeheader();writer.writerows(counts)
print(counts)
print(OUT/"drag_and_penetration.png")
print(OUT/"final_body_particles.png")

if tag == "updated_mdbc_paper_audit_c160":
    old_root=ROOT/"results/updated_mdbc_retry_c160"
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for column,(name,title) in enumerate((("updated","UpdatedMDBC · no shifting"),
                                         ("updated_shifting","UpdatedMDBC · planar shifting"))):
        for directory,label,color in ((old_root,"Before correction","#d55e00"),(OUT,"Physical wall velocity in continuity","#0072b2")):
            d=np.genfromtxt(directory/name/"native.csv",delimiter=",",names=True)
            s=np.genfromtxt(directory/name/"native_state.csv",delimiter=",",names=True)
            axes[0,column].plot(d["time"],d["cd_viscous"],lw=1.1,color=color,label=label)
            axes[1,column].plot(s["time"],1000*np.maximum(0,-s["minimum_clearance"]),lw=1.1,color=color,label=label)
        axes[0,column].plot(reference[:,0],reference[:,2],color="black",lw=1.3,label="Official FD reference")
        axes[0,column].set(title=title,ylabel="Viscous drag coefficient")
        axes[1,column].set(ylabel="Maximum solid penetration [mm]",yscale="symlog")
    for ax in axes.flat:
        ax.set(xlim=(0,8),xlabel="Time [s]");ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=9)
    fig.suptitle("Confirmed implementation correction · identical 160 m/s moving-square setup\n"
                 "Continuity uses physical wall velocity; no collision or particle-pushing rule added",fontsize=12)
    fig.tight_layout(rect=(0,.06,1,.92))
    fig.savefig(OUT/"continuity_fix_comparison.png",dpi=180)
    fig.savefig(OUT/"continuity_fix_comparison.pdf")
