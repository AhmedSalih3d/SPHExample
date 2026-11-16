"""Raw matched force histories; optional arguments: L/dp and sound speed."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("resolution",type=int,nargs="?",default=40,help="L/dp")
parser.add_argument("sound_speed",type=float,nargs="?",default=160.0,help="m/s")
args=parser.parse_args()
resolution,sound_speed=args.resolution,args.sound_speed
tag=f"motion_c{sound_speed:g}" + ("" if resolution==40 else f"_n{resolution:03d}")
basename=f"Re100_n{resolution:03d}.csv"
OUT=ROOT/"results"/tag
reference=np.loadtxt(ROOT/"reference/Force_Re100.dat",skiprows=2)
variants=[("hermite","Rounded-position Hermite","#d55e00"),
          ("integrated","Integrated-velocity motion","#0072b2")]
fig,axes=plt.subplots(2,2,figsize=(13,8))
histories={}
for name,label,color in variants:
    d=np.genfromtxt(OUT/name/basename,delimiter=",",names=True)
    histories[name]=d
    for column,component in enumerate(("pressure","viscous")):
        for ax in axes[:,column]:
            ax.plot(d["time"],d[f"cd_{component}"],color=color,lw=1.15,label=label)
for column,component in enumerate(("pressure","viscous")):
    for ax in axes[:,column]:
        ax.plot(reference[:,0],reference[:,column+1],color="black",lw=1.3,label="Official FD reference")
        ax.set(xlabel="Time [s]",ylabel=f"{component.capitalize()} drag coefficient")
        ax.grid(alpha=.2)
    axes[0,column].set(xlim=(0,8),title=f"{component.capitalize()} drag · full history")
    axes[1,column].set(xlim=(3,7),title=f"{component.capitalize()} drag · later interval")
    values=[reference[(reference[:,0]>=3)&(reference[:,0]<=7),column+1]]
    values.extend(d[f"cd_{component}"][(d["time"]>=3)&(d["time"]<=7)] for d in histories.values())
    lo,hi=min(v.min() for v in values),max(v.max() for v in values)
    axes[1,column].set_ylim(lo-.08*(hi-lo),hi+.08*(hi-lo))
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=9)
fig.suptitle(f"Matched motion comparison · c₀ = {sound_speed:.1f} m/s · original 10 × 5 m tank\n"
             f"Re = 100 · DBC · Δp = {1/resolution:g} m · CFL = 0.2 · δ = 0.1 · no sponge, cleaning or shifting",fontsize=12)
fig.tight_layout(rect=(0,.06,1,.92))
fig.savefig(OUT/"drag_comparison.png",dpi=180)
fig.savefig(OUT/"drag_comparison.pdf")
print(OUT/"drag_comparison.png")

if resolution != 40:
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for column,(name,label,_) in enumerate(variants):
        coarse=np.genfromtxt(ROOT/"results"/f"motion_c{sound_speed:g}"/name/"Re100_n040.csv",delimiter=",",names=True)
        fine=histories[name]
        for row,component in enumerate(("pressure","viscous")):
            ax=axes[row,column]
            ax.plot(coarse["time"],coarse[f"cd_{component}"],color="#757575",ls="--",lw=1.1,label="Δp = 0.025 m")
            ax.plot(fine["time"],fine[f"cd_{component}"],color="#0072b2",lw=1.1,label=f"Δp = {1/resolution:g} m")
            ax.plot(reference[:,0],reference[:,row+1],color="black",lw=1.3,label="Official FD reference")
            ax.set(xlim=(0,8),xlabel="Time [s]",ylabel=f"{component.capitalize()} drag coefficient",title=label)
            ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=9)
    fig.suptitle(f"Spatial refinement · c₀ = {sound_speed:.1f} m/s · original 10 × 5 m tank\n"
                 "Same CFL, h/Δp and diffusion parameter; refinement also reduces timestep and diffusion coefficient",fontsize=12)
    fig.tight_layout(rect=(0,.06,1,.92))
    fig.savefig(OUT/"resolution_comparison.png",dpi=180)
    fig.savefig(OUT/"resolution_comparison.pdf")
    print(OUT/"resolution_comparison.png")

if sound_speed != 160:
    baseline_tag="motion_c160" + ("" if resolution==40 else f"_n{resolution:03d}")
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for column,(name,label,_) in enumerate(variants):
        old=np.genfromtxt(ROOT/"results"/baseline_tag/name/basename,delimiter=",",names=True)
        new=histories[name]
        for row,component in enumerate(("pressure","viscous")):
            ax=axes[row,column]
            ax.plot(old["time"],old[f"cd_{component}"],color="#757575",ls="--",lw=1.15,label="c₀ = 160 m/s")
            ax.plot(new["time"],new[f"cd_{component}"],color="#0072b2",lw=1.15,label=f"c₀ = {sound_speed:g} m/s")
            ax.plot(reference[:,0],reference[:,row+1],color="black",lw=1.3,label="Official FD reference")
            ax.set(xlim=(0,8),xlabel="Time [s]",ylabel=f"{component.capitalize()} drag coefficient",title=label)
            ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=9)
    fig.suptitle(f"Sound-speed comparison · Δp = {1/resolution:g} m · original 10 × 5 m tank\n"
                 "Same CFL and diffusion parameter; increasing c₀ also changes timestep and density diffusion",fontsize=12)
    fig.tight_layout(rect=(0,.06,1,.92))
    fig.savefig(OUT/"sound_speed_comparison.png",dpi=180)
    fig.savefig(OUT/"sound_speed_comparison.pdf")
    print(OUT/"sound_speed_comparison.png")
