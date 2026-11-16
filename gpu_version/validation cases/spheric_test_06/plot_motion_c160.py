"""Raw matched force histories for the two motion interpolants at c0=160."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/motion_c160"
reference=np.loadtxt(ROOT/"reference/Force_Re100.dat",skiprows=2)
variants=[("hermite","Rounded-position Hermite","#d55e00"),
          ("integrated","Integrated-velocity motion","#0072b2")]
fig,axes=plt.subplots(2,2,figsize=(13,8))
histories={}
for name,label,color in variants:
    d=np.genfromtxt(OUT/name/"Re100_n040.csv",delimiter=",",names=True)
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
fig.suptitle("Matched motion comparison · c₀ = 160.0 m/s · original 10 × 5 m tank\n"
             "Re = 100 · DBC · Δp = 0.025 m · CFL = 0.2 · δ = 0.1 · no sponge, cleaning or shifting",fontsize=12)
fig.tight_layout(rect=(0,.06,1,.92))
fig.savefig(OUT/"drag_comparison.png",dpi=180)
fig.savefig(OUT/"drag_comparison.pdf")
print(OUT/"drag_comparison.png")
