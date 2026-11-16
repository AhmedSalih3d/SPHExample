"""Raw all-Reynolds drag, stock DualSPHysics comparison, and 1 Pa control."""
from pathlib import Path
import csv
import argparse
import tomllib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/coefh_reynolds"
REYNOLDS=(50,100,150)
parser=argparse.ArgumentParser()
parser.add_argument("--native-only",action="store_true")
args=parser.parse_args()
data={};state={};meta={};reference={}
independent=[]
for re in REYNOLDS:
    values=[]
    for line in (ROOT/f"reference/Force_Re{re:03d}.dat").read_text().splitlines():
        try:
            row=[float(x) for x in line.split()]
        except ValueError:
            continue
        if len(row)==3:values.append(row)
    reference[re]=np.array(values)
    for variant in (("bg0","bg1") if args.native_only else ("bg0","bg1","dual")):
        directory=OUT/f"re{re}_{variant}"
        stem="dual" if variant=="dual" else f"Re{re:03d}_n040"
        d=np.genfromtxt(directory/(stem+".csv"),delimiter=",",names=True)
        m=tomllib.loads((directory/(stem+".toml")).read_text())
        assert m["frames"]==401 and m["sound_speed"]==160 and m["reynolds"]==re
        assert len(d)==401 and d["time"][0]==0 and d["time"][-1]>=8-1e-6
        assert all(np.isfinite(d[k]).all() for k in d.dtype.names)
        data[re,variant]=d;meta[re,variant]=m
        ref=reference[re]
        t=np.unique(np.r_[.1,8,d["time"][(d["time"]>.1)&(d["time"]<8)],ref[(ref[:,0]>.1)&(ref[:,0]<8),0]])
        midpoint=(t[:-1]+t[1:])/2;half=(t[1:]-t[:-1])/2
        x=np.r_[midpoint-half/np.sqrt(3),midpoint+half/np.sqrt(3)];weights=np.r_[half,half]
        metrics=dict(reynolds=re,variant=variant)
        for j,k in enumerate(("pressure","viscous"),1):
            actual=np.interp(x,d["time"],d[f"cd_{k}"]);target=np.interp(x,ref[:,0],ref[:,j])
            metrics[k+"_error"]=np.sqrt(np.sum(weights*(actual-target)**2)/np.sum(weights*target**2))
        independent.append(metrics)
        if variant!="dual":
            s=np.genfromtxt(directory/(stem+"_state.csv"),delimiter=",",names=True)
            assert m["background_pressure"]==(variant=="bg1")
            assert abs(s["pressure_mean"][0]-(variant=="bg1"))<1e-8
            assert len(s)==401 and np.all(s["fluid_particles"]==78400) and np.all(s["square_particles"]==1600)
            assert all(np.isfinite(s[k]).all() for k in s.dtype.names)
            state[re,variant]=s

with (OUT/"independent_metrics.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(independent[0]));writer.writeheader();writer.writerows(independent)
summary=OUT/("native_summary.csv" if args.native_only else "summary.csv")
if summary.exists():
    expected={(int(r["reynolds"]),r["variant"]):r for r in csv.DictReader(summary.open())}
    for row in independent:
        for k in ("pressure_error","viscous_error"):
            assert abs(row[k]-float(expected[row["reynolds"],row["variant"]][k]))<1e-10
    print("All force errors match independent two-point Gauss quadrature.")

def comparison(variants,filename,title):
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for column,re in enumerate(REYNOLDS):
        axes[0,column].set_title(f"Re = {re} · ν = {1/re:.6g} m²/s")
        for variant,label,color in variants:
            d=data[re,variant]
            for row,k in enumerate(("pressure","viscous")):
                axes[row,column].plot(d["time"],d[f"cd_{k}"],lw=1.05,color=color,label=label)
        ref=reference[re]
        for row,k in enumerate(("pressure","viscous")):
            ax=axes[row,column]
            ax.plot(ref[:,0],ref[:,row+1],color="black",lw=1.2,label="Official FD reference")
            ax.set(xlim=(0,8),xlabel="Time [s]",ylabel=f"{k.capitalize()} drag coefficient");ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=len(labels),fontsize=9)
    fig.suptitle(title+"\ncoefh=1.2 · c₀=160 m/s · dp=.025 m · original tank · integrated motion · DBC · raw forces",fontsize=12)
    fig.tight_layout(rect=(0,.06,1,.92))
    fig.savefig(OUT/(filename+".png"),dpi=180);fig.savefig(OUT/(filename+".pdf"))

comparison((("bg0","Julia · background 0 Pa","#d55e00"),),"all_reynolds","All Reynolds numbers · Julia GPU")
if not args.native_only:
    comparison((("bg0","Julia GPU","#d55e00"),("dual","DualSPHysics 5.4 GPU","#0072b2")),
               "dual_comparison","Matched physical settings · Julia versus stock DualSPHysics · background 0 Pa")
comparison((("bg0","Background 0 Pa","#0072b2"),("bg1","Background 1 Pa","#d55e00")),
           "background_comparison","Julia GPU · impact of BackgroundPressure = 1 Pa")

fig,axes=plt.subplots(2,3,figsize=(15,7.5))
changes=[]
for column,re in enumerate(REYNOLDS):
    a,b=data[re,"bg0"],data[re,"bg1"]
    assert np.array_equal(a["time"],b["time"])
    row=dict(reynolds=re)
    for j,k in enumerate(("pressure","viscous")):
        difference=b[f"cd_{k}"]-a[f"cd_{k}"]
        ax=axes[j,column];ax.plot(a["time"],difference,color="#0072b2",lw=1)
        ax.axhline(0,color="black",lw=.8);ax.set(xlim=(0,8),xlabel="Time [s]",ylabel=f"Δ {k} Cd (1 Pa − 0 Pa)");ax.grid(alpha=.2)
        row[k+"_difference_rms"]=np.sqrt(np.mean(difference**2))
        row[k+"_difference_abs_max"]=np.max(np.abs(difference))
    axes[0,column].set_title(f"Re = {re}")
    for variant in ("bg0","bg1"):
        row[f"{variant}_maximum_penetration_mm"]=1000*max(0,-state[re,variant]["minimum_clearance"].min())
        row[f"{variant}_density_relative_max"]=state[re,variant]["density_relative_max"].max()
    changes.append(row)
fig.suptitle("Background-pressure sensitivity · raw differences · all other Julia settings fixed\n"
             "These differences include altered trajectories; they are not the direct force of a uniform 1 Pa offset",fontsize=12)
fig.tight_layout(rect=(0,0,1,.91));fig.savefig(OUT/"background_difference.png",dpi=180);fig.savefig(OUT/"background_difference.pdf")
with (OUT/"background_changes.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(changes[0]));writer.writeheader();writer.writerows(changes)
print(changes)

fig,axes=plt.subplots(2,3,figsize=(13,8.5))
counts=[]
for row,variant in enumerate(("bg0","bg1")):
    for column,re in enumerate(REYNOLDS):
        path=ROOT/f"runs/coefh_reynolds/re{re}_{variant}/MovingSquare.vtkhdf"
        if re==100 and variant=="bg0":path=ROOT/"runs/coefh/Re100_n040/MovingSquare.vtkhdf"
        with h5py.File(path,"r") as f:
            root=f["VTKHDF/Particles"] if "VTKHDF/Particles" in f else f["VTKHDF"]
            offset=int(root["Steps/PointOffsets"][-1]);n=int(root["NumberOfPoints"][-1])
            assert float(root["Steps/Values"][-1])==8 and n==86100
            points=np.asarray(root["Points"][offset:offset+n,:])[:,:2]
            marker=np.asarray(root["PointData/GroupMarker"][offset:offset+n])
        assert np.sum(marker==2)==78400 and np.sum(marker==3)==1600
        center=points[marker==3].mean(axis=0);fluid=points[marker==2]-center
        inside=np.all(np.abs(fluid)<.5,axis=1);near=np.all(np.abs(fluid)<1,axis=1)
        ax=axes[row,column];ax.add_patch(Rectangle((-.5,-.5),1,1,facecolor="#eee",edgecolor="black"))
        ax.scatter(fluid[near&~inside,0],fluid[near&~inside,1],s=1.5,color="#0072b2")
        ax.scatter(fluid[inside,0],fluid[inside,1],s=4,color="#d55e00")
        ax.set(xlim=(-1,1),ylim=(-1,1),aspect="equal",xlabel="x − body centre [m]",ylabel="y − body centre [m]",
               title=f"Re={re} · background {row} Pa\n{inside.sum()} fluid centres inside at 8 s")
        counts.append(dict(reynolds=re,variant=variant,inside_final=int(inside.sum())))
fig.tight_layout();fig.savefig(OUT/"native_final_particles.png",dpi=180)
with (OUT/"native_final_particles.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(counts[0]));writer.writeheader();writer.writerows(counts)
print(counts)
