"""Compare DBC, SimpleMDBC and corrected UpdatedMDBC at the improved coefh."""
from pathlib import Path
import csv
import tomllib
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/coefh_mdbc"
REYNOLDS=(50,100,150)
MODES=(("dbc","DBC","#009e73"),("simple","SimpleMDBC","#d55e00"),("updated","UpdatedMDBC","#0072b2"))
data={};states={};references={};independent=[]
expected={(int(r["reynolds"]),r["boundary"]):r for r in csv.DictReader((OUT/"summary.csv").open())}
for re in REYNOLDS:
    values=[]
    for line in (ROOT/f"reference/Force_Re{re:03d}.dat").read_text().splitlines():
        try:row=[float(x) for x in line.split()]
        except ValueError:continue
        if len(row)==3:values.append(row)
    ref=references[re]=np.array(values)
    for mode,_,_ in MODES:
        directory=ROOT/f"results/coefh_reynolds/re{re}_bg0" if mode=="dbc" else OUT/f"re{re}_{mode}"
        stem=f"Re{re:03d}_n040" if mode=="dbc" else "native"
        d=np.genfromtxt(directory/(stem+".csv"),delimiter=",",names=True)
        s=np.genfromtxt(directory/(stem+"_state.csv"),delimiter=",",names=True)
        m=tomllib.loads((directory/(stem+".toml")).read_text())
        assert len(d)==len(s)==m["frames"]==401 and d["time"][-1]==8
        assert m["sound_speed"]==160 and m["background_pressure"]==0 and m["wall_layers"]==5
        assert abs(m["h"]-.04242640733718872)<1e-12 and not m["divergence_cleaning"]
        assert all(np.isfinite(d[k]).all() for k in d.dtype.names)
        assert all(np.isfinite(s[k]).all() for k in s.dtype.names)
        assert np.all(s["fluid_particles"]==78400) and np.all(s["square_particles"]==1600)
        data[re,mode]=d;states[re,mode]=s
        t=np.unique(np.r_[.1,8,d["time"][(d["time"]>.1)&(d["time"]<8)],ref[(ref[:,0]>.1)&(ref[:,0]<8),0]])
        mid=(t[:-1]+t[1:])/2;half=np.diff(t)/2
        x=np.r_[mid-half/np.sqrt(3),mid+half/np.sqrt(3)];w=np.r_[half,half]
        row=dict(reynolds=re,boundary=mode)
        for j,k in enumerate(("pressure","viscous"),1):
            a=np.interp(x,d["time"],d[f"cd_{k}"]);b=np.interp(x,ref[:,0],ref[:,j])
            error=np.sqrt(np.sum(w*(a-b)**2)/np.sum(w*b*b))
            assert abs(error-float(expected[re,mode][k+"_error"]))<1e-10
            row[k+"_error"]=error
        independent.append(row)

fig,axes=plt.subplots(2,3,figsize=(15,8))
for column,re in enumerate(REYNOLDS):
    axes[0,column].set_title(f"Re = {re}")
    for mode,label,color in MODES:
        d=data[re,mode]
        for row,k in enumerate(("pressure","viscous")):
            axes[row,column].plot(d["time"],d[f"cd_{k}"],lw=1.1,color=color,label=label)
    for row,k in enumerate(("pressure","viscous")):
        ax=axes[row,column];ref=references[re]
        ax.plot(ref[:,0],ref[:,row+1],color="black",lw=1.2,label="Official FD reference")
        ax.set(xlim=(0,8),xlabel="Time [s]",ylabel=f"{k.capitalize()} drag coefficient");ax.grid(alpha=.2)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc="lower center",ncol=4,fontsize=9)
fig.suptitle("DBC, SimpleMDBC and corrected UpdatedMDBC · all Reynolds numbers\n"
             "coefh=1.2 · c₀=160 m/s · original tank and spacing · integrated motion · no shifting, cleaning or sponge",fontsize=12)
fig.tight_layout(rect=(0,.06,1,.92));fig.savefig(OUT/"drag_comparison.png",dpi=180);fig.savefig(OUT/"drag_comparison.pdf")

fig,axes=plt.subplots(3,3,figsize=(15,11))
for column,re in enumerate(REYNOLDS):
    axes[0,column].set_title(f"Re = {re}")
    for mode,label,color in MODES:
        d,s=data[re,mode],states[re,mode]
        axes[0,column].plot(d["time"],d["cd_pressure"],lw=1,color=color,label=label)
        axes[1,column].plot(s["time"],1000*np.maximum(0,-s["minimum_clearance"]),lw=1,color=color,label=label)
        axes[2,column].plot(s["time"],100*s["density_relative_max"],lw=1,color=color,label=label)
    ref=references[re];axes[0,column].plot(ref[:,0],ref[:,1],color="black",lw=1.2,label="Official FD reference")
for row,label in enumerate(("Pressure drag coefficient","Maximum penetration [mm]","Maximum fluid density departure [%]")):
    for ax in axes[row,:]:
        ax.set(xlim=(0,8),xlabel="Time [s]",ylabel=label);ax.grid(alpha=.2)
for column,re in enumerate(REYNOLDS):
    visible=[]
    for mode,_,_ in MODES:
        d=data[re,mode];mask=(d["time"]>=1.5)&(d["time"]<=6.5)
        visible.extend(d["cd_pressure"][mask])
    ref=references[re];visible.extend(ref[(ref[:,0]>=1.5)&(ref[:,0]<=6.5),1])
    lo,hi=min(visible),max(visible);margin=.1*(hi-lo)+.02
    axes[0,column].set(xlim=(1.5,6.5),ylim=(lo-margin,hi+margin))
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",ncol=4,fontsize=9)
fig.suptitle("Central pressure window and state checks · raw curves\n"
             "Maximum penetration includes transient events; density is a separate validity check",fontsize=12)
fig.tight_layout(rect=(0,.06,1,.92));fig.savefig(OUT/"state_comparison.png",dpi=180);fig.savefig(OUT/"state_comparison.pdf")

fig,axes=plt.subplots(3,3,figsize=(12,12))
counts=[]
initial_reference={}
for row,(mode,label,_) in enumerate(MODES):
    for column,re in enumerate(REYNOLDS):
        path=ROOT/f"runs/coefh_mdbc/re{re}_{mode}/MovingSquare.vtkhdf"
        if mode=="dbc":
            path=ROOT/("runs/coefh/Re100_n040/MovingSquare.vtkhdf" if re==100 else f"runs/coefh_reynolds/re{re}_bg0/MovingSquare.vtkhdf")
        with h5py.File(path,"r") as f:
            root=f["VTKHDF/Particles"] if "VTKHDF/Particles" in f else f["VTKHDF"]
            n0=int(root["NumberOfPoints"][0]);p0=np.asarray(root["Points"][:n0,:])[:,:2]
            sorted_initial=p0[np.lexsort((p0[:,1],p0[:,0]))]
            if mode=="dbc":initial_reference[re]=sorted_initial
            else:assert np.array_equal(sorted_initial,initial_reference[re]),"Initial particle lattice differs"
            offset=int(root["Steps/PointOffsets"][-1]);n=int(root["NumberOfPoints"][-1])
            assert float(root["Steps/Values"][-1])==8 and n==86100
            points=np.asarray(root["Points"][offset:offset+n,:])[:,:2]
            marker=np.asarray(root["PointData/GroupMarker"][offset:offset+n])
        assert np.sum(marker==2)==78400 and np.sum(marker==3)==1600
        center=points[marker==3].mean(axis=0);fluid=points[marker==2]-center
        inside=np.all(np.abs(fluid)<.5,axis=1);near=np.all(np.abs(fluid)<1,axis=1)
        ax=axes[row,column];ax.add_patch(Rectangle((-.5,-.5),1,1,facecolor="#eee",edgecolor="black"))
        ax.scatter(fluid[near&~inside,0],fluid[near&~inside,1],s=1.5,color="#0072b2")
        ax.scatter(fluid[inside,0],fluid[inside,1],s=3,color="#d55e00")
        ax.set(xlim=(-1,1),ylim=(-1,1),aspect="equal",xlabel="x − body centre [m]",ylabel="y − body centre [m]",
               title=f"{label} · Re={re}\n{inside.sum()} fluid centres inside at 8 s")
        counts.append(dict(reynolds=re,boundary=mode,inside_final=int(inside.sum())))
fig.tight_layout();fig.savefig(OUT/"final_body_particles.png",dpi=180)
for filename,records in (("independent_metrics.csv",independent),("final_body_particles.csv",counts)):
    with (OUT/filename).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
print("All 18 drag errors independently verified.")
print("All initial particle lattices exactly match DBC.")
print(counts)
