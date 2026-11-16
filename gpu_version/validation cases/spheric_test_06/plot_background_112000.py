"""Distinguish an additive 112000 Pa offset from the FAQ's EOS stiffness B."""
from pathlib import Path
import csv
import tomllib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/background_112000"
ref=np.loadtxt(ROOT/"reference/Force_Re100.dat",skiprows=2)
entries=(
    ("julia_c160_bg0",ROOT/"results/coefh","Re100_n040"),
    ("julia_c160_bg112000",OUT/"offset112000_c160","Re100_n040"),
    ("julia_c28_B112000",OUT/"stiffness112000_c28","Re100_n040"),
    ("dsph_c160_bg0",ROOT/"results/coefh_reynolds/re100_dual","dual"),
    ("dsph_c28_B112000",OUT/"dual_stiffness112000_c28","dual"))
data={};state={}
for name,directory,stem in entries:
    d=np.genfromtxt(directory/(stem+".csv"),delimiter=",",names=True)
    m=tomllib.loads((directory/(stem+".toml")).read_text())
    assert len(d)==m["frames"]==401 and d["time"][0]==0 and d["time"][-1]>=8
    assert all(np.isfinite(d[k]).all() for k in d.dtype.names)
    data[name]=d
    if stem!="dual":
        s=np.genfromtxt(directory/(stem+"_state.csv"),delimiter=",",names=True)
        assert np.all(s["fluid_particles"]==78400) and np.all(s["square_particles"]==1600)
        assert all(np.isfinite(s[k]).all() for k in s.dtype.names)
        assert abs(s["pressure_mean"][0]-m["background_pressure"])<1e-6
        state[name]=s

fig,axes=plt.subplots(2,2,figsize=(13,8))
for name,label,color in (("julia_c160_bg0","Background 0 Pa","#0072b2"),
                         ("julia_c160_bg112000","Background 112000 Pa","#d55e00")):
    d,s=data[name],state[name]
    for column,k in enumerate(("pressure","viscous")):
        axes[0,column].plot(d["time"],d[f"cd_{k}"],lw=1.1,color=color,label=label)
    axes[1,0].plot(s["time"],1000*np.maximum(0,-s["minimum_clearance"]),lw=1.1,color=color,label=label)
    axes[1,1].plot(s["time"],100*s["density_relative_max"],lw=1.1,color=color,label=label)
for j,k in enumerate(("pressure","viscous")):
    axes[0,j].plot(ref[:,0],ref[:,j+1],color="black",lw=1.2,label="Official FD reference")
    axes[0,j].set_ylabel(f"{k.capitalize()} drag coefficient")
axes[1,0].set_ylabel("Maximum penetration [mm]")
axes[1,1].set_ylabel("Maximum fluid density departure [%]")
for ax in axes.flat:ax.set(xlim=(0,8),xlabel="Time [s]");ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle("Literal background-pressure test · Re100 · Julia · c₀=160 m/s\n"
             "p = B[(ρ/ρ₀)⁷ − 1] + p_background · B=3657143 Pa unchanged · raw forces",fontsize=12)
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/"offset_comparison.png",dpi=180);fig.savefig(OUT/"offset_comparison.pdf")

fig,axes=plt.subplots(2,2,figsize=(13,8))
for column,(solver,title) in enumerate((("julia","Julia GPU"),("dsph","Stock DSPH 5.4 GPU"))):
    for suffix,label,color in (("c160_bg0","c₀=160 · B=3657143 Pa","#0072b2"),
                               ("c28_B112000","c₀=28 · B=112000 Pa","#d55e00")):
        d=data[solver+"_"+suffix]
        for row,k in enumerate(("pressure","viscous")):
            axes[row,column].plot(d["time"],d[f"cd_{k}"],lw=1.1,color=color,label=label)
    axes[0,column].set_title(title)
    for row,k in enumerate(("pressure","viscous")):
        axes[row,column].plot(ref[:,0],ref[:,row+1],color="black",lw=1.2,label="Official FD reference")
        axes[row,column].set_ylabel(f"{k.capitalize()} drag coefficient")
for ax in axes.flat:ax.set(xlim=(0,8),xlabel="Time [s]");ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle("DSPH FAQ interpretation · EOS stiffness B=112000 Pa · Re100\n"
             "Additive background pressure is zero in every curve · coefh=1.2 · integrated motion · raw forces",fontsize=12)
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/"stiffness_comparison.png",dpi=180);fig.savefig(OUT/"stiffness_comparison.pdf")

# Independent two-point Gauss quadrature of the piecewise-linear residual.
expected={r["name"]:r for r in csv.DictReader((OUT/"summary.csv").open())}
for name,d in data.items():
    t=np.unique(np.r_[.1,8,d["time"][(d["time"]>.1)&(d["time"]<8)],ref[(ref[:,0]>.1)&(ref[:,0]<8),0]])
    mid=(t[:-1]+t[1:])/2;half=np.diff(t)/2
    x=np.r_[mid-half/np.sqrt(3),mid+half/np.sqrt(3)];w=np.r_[half,half]
    for j,k in enumerate(("pressure","viscous"),1):
        a=np.interp(x,d["time"],d[f"cd_{k}"]);b=np.interp(x,ref[:,0],ref[:,j])
        error=np.sqrt(np.sum(w*(a-b)**2)/np.sum(w*b*b))
        assert abs(error-float(expected[name][k+"_error"]))<1e-10
print("All ten force errors independently verified.")
