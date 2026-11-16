"""Audit saved smoothing-length histories; plot raw forces and matched controls."""
from pathlib import Path
import csv
import tomllib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"results/coefh_validation"
OUT.mkdir(exist_ok=True)
ref=np.loadtxt(ROOT/"reference/Force_Re100.dat",skiprows=2)
entries=[("previous",ROOT/"results/updated_mdbc_paper_audit_c160/dbc","native","Previous coefh=1 · 4 wall layers"),
         ("user",ROOT/"results/coefh","Re100_n040","Your coefh=1.2 · 5 wall layers")]
for name,label in (("coef1_cfl02","coefh=1 · CFL=.2 · 5 layers"),
                   ("coef12_cfl02","coefh=1.2 · CFL=.2 · reproduction"),
                   ("coef12_matched_dt","coefh=1.2 · CFL=.1667 · matched nominal dt"),
                   ("coef12_cfl01","coefh=1.2 · CFL=.1")):
    directory=OUT/name
    if (directory/"Re100_n040.toml").exists():
        entries.append((name,directory,"Re100_n040",label))

def square_integral(t,z):
    return np.sum(np.diff(t)*(z[:-1]**2+z[:-1]*z[1:]+z[1:]**2)/3)

def average(t,z,a,b):
    x=np.r_[a,t[(t>a)&(t<b)],b]
    return np.trapz(np.interp(x,t,z),x)/(b-a)

data={};states={};rows=[]
for name,directory,stem,label in entries:
    d=np.genfromtxt(directory/(stem+".csv"),delimiter=",",names=True)
    s=np.genfromtxt(directory/(stem+"_state.csv"),delimiter=",",names=True)
    m=tomllib.loads((directory/(stem+".toml")).read_text())
    assert len(d)==len(s)==m["frames"]==401 and d["time"][-1]==8
    assert all(np.isfinite(d[k]).all() for k in d.dtype.names)
    assert all(np.isfinite(s[k]).all() for k in s.dtype.names)
    assert np.all(s["fluid_particles"]==78400) and np.all(s["square_particles"]==1600)
    data[name]=d;states[name]=s
    row=dict(name=name,pressure_error=0.,viscous_error=0.,pressure_noise=0.,
             pressure_mean_3_7=average(d["time"],d["cd_pressure"],3,7),
             viscous_mean_3_7=average(d["time"],d["cd_viscous"],3,7),
             maximum_penetration_mm=1000*max(0,-s["minimum_clearance"].min()),
             density_relative_max=s["density_relative_max"].max(),steps=m["steps"])
    t=np.unique(np.r_[.1,8,d["time"][(d["time"]>.1)&(d["time"]<8)],ref[(ref[:,0]>.1)&(ref[:,0]<8),0]])
    for j,k in enumerate(("pressure","viscous"),1):
        z=np.interp(t,d["time"],d[f"cd_{k}"]);r=np.interp(t,ref[:,0],ref[:,j])
        row[k+"_error"]=np.sqrt(square_integral(t,z-r)/square_integral(t,r))
    nodes=np.arange(15,386)*.02
    residual=np.interp(nodes,d["time"],d["cd_pressure"])-np.interp(nodes,ref[:,0],ref[:,1])
    selected=np.arange(75,326)*.02
    noise=[np.interp(x,nodes,residual)-average(nodes,residual,x-.25,x+.25) for x in selected]
    row["pressure_noise"]=np.sqrt(np.mean(np.square(noise)))
    rows.append(row)

def figure(names,filename,title):
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for name in names:
        label=next(e[3] for e in entries if e[0]==name)
        d,s=data[name],states[name]
        for ax in axes[0,:]:ax.plot(d["time"],d["cd_pressure"],lw=1.1,label=label)
        axes[1,0].plot(d["time"],d["cd_viscous"],lw=1.1,label=label)
        axes[1,1].plot(s["time"],1000*np.maximum(0,-s["minimum_clearance"]),lw=1.1,label=label)
    for ax in axes[0,:]:
        ax.plot(ref[:,0],ref[:,1],color="black",lw=1.2,label="Official FD reference")
        ax.set_ylabel("Pressure drag coefficient")
    axes[1,0].plot(ref[:,0],ref[:,2],color="black",lw=1.2,label="Official FD reference")
    axes[1,0].set_ylabel("Viscous drag coefficient")
    axes[1,1].set_ylabel("Maximum penetration [mm]")
    axes[0,1].set_title("Central pressure window · raw curves")
    for ax in axes.flat:ax.set(xlim=(0,8),xlabel="Time [s]");ax.grid(alpha=.2)
    axes[0,1].set(xlim=(1.5,6.5),ylim=(1.3,2.7))
    axes[0,0].legend(fontsize=7)
    fig.suptitle(title+"\nc₀=160 m/s · original tank · dp=.025 m · integrated motion · DBC · no cleaning or sponge",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.92))
    fig.savefig(OUT/(filename+".png"),dpi=180);fig.savefig(OUT/(filename+".pdf"))

figure(["previous","user"],"user_comparison","Smoothing-length change · supplied result versus previous control")
controls=[n for n,_,_,_ in entries if n not in ("previous","user")]
if controls:figure(controls,"matched_controls","Independent smoothing-length and timestep controls · fixed five wall layers")

counts=[]
particle_files=[("Your coefh=1.2",ROOT/"runs/coefh/Re100_n040/MovingSquare.vtkhdf")]
if (ROOT/"runs/coefh_validation/coef12_cfl01/MovingSquare.vtkhdf").exists() and "coef12_cfl01" in data:
    particle_files.append(("coefh=1.2 · CFL=.1",ROOT/"runs/coefh_validation/coef12_cfl01/MovingSquare.vtkhdf"))
fig,axes=plt.subplots(1,len(particle_files),figsize=(5*len(particle_files),5),squeeze=False)
for ax,(label,path) in zip(axes.flat,particle_files):
    with h5py.File(path,"r") as f:
        root=f["VTKHDF/Particles"] if "VTKHDF/Particles" in f else f["VTKHDF"]
        offset=int(root["Steps/PointOffsets"][-1]);n=int(root["NumberOfPoints"][-1])
        points=np.asarray(root["Points"][offset:offset+n,:])[:,:2]
        marker=np.asarray(root["PointData/GroupMarker"][offset:offset+n])
        assert float(root["Steps/Values"][-1])==8 and n==86100
    center=points[marker==3].mean(axis=0);fluid=points[marker==2]-center
    inside=np.all(np.abs(fluid)<.5,axis=1);near=np.all(np.abs(fluid)<1,axis=1)
    ax.add_patch(Rectangle((-.5,-.5),1,1,facecolor="#eee",edgecolor="black"))
    ax.scatter(fluid[near&~inside,0],fluid[near&~inside,1],s=2,color="#0072b2")
    ax.scatter(fluid[inside,0],fluid[inside,1],s=4,color="#d55e00")
    ax.set(xlim=(-1,1),ylim=(-1,1),aspect="equal",xlabel="x − body centre [m]",ylabel="y − body centre [m]",
           title=f"{label}\n{inside.sum()} fluid centres inside at 8 s")
    counts.append(dict(name=label,inside_final=int(inside.sum())))
fig.tight_layout();fig.savefig(OUT/"final_body_particles.png",dpi=180)
for filename,records in (("summary.csv",rows),("final_body_particles.csv",counts)):
    with (OUT/filename).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
print(rows)
print(counts)
