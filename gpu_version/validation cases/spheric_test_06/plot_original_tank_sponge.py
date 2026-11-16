"""Raw original-domain drag and resolved reference-residual spectra at c0=160."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/original_tank_sponge_c160"
reference = np.loadtxt(ROOT / "reference/Force_Re100.dat", skiprows=2)
variants = [
    (ROOT/"results/wide_sponge_c160/original_tank", "No sponge", "#757575", "--"),
    (OUT/"sponge_w0p5", "0.5 m sponge · rate 100 s⁻¹", "#0072b2", "-"),
    (OUT/"sponge_w2", "2 m sponge · rate 100 s⁻¹", "#d55e00", "-"),
]
fig, axes = plt.subplots(3,2,figsize=(13,11))
for directory,label,color,style in variants:
    data = np.genfromtxt(directory/"Re100_n040.csv",delimiter=",",names=True)
    for column,component in enumerate(("pressure","viscous")):
        selected_rows = (1,) if directory.name == "sponge_w2" else (0,1,2) if directory.name == "original_tank" else (0,2)
        for ax in axes[list(selected_rows),column]:
            ax.plot(data["time"],data[f"cd_{component}"],color=color,ls=style,lw=1.15,label=label)
for column,component in enumerate(("pressure","viscous")):
    values=[]
    for ax in axes[:,column]:
        ax.plot(reference[:,0],reference[:,column+1],color="black",lw=1.4,label="Official FD reference")
        ax.set(xlabel="Time [s]",ylabel=f"{component.capitalize()} drag coefficient")
        ax.grid(alpha=.2)
    axes[0,column].set(xlim=(0,8),title=f"{component.capitalize()} drag · no sponge versus 0.5 m sponge")
    axes[1,column].set(xlim=(0,8),title=f"{component.capitalize()} drag · 2 m sponge (separate scale)")
    axes[2,column].set(xlim=(3,7),title=f"{component.capitalize()} drag · no sponge versus 0.5 m, later interval")
    values.append(reference[(reference[:,0]>=3)&(reference[:,0]<=7),column+1])
    for directory,*_ in variants:
        if directory.name == "sponge_w2":
            continue
        d=np.genfromtxt(directory/"Re100_n040.csv",delimiter=",",names=True)
        values.append(d[f"cd_{component}"][(d["time"]>=3)&(d["time"]<=7)])
    lo,hi=min(np.min(v) for v in values),max(np.max(v) for v in values)
    margin=.08*(hi-lo)
    axes[2,column].set_ylim(lo-margin,hi+margin)
handles,labels=axes[0,0].get_legend_handles_labels()
extra_handles,extra_labels=axes[1,0].get_legend_handles_labels()
for handle,label in zip(extra_handles,extra_labels):
    if label not in labels:
        handles.append(handle); labels.append(label)
fig.legend(handles,labels,loc="lower center",ncol=2,fontsize=9)
fig.suptitle("Original 10 × 5 m tank · c₀ = 160 m/s · DBC · integrated motion\n"
             "Δp = 0.025 m · cleaning off · 2 m sponge begins at the square's top/bottom faces",fontsize=12)
fig.tight_layout(rect=(0,.09,1,.92))
fig.savefig(OUT/"drag_comparison.png",dpi=180)
fig.savefig(OUT/"drag_comparison.pdf")

spectra=np.genfromtxt(OUT/"spectra.csv",delimiter=",",names=True,dtype=None,encoding="utf-8")
fig,(raw_ax,ax)=plt.subplots(2,1,figsize=(11,8.5))
raw_rows=[]
nodes=np.arange(1.5,6.50001,.02)
frequencies=np.arange(1,int(20*len(nodes)*.02)+1)/(len(nodes)*.02)
window=np.hanning(len(nodes))
design=np.column_stack([np.ones(len(nodes)),nodes-nodes.mean()])
def raw_spectrum(values):
    detrended=values-design@np.linalg.lstsq(design,values,rcond=None)[0]
    return np.array([2*abs(np.sum(detrended*window*np.exp(-2j*np.pi*f*nodes)))/window.sum() for f in frequencies])
paths={"original_no_sponge":ROOT/"results/wide_sponge_c160/original_tank",
       "expanded_no_sponge":ROOT/"results/wide_sponge_c160/expanded_tank",
       "sponge_w0p5":OUT/"sponge_w0p5", "sponge_w2":OUT/"sponge_w2"}
for name,label,color,style in [
    ("original_no_sponge","Original tank · no sponge","#757575","--"),
    ("expanded_no_sponge","Expanded tank · no sponge","#009e73","-"),
    ("sponge_w0p5","Original tank · 0.5 m sponge","#0072b2","-"),
    ("sponge_w2","Original tank · 2 m sponge","#d55e00","-"),
]:
    data=np.genfromtxt(paths[name]/"Re100_n040.csv",delimiter=",",names=True)
    amplitude=raw_spectrum(np.interp(nodes,data["time"],data["cd_pressure"]))
    raw_ax.plot(frequencies,amplitude,label=label,color=color,ls=style,lw=1.3)
    raw_rows.extend(dict(variant=name,frequency=f,amplitude=a) for f,a in zip(frequencies,amplitude))
    mask=(spectra["variant"]==name)&(spectra["component"]=="pressure")
    ax.plot(spectra["frequency"][mask],spectra["amplitude"][mask],label=label,color=color,ls=style,lw=1.3)
reference_amplitude=raw_spectrum(np.interp(nodes,reference[:,0],reference[:,1]))
raw_ax.plot(frequencies,reference_amplitude,color="black",lw=1.3,label="Official FD reference")
raw_rows.extend(dict(variant="reference",frequency=f,amplitude=a) for f,a in zip(frequencies,reference_amplitude))
with (OUT/"raw_pressure_spectra.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=["variant","frequency","amplitude"])
    writer.writeheader();writer.writerows(raw_rows)
raw_ax.set(xlim=(1,20),xlabel="Frequency [Hz]",ylabel="Windowed raw pressure-drag amplitude",
           title="Raw pressure-drag spectrum · 1.5–6.5 s · reference shown separately")
raw_ax.grid(alpha=.2);raw_ax.legend(fontsize=8,ncol=2)
ax.set(xlim=(1,20),xlabel="Frequency [Hz]",ylabel="Windowed pressure-drag residual amplitude",
       title="Resolved spectrum of pressure-drag error · 1.5–6.5 s\n"
             "Mean/linear trend removed, Hann window; Δf ≈ 0.20 Hz; output Nyquist = 25 Hz")
ax.grid(alpha=.2)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT/"pressure_spectra.png",dpi=180)
print(OUT/"drag_comparison.png")
print(OUT/"pressure_spectra.png")
