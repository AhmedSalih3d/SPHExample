"""Measure the smooth-stop/hold/restart diagnostic, without FD error claims."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

case = Path(__file__).resolve().parent
results = case/'results/pause_diagnostic'
with (results/'decay.csv').open(newline='') as f:
    records = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
for speed, color in ((40, '#be7b33'), (160, '#167f86')):
    name = f're100_c{speed}_pause'
    with (results/name/'native.csv').open(newline='') as f:
        force = list(csv.DictReader(f))
    axes[0].plot([float(r['time']) for r in force], [float(r['cd_pressure']) for r in force],
                 color=color, lw=.8, label=f'c0 = {speed} m/s')
    local = [row for row in records if row['c0'] == speed]
    for ax, key in zip(axes[1:], ('pressure_spatial_std', 'kinetic_energy')):
        ax.plot([r['time'] for r in local], [r[key] for r in local], color=color)
for ax in axes:
    ax.axvspan(2, 4, color='#93a1af', alpha=.2)
    ax.axvline(1.75, color='gray', ls=':', lw=.7)
    ax.axvline(4.25, color='gray', ls=':', lw=.7)
    ax.grid(alpha=.2)
axes[0].legend(frameon=False)
axes[0].set_ylabel('Raw pressure drag Cd')
axes[1].set_ylabel('Fluid pressure spatial std [Pa]')
axes[2].set(ylabel='Fluid kinetic energy [J/m]', xlabel='Time [s]')
fig.suptitle('Controlled square pause: Re = 100, CFL = 0.1, cleaning off\nSmooth deceleration 1.75–2 s; stationary 2–4 s; smooth restart 4–4.25 s')
fig.savefig(results/'pause_decay.png', dpi=170)
