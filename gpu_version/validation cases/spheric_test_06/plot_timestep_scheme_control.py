"""Render the completed, audited reference-step controls without filtering."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_edge_damping import reference

case = Path(__file__).resolve().parent
results = case/'results/timestep_scheme_control'
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, constrained_layout=True)
for r, speed in enumerate((40, 160)):
    choices = [(case/f'results/comparison/re100_c{speed}/native.csv', 'Earlier ordinary scheme', '#929ba5'),
               (results/f're100_c{speed}_hermite/native.csv', 'Audited reference, original motion', '#be7b33'),
               (results/f're100_c{speed}_integrated/native.csv', 'Audited reference, integrated motion', '#167f86')]
    for c, component in enumerate(('pressure', 'viscous')):
        ax = axes[r, c]
        for path, label, color in choices:
            with path.open(newline='') as f:
                rows = list(csv.DictReader(f))
            ax.plot([float(row['time']) for row in rows], [float(row[f'cd_{component}']) for row in rows],
                    color=color, lw=.85, label=label)
        ref = reference(100)
        ax.plot(ref[:, 0], ref[:, c+1], color='#273643', ls='--', lw=1.3, label='FD reference')
        ax.set(xlim=(0, 8), title=f'c0 = {speed} m/s: {component}',
               ylabel='Pressure Cd' if c == 0 else 'Viscous Cd')
        ax.grid(alpha=.2)
        if r == 1:
            ax.set_xlabel('Time [s]')
axes[0, 0].legend(fontsize=7.5, ncols=2, frameon=False)
fig.suptitle('Re = 100: enforcing timestep criteria does not by itself validate the loads\nReference controls also change the integrator and wall-gated density diffusion; all use laminar viscosity', fontsize=12)
fig.savefig(results/'forces.png', dpi=180)
