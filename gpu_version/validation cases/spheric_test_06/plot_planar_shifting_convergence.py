"""Measure self-convergence of matched DDT2 force histories."""
import argparse
import csv

from plot_planar_shifting_benchmark import ROOT, np, plt, read_history

RESOLUTIONS = (40, 80, 160)
REYNOLDS = (50, 100, 150)
SOLVERS = {"native": "SPHExampleGPU", "dual": "DualSPHysics"}
COMPONENTS = ("pressure", "viscous")
RESULTS = ROOT / "results"
OUT = RESULTS / "planar_shifting_convergence_ddt2"


def history_path(n, solver, re):
    resolution = "" if n == 160 else f"_n{n}"
    filename = f"Re{re:03d}_n{n:03d}.csv" if solver == "native" else "dual.csv"
    return RESULTS / f"planar_shifting_benchmark{resolution}_ddt2" / solver / f"re{re}" / filename


def rms_change(a, b, component, start=0.0, stop=8.0):
    """Integrate the squared difference of the piecewise-linear histories."""
    times = np.unique(np.concatenate((
        [start, stop],
        *[d["time"][(start < d["time"]) & (d["time"] < stop)] for d in (a, b)],
    )))
    key = f"cd_{component}"
    diff = np.interp(times, a["time"], a[key]) - np.interp(times, b["time"], b[key])
    left, right = diff[:-1], diff[1:]
    integral = np.sum(np.diff(times) * (left**2 + left*right + right**2) / 3)
    return np.sqrt(integral / (stop - start))


def startup_peak(history):
    samples = history[(0 <= history["time"]) & (history["time"] <= 1.2)]
    index = np.argmax(samples["cd_pressure"])
    return samples["time"][index], samples["cd_pressure"][index]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(partial):
    histories = {}
    peaks = []
    for solver in SOLVERS:
        for re in REYNOLDS:
            for n in RESOLUTIONS:
                path = history_path(n, solver, re)
                if partial and not path.exists():
                    continue
                history = read_history(path)
                if not np.all(np.diff(history["time"]) > 0):
                    raise ValueError(f"Non-increasing sample times: {path}")
                histories[solver, re, n] = history
                time, peak = startup_peak(history)
                peaks.append(dict(solver=solver, reynolds=re, resolution=n,
                                  peak_time=time, peak_pressure_cd=peak))
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "startup_peaks.csv", peaks)

    summary = []
    window_changes = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, layout="constrained")
    startup, zoom = plt.subplots(2, 3, figsize=(13, 7), sharex=True, layout="constrained")
    for row, (solver, label) in enumerate(SOLVERS.items()):
        for col, re in enumerate(REYNOLDS):
            available = {n: histories[solver, re, n] for n in RESOLUTIONS if (solver, re, n) in histories}
            for n, data in available.items():
                zoom[row, col].plot(data["time"], data["cd_pressure"], label=f"L/dp = {n}")
            zoom[row, col].set(title=f"{label}, Re = {re}", xlim=(0, 1.2), ylabel="Pressure drag Cd")
            zoom[row, col].grid(alpha=.25)
            for component in COMPONENTS:
                changes = {}
                for low, high in zip(RESOLUTIONS, RESOLUTIONS[1:]):
                    if low in available and high in available:
                        changes[high] = rms_change(available[low], available[high], component)
                if len(changes) == 2:
                    coarse, fine = changes[80], changes[160]
                    order = np.log2(coarse / fine) if coarse > 0 and fine > 0 else np.nan
                    summary.append(dict(solver=solver, reynolds=re, component=component,
                                        rms_40_to_80=coarse, rms_80_to_160=fine, change_order=order))
                    for start, stop in ((0, 1.2), (1.2, 7), (7, 8)):
                        window_changes.append(dict(
                            solver=solver, reynolds=re, component=component, start=start, stop=stop,
                            rms_40_to_80=rms_change(available[40], available[80], component, start, stop),
                            rms_80_to_160=rms_change(available[80], available[160], component, start, stop),
                        ))
                ax = axes[row, col]
                if changes:
                    ax.plot(list(changes), list(changes.values()), "o-", label=component.capitalize())
                ax.set(title=f"{label}, Re = {re}", ylabel="RMS change in Cd", xticks=(80, 160))
                ax.grid(alpha=.25)
            axes[row, col].set_xlabel("Finer L/dp in each pair")
            zoom[row, col].set_xlabel("Time (s)")
    if summary:
        write_csv(OUT / "drag_self_convergence.csv", summary)
        write_csv(OUT / "drag_changes_by_time_window.csv", window_changes)
    if any(ax.has_data() for ax in axes.flat):
        handles, labels = next(ax.get_legend_handles_labels() for ax in axes.flat if ax.has_data())
        fig.legend(handles, labels, loc="outside lower center", ncol=2)
        fig.suptitle("SPHERIC Test 06 · Change in drag with refinement · Fourtakas DDT2")
        fig.savefig(OUT / "drag_self_convergence.png", dpi=180)
    handles, labels = max((ax.get_legend_handles_labels() for ax in zoom.flat), key=lambda pair: len(pair[0]))
    startup.legend(handles, labels, loc="outside lower center", ncol=3)
    startup.suptitle("SPHERIC Test 06 · Startup pressure drag by resolution · Fourtakas DDT2")
    startup.savefig(OUT / "startup_refinement.png", dpi=180)
    plt.close("all")
    print(f"Measured {len(peaks)} startup peaks and {len(summary)} complete refinement comparisons in {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partial", action="store_true", help="Plot completed histories while the sweep runs")
    main(parser.parse_args().partial)
