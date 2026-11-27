"""Regenerate the StillWedge baseline and candidate-range figures."""

import csv
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

plt.switch_backend("Agg")
plt.rcParams["svg.hashsalt"] = "still-wedge"


DIRECTORY = Path(__file__).resolve().parent


def read_rows(filename):
    with (DIRECTORY / filename).open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def save_figure(fig, filename):
    path = DIRECTORY / filename
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def plot_baseline():
    pairs = read_rows("still_wedge_baseline_warframe_closed.csv")
    rounds = [int(row["round"]) for row in pairs]
    clone = [1000 * float(row["clone_seconds"]) for row in pairs]
    installed = [1000 * float(row["installed_seconds"]) for row in pairs]
    deltas = [100 * (current / control - 1) for current, control in zip(installed, clone)]

    fig, (elapsed, change) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), layout="constrained")
    fig.suptitle("StillWedge same-source control · RTX 5080, Warframe closed")
    elapsed.plot(rounds, clone, "o-", label="Same-source clone", color="#287271")
    elapsed.plot(rounds, installed, "s-", label="Installed solver", color="#a04425")
    elapsed.set_ylabel("Complete example (ms)")
    elapsed.legend()
    change.axhline(0, color="#555555", linewidth=1)
    change.axhline(median(deltas), color="#a04425", linestyle="--", label="Paired median")
    change.bar(rounds, deltas, color="#287271")
    change.set(xlabel="Alternating pair", ylabel="Installed / clone − 1 (%)", xticks=rounds)
    change.legend()
    save_figure(fig, "still_wedge_baseline_warframe_closed.svg")


def plot_candidates():
    rows = read_rows("still_wedge_candidate_ranges_results.csv")
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    colors = ("#287271", "#a04425", "#4b5185")
    for rebuild, color in zip(sorted({row["rebuild"] for row in rows}), colors):
        data = [row for row in rows if row["rebuild"] == rebuild]
        fraction = [100 * float(row["fraction"]) for row in data]
        for field, linestyle, marker in (("envelope", "-", "o"), ("kept", "--", None)):
            removed = [100 * (1 - int(row[field]) / int(row["raw"])) for row in data]
            ax.plot(fraction, removed, linestyle=linestyle, marker=marker, color=color,
                    label=f"Rebuild {rebuild}" if field == "envelope" else None)
    ax.plot([], [], color="#444444", label="One contiguous span per row")
    ax.plot([], [], "--", color="#444444", label="Individual candidate screen")
    ax.set(title="Candidate visits that could be removed at each rebuild",
           xlabel="Displacement allowance per fluid particle (% of support radius)",
           ylabel="Original row visits removed (%)", ylim=(0, 100))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    save_figure(fig, "still_wedge_candidate_ranges.svg")


def plot_displacement():
    rows = read_rows("still_wedge_displacement_results.csv")
    steps = [int(row["step"]) for row in rows]
    accepted = [100 * float(row["accepted_fraction"]) for row in rows]
    predictor = [100 * float(row["predictor_fraction"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    ax.plot(steps, accepted, color="#287271", label="Accepted input")
    ax.plot(steps, predictor, color="#a04425", label="Predictor input", linestyle="--")
    if "scalar_predictor_fraction" in rows[0]:
        bound = [100 * float(row["scalar_predictor_fraction"]) for row in rows]
        ax.plot(steps, bound, color="#4b5185", label="Scalar path-length estimate")
    for i in range(1, len(rows)):
        if rows[i]["rebuild"] != rows[i - 1]["rebuild"]:
            ax.axvline(steps[i], color="#777777", alpha=0.6)
    ax.set(title="Maximum particle displacement from the latest rebuild",
           xlabel="Symplectic step", ylabel="Displacement (% of support radius)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, "still_wedge_displacement.svg")


def plot_masks():
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    variants = (
        ("still_wedge_baseline_warframe_closed.csv", "Same-source control", "#777777"),
        ("still_wedge_candidate_masks_results.csv", "64-bit mask, measured allowance", "#287271"),
        ("still_wedge_candidate_masks_guarded_results.csv", "64-bit mask, per-particle guard", "#a04425"),
        ("still_wedge_candidate_masks_32bit_results.csv", "32-bit mask, per-particle guard", "#4b5185"),
        ("still_wedge_candidate_masks_scalar_results.csv", "32-bit mask, scalar guard", "#806322"),
        ("still_wedge_candidate_masks_scalar_confirmation.csv", "Scalar guard confirmation", "#806322"),
        ("still_wedge_candidate_masks_scalar_disabled.csv", "Scalar guard, pruning disabled", "#287271"),
        ("still_wedge_row_walk_results.csv", "Short row bit walk alone", "#4b5185"),
        ("still_wedge_row_walk_installed_results.csv", "Installed StillWedge traversal", "#a04425"),
    )
    ax.axvline(0, color="#555555", linewidth=1)
    labels = []
    for filename, label, color in variants:
        if not (DIRECTORY / filename).exists():
            continue
        rows = read_rows(filename)
        changes = [
            100 * (float(row["ratio"]) - 1) if "ratio" in row
            else 100 * (float(row["installed_seconds"]) / float(row["clone_seconds"]) - 1)
            for row in rows
        ]
        rank = len(labels)
        labels.append(label)
        ax.hlines(rank, min(changes), max(changes), color=color, linewidth=2)
        ax.scatter(median(changes), rank, color=color, s=36, zorder=3)
    ax.set(title="Paired complete-runtime change for neighbor masks",
           xlabel="Candidate / control − 1 (%) · dot shows median, line shows range",
           yticks=range(len(labels)), yticklabels=labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, "still_wedge_candidate_masks.svg")


def plot_batch_tail():
    rows = read_rows("still_wedge_batch_tail_results.csv")
    labels = ("Inactive steps", "Graph launches", "State readbacks")
    fig, (work, timing) = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for offset, prefix, label, color in (
        (-0.18, "baseline", "Installed sizing", "#287271"),
        (0.18, "trimmed", "One fewer queued step", "#a04425"),
    ):
        counts = (
            median(int(row[f"{prefix}_submitted"]) - int(row["accepted_steps"]) for row in rows),
            median(int(row[f"{prefix}_graph_batches"]) for row in rows),
            median(int(row[f"{prefix}_readbacks"]) for row in rows),
        )
        bars = work.bar([i + offset for i in range(3)], counts, width=0.36, label=label, color=color)
        work.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
    work.set(xticks=range(3), xticklabels=labels, ylabel="Median count per run")
    work.legend()
    deltas = [100 * (float(row["ratio"]) - 1) for row in rows]
    timing.axhline(0, color="#555555", linewidth=1)
    timing.bar([int(row["round"]) for row in rows], deltas, color="#a04425")
    timing.axhline(median(deltas), color="#4b5185", linestyle="--", label="Paired median")
    timing.set(xlabel="Alternating pair", ylabel="Trimmed / control − 1 (%)")
    timing.legend()
    fig.suptitle("Effect of trimming speculative CUDA graph batches")
    save_figure(fig, "still_wedge_batch_tail.svg")


def plot_blocks():
    rows = read_rows("still_wedge_blocks_results.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2), layout="constrained")
    threads = (64, 96, 128, 256)
    groups = [
        [100 * (float(row["ratio"]) - 1) for row in rows if int(row["threads"]) == size]
        for size in threads
    ]
    ax.boxplot(groups, labels=threads, medianprops={"color": "#a04425", "linewidth": 2})
    ax.axhline(0, color="#777777", linewidth=1, linestyle="--")
    ax.set(title="StillWedge interaction block-size sweep",
           xlabel="Interaction threads per block", ylabel="Runtime relative to 128 threads (%)")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "still_wedge_blocks.svg")


if __name__ == "__main__":
    plot_baseline()
    plot_candidates()
    plot_displacement()
    plot_masks()
    if (DIRECTORY / "still_wedge_batch_tail_results.csv").exists():
        plot_batch_tail()
    if (DIRECTORY / "still_wedge_blocks_results.csv").exists():
        plot_blocks()
