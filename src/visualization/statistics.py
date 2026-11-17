"""Signal statistics: how strong and how impulsive each condition is, averaged over loads."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.paths import FIGURES, REPORTS
from src.visualization import style

MEASURES = {"rms": "RMS (g)", "kurtosis": "Excess kurtosis"}


def plot(audit: Path = REPORTS / "audit.csv", name: str = "signal-statistics") -> None:
    rows = list(csv.DictReader(audit.open()))
    conditions = list(dict.fromkeys(row["condition"] for row in rows))

    style.apply()
    figure, axes = plt.subplots(1, len(MEASURES), figsize=(9, 2.8), gridspec_kw={"wspace": 0.25})

    for axis, (measure, title) in zip(axes, MEASURES.items()):
        values = defaultdict(list)
        for row in rows:
            values[row["condition"]].append(float(row[measure]))
        means = [float(np.mean(values[c])) for c in conditions]
        positions = range(len(conditions))

        lowest = min(0, *means)
        highest = max(means)
        span = highest - lowest
        axis.set_xlim(-0.5, len(conditions) - 0.5)
        axis.set_ylim(lowest - 0.08 * span if lowest < 0 else 0, highest + 0.18 * span)

        style.rounded_bars(axis, positions, means, [style.COLORS[c] for c in conditions])
        for position, mean in zip(positions, means):
            axis.annotate(f"{mean:.2f}", (position, max(mean, 0)), xytext=(0, 5),
                          textcoords="offset points", ha="center", va="bottom",
                          fontsize=8, color=style.NEUTRAL["ink"])

        axis.set_title(title)
        axis.set_xticks(list(positions), [style.LABELS[c] for c in conditions])
        axis.axhline(0, color=style.NEUTRAL["tertiary"], linewidth=0.8, zorder=1)

    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        figure.savefig(FIGURES / f"{name}.{extension}")
    plt.close(figure)


if __name__ == "__main__":
    plot()
