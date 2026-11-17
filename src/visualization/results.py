"""Results figures: how each model holds up across the evaluation protocols."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.paths import FIGURES, REPORTS
from src.visualization import style

PROTOCOL_TITLES = {"leave one load out": "Train on three loads", "single load": "Train on one load"}
PROTOCOL_AXES = {"leave one load out": "Test load (hp)", "single load": "Training load (hp)"}
BAR = style.ACCENT


def protocols(results: dict, name: str) -> None:
    style.apply()
    figure, axes = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True, gridspec_kw={"wspace": 0.12})
    for axis, protocol in zip(axes, PROTOCOL_TITLES):
        folds = results[protocol]
        scores = [fold["macro_f1"] for fold in folds.values()]
        axis.set_xlim(-0.5, len(scores) - 0.5)
        axis.set_ylim(0, 1.12)
        style.rounded_bars(axis, range(len(scores)), scores, [BAR] * len(scores))
        for position, value in enumerate(scores):
            axis.annotate(f"{value:.2f}", (position, value), xytext=(0, 5), textcoords="offset points",
                          ha="center", va="bottom", fontsize=8, color=style.NEUTRAL["ink"])
        axis.set_xticks(range(len(scores)), [key.split()[0] for key in folds])
        axis.set_xlabel(PROTOCOL_AXES[protocol])
        axis.set_title(PROTOCOL_TITLES[protocol])
    axes[0].set_ylabel("Macro-F1")
    save(figure, name)


def noise(results: dict, name: str) -> None:
    style.apply()
    levels = list(results["noise"])
    snrs = [int(level.split()[0]) for level in levels]
    figure, (overall, per_condition) = plt.subplots(1, 2, figsize=(9, 2.9), sharey=True,
                                                    gridspec_kw={"wspace": 0.12})

    overall.plot(snrs, [results["noise"][l]["macro_f1"] for l in levels], color=BAR,
                 linewidth=1.8, marker="o", markersize=4)
    overall.set_title("All conditions")
    overall.set_ylabel("Macro-F1")

    for condition, color in style.COLORS.items():
        recalls = [results["noise"][l]["recall"][condition] for l in levels]
        per_condition.plot(snrs, recalls, color=color, linewidth=1.8, marker="o", markersize=4)
    per_condition.set_title("Each condition")
    per_condition.set_ylabel("Recall")
    per_condition.yaxis.set_tick_params(labelleft=True)

    for axis in (overall, per_condition):
        axis.set_xlim(max(snrs) + 1.5, min(snrs) - 1.5)
        axis.set_ylim(-0.04, 1.08)
        axis.set_xticks(snrs)
        axis.set_xlabel("Signal-to-noise ratio (dB)")

    figure.subplots_adjust(top=0.8)
    style.legend(figure, [("swatch", c, style.LABELS[k]) for k, c in style.COLORS.items()],
                 anchor=(0.9, 0.95))
    save(figure, name)


def save(figure, name: str) -> None:
    style.save(figure, FIGURES, name)


def plot(model: str = "baseline") -> None:
    results = json.loads((REPORTS / f"{model}.json").read_text())
    protocols(results, f"{model}-protocols")
    noise(results, f"{model}-noise")




APPROACHES = {
    "baseline": ("Features + trees", style.COLORS["normal"]),
    "convolution-clean": ("Network", style.SHADES["muted"]),
    "convolution": ("+ noise training", style.SHADES["soft"]),
    "convolution-transfer": ("+ transfer training", style.SHADES["base"]),
}
NETWORKS = [a for a in APPROACHES if a != "baseline"]


def mean_f1(folds: dict) -> float:
    return sum(f["macro_f1"] for f in folds.values()) / len(folds)


def grouped_bars(axis, groups: list[str], series: dict[str, tuple[list[float], str]], width: float = 0.18) -> None:
    """Rounded bars side by side within each group, each labelled with its value."""
    count = len(series)
    axis.set_xlim(-0.5, len(groups) - 0.5)
    for index, (values, color) in enumerate(series.values()):
        offset = (index - (count - 1) / 2) * (width + 0.03)
        positions = [g + offset for g in range(len(groups))]
        present = [(x, v) for x, v in zip(positions, values) if v is not None]
        style.rounded_bars(axis, [x for x, _ in present], [v for _, v in present],
                           [color] * len(present), width=width, radius=0.05)
        for x, v in present:
            axis.annotate(f"{v:.2f}", (x, v), xytext=(0, 4), textcoords="offset points",
                          ha="center", va="bottom", fontsize=7, color=style.NEUTRAL["ink"])
    axis.set_xticks(range(len(groups)), groups)


def load(approaches) -> dict:
    return {a: json.loads((REPORTS / f"{a}.json").read_text()) for a in approaches}


def compare(name: str = "comparison") -> None:
    reports = load(APPROACHES)
    style.apply()
    figure, (bars, lines) = plt.subplots(1, 2, figsize=(9, 3), gridspec_kw={"wspace": 0.22})

    bars.set_ylim(0, 1.14)
    grouped_bars(bars, [PROTOCOL_TITLES[p] for p in PROTOCOL_TITLES], {
        a: ([mean_f1(reports[a][p]) for p in PROTOCOL_TITLES], color) for a, (_, color) in APPROACHES.items()
    })
    bars.set_ylabel("Macro-F1")
    bars.set_title("Load shift")

    for approach, (_, color) in APPROACHES.items():
        levels = list(reports[approach]["noise"])
        snrs = [int(level.split()[0]) for level in levels]
        lines.plot(snrs, [reports[approach]["noise"][l]["macro_f1"] for l in levels],
                   color=color, linewidth=1.8, marker="o", markersize=4)
    lines.set_xlim(max(snrs) + 1.5, min(snrs) - 1.5)
    lines.set_ylim(0, 1.08)
    lines.set_xticks(snrs)
    lines.set_xlabel("Signal-to-noise ratio (dB)")
    lines.set_ylabel("Macro-F1")
    lines.set_title("Noise")

    figure.subplots_adjust(top=0.8)
    style.legend(figure, [("swatch", color, label) for label, color in APPROACHES.values()], anchor=(0.9, 0.95))
    save(figure, name)


def sensor_shift(name: str = "sensor-shift") -> None:
    reports = load(APPROACHES)
    style.apply()
    figure, axis = plt.subplots(figsize=(9, 3))
    axis.set_ylim(0, 1.1)
    grouped_bars(axis, [label for label, _ in APPROACHES.values()], {
        "before": ([mean_f1(reports[a]["sensor shift"]) for a in APPROACHES], style.SHADES["soft"]),
        "after": ([mean_f1(reports[a]["sensor shift, adapted"]) if a in NETWORKS else None for a in APPROACHES],
                  style.SHADES["strong"]),
    }, width=0.28)
    axis.set_ylabel("Macro-F1")
    axis.set_title("Fan-end sensor, unseen in training")
    figure.subplots_adjust(top=0.8)
    style.legend(figure, [("swatch", style.SHADES["soft"], "Before adaptation"),
                          ("swatch", style.SHADES["strong"], "After adaptation")], anchor=(0.9, 0.95))
    save(figure, name)


def drift_alarms(name: str = "drift-alarms") -> None:
    reports = load(NETWORKS)
    scenarios = list(reports[NETWORKS[0]]["drift alarms"])
    style.apply()
    figure, axis = plt.subplots(figsize=(9, 3))
    axis.set_ylim(0, 1.1)
    grouped_bars(axis, [s[0].upper() + s[1:] for s in scenarios], {
        a: ([reports[a]["drift alarms"][s] for s in scenarios], APPROACHES[a][1]) for a in NETWORKS
    }, width=0.22)
    axis.set_ylabel("Batches flagged")
    axis.set_title("Drift alarms")
    figure.subplots_adjust(top=0.8)
    style.legend(figure, [("swatch", APPROACHES[a][1], APPROACHES[a][0]) for a in NETWORKS], anchor=(0.9, 0.95))
    save(figure, name)


if __name__ == "__main__":
    plot("baseline")
    compare()
    sensor_shift()
    drift_alarms()
