"""Leave-one-load-out: which recordings train and which test, in each fold."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.acquisition import cwru
from src.paths import FIGURES
from src.visualization import style

CONDITIONS = list(cwru.FIRST_ID)
LOADS = list(cwru.RPM)
CELL, GAP = 1.0, 0.18


def plot(name: str = "load-splits") -> None:
    style.apply()
    figure, panels = plt.subplots(1, len(LOADS), figsize=(9, 2.7), gridspec_kw={"wspace": 0.08})
    figure.subplots_adjust(top=0.84)

    for panel, test_load in zip(panels, LOADS):
        for row, condition in enumerate(CONDITIONS):
            for column, load in enumerate(LOADS):
                held_out = load == test_load
                panel.add_patch(FancyBboxPatch(
                    (column * (CELL + GAP), -row * (CELL + GAP)), CELL, CELL,
                    boxstyle="round,pad=0,rounding_size=0.22",
                    facecolor=style.COLORS[condition] if held_out else style.NEUTRAL["hairline"],
                    edgecolor="none",
                ))
        span = len(LOADS) * (CELL + GAP) - GAP
        panel.set_xlim(-0.05, span + 0.05)
        panel.set_ylim(-(len(CONDITIONS) - 1) * (CELL + GAP) - 0.05, CELL + 0.05)
        panel.set_aspect("equal")
        panel.axis("off")
        panel.set_title(f"Test {test_load} hp", loc="center", fontsize=9.5)
        for column, load in enumerate(LOADS):
            panel.text(column * (CELL + GAP) + CELL / 2, -len(CONDITIONS) * (CELL + GAP) + GAP + 0.55,
                       f"{load}", ha="center", va="top", fontsize=8, color=style.NEUTRAL["secondary"])

    for row, condition in enumerate(CONDITIONS):
        panels[0].text(-0.3, -row * (CELL + GAP) + CELL / 2, style.LABELS[condition],
                       ha="right", va="center", fontsize=8.5, color=style.NEUTRAL["ink"])
    figure.text(0.5, -0.03, "Motor load (hp)", ha="center", fontsize=8.5, color=style.NEUTRAL["secondary"])

    style.legend(figure, [
        ("swatch", style.NEUTRAL["hairline"], "Train"),
        ("swatch", [style.COLORS[c] for c in CONDITIONS], "Test"),
    ], anchor=(0.9, 0.97))

    style.save(figure, FIGURES, name)


if __name__ == "__main__":
    plot()
