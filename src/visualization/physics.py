"""How often physics alone, and the model, name each condition correctly."""

import json

import matplotlib.pyplot as plt
import numpy as np

from src.paths import FIGURES, REPORTS
from src.visualization import style

CONDITIONS = ("normal", "inner-race", "ball", "outer-race")


def main() -> None:
    style.apply()
    result = json.loads((REPORTS / "physics.json").read_text())
    physics = [100 * result["by_condition"][c]["physics_matches_label"] for c in CONDITIONS]
    model = [100 * result["by_condition"][c]["model_matches_label"] for c in CONDITIONS]

    figure, axis = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(CONDITIONS))
    axis.set_xlim(-0.6, len(CONDITIONS) - 0.4)
    axis.set_ylim(0, 110)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_xticks(x, [style.LABELS[c] for c in CONDITIONS])
    axis.set_ylabel("Windows named correctly, %")
    axis.set_title("Physics and Model")
    figure.canvas.draw()
    style.rounded_bars(axis, x - 0.17, physics, [style.SHADES["soft"]] * 4, width=0.3)
    style.rounded_bars(axis, x + 0.17, model, [style.SHADES["base"]] * 4, width=0.3)
    style.legend(figure, [("swatch", style.SHADES["soft"], "Envelope physics alone"),
                          ("swatch", style.SHADES["base"], "Model")])
    style.save(figure, FIGURES, "physics")


if __name__ == "__main__":
    main()
