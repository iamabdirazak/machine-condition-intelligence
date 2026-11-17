"""Documentation diagram: the network's architecture.

The layout: a soft frame with a title, a symmetric row of raised
cards, each with an icon, a name and a short description, and small arrows
between them.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

from src.paths import ROOT
from src.visualization import style

DOCS = ROOT / "docs"
STRONG, BASE, SOFT, MUTED, FAINT = (style.SHADES[k] for k in ("strong", "base", "soft", "muted", "faint"))
FRAME, CARD, ARROW = style.NEUTRAL["background"], style.NEUTRAL["surface"], style.NEUTRAL["tertiary"]

CARD_W, CARD_H, GAP = 2.6, 3.4, 0.41  # portrait cards, like the UAV figure
MARGIN = 0.4  # around the row of cards, equal to CARD_Y
CARD_Y = 0.4
CARD_RADIUS = 0.34  # the frame's corner is this plus the margin, so the two curves are concentric
HEIGHT = CARD_Y + CARD_H + 1.0  # room above the cards for the title
ICON_Y = CARD_Y + 2.25
NAME_Y, SUBTITLE_Y = CARD_Y + 1.05, CARD_Y + 0.65
MIDDLE = CARD_Y + CARD_H / 2


# Drawing primitives ---------------------------------------------------------

def rounded(axis, x, y, w, h, color, radius=0.12, z=3):
    axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                  facecolor=color, edgecolor="none", zorder=z))


def wave(axis, cx, cy, width, amplitude, color, cycles=3.0, seed=0, noise=0.0, lw=2.2, z=4):
    x = np.linspace(-0.5, 0.5, 200)
    rng = np.random.default_rng(seed)
    y = np.sin(2 * np.pi * cycles * x) * (0.6 + 0.4 * np.cos(2 * np.pi * x))
    y += noise * rng.standard_normal(len(x))
    y = y / np.max(np.abs(y))
    axis.plot(cx + x * width, cy + y * amplitude, color=color, linewidth=lw,
              solid_capstyle="round", zorder=z)


# Icons, each centred on (cx, cy) ------------------------------------------

def icon_signal(axis, cx, cy):
    rounded(axis, cx - 0.62, cy - 0.5, 1.0, 0.66, FAINT, 0.1)
    rounded(axis, cx - 0.5, cy - 0.4, 1.0, 0.66, SOFT, 0.1)
    rounded(axis, cx - 0.36, cy - 0.28, 1.04, 0.68, BASE, 0.1, z=4)
    wave(axis, cx + 0.16, cy + 0.06, 0.8, 0.18, style.NEUTRAL["ink"], cycles=3.5, lw=1.8, z=5)


def icon_normalise(axis, cx, cy):
    """A large, off-centre signal becomes the same signal, centred and scaled."""
    axis.plot([cx - 0.75, cx + 0.75], [cy - 0.1, cy - 0.1], color=SOFT, linewidth=1.2,
              linestyle=(0, (1.5, 2.5)), zorder=3)
    wave(axis, cx - 0.44, cy + 0.12, 0.56, 0.3, SOFT, cycles=2.5, lw=2.2)
    axis.add_patch(Polygon([[cx - 0.06, cy + 0.02], [cx - 0.06, cy + 0.2], [cx + 0.08, cy + 0.11]],
                           facecolor=ARROW, edgecolor="none", zorder=4))
    wave(axis, cx + 0.44, cy - 0.1, 0.56, 0.14, BASE, cycles=2.5, lw=2.2)

def icon_classes(axis, cx, cy):
    """One window branches towards four conditions and follows the most likely one."""
    start, split, finish = cx - 0.7, cx - 0.25, cx + 0.55
    targets = (cy + 0.51, cy + 0.17, cy - 0.17, cy - 0.51)
    chosen = 1
    axis.plot([start, split], [cy, cy], color=BASE, linewidth=2.6, solid_capstyle="round", zorder=4)
    t = np.linspace(0, 1, 60)[:, None]
    for index, target in enumerate(targets):
        p0, p1, p2, p3 = (np.array(point) for point in
                          ((split, cy), (split + 0.4, cy), (finish - 0.4, target), (finish, target)))
        curve = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
        picked = index == chosen
        axis.plot(curve[:, 0], curve[:, 1], color=BASE if picked else MUTED, linewidth=2.6 if picked else 2.0,
                  solid_capstyle="round", zorder=5 if picked else 3)
        axis.add_patch(Circle((finish + 0.04, target), 0.1 if picked else 0.07,
                              facecolor=STRONG if picked else MUTED,
                              edgecolor="none", zorder=6))
    axis.add_patch(Circle((start, cy), 0.08, facecolor=BASE, edgecolor="none", zorder=5))

def icon_filters(axis, cx, cy):
    """Four learned filters, each passing its own frequency band."""
    x = np.linspace(-0.62, 0.62, 200)
    for index, (peak, color) in enumerate(zip((-0.3, -0.1, 0.12, 0.34), (SOFT, STRONG, SOFT, STRONG))):
        y = cy + 0.42 - index * 0.28
        response = 0.17 * np.exp(-((x - peak) / 0.07) ** 2)
        axis.plot(cx + x, y + response, color=color, linewidth=2.2, solid_capstyle="round", zorder=4)


def icon_pool(axis, cx, cy):
    """Activity across time, averaged into one value per channel."""
    heights = (0.28, 0.62, 0.2, 0.5, 0.34)
    for index, height in enumerate(heights):
        rounded(axis, cx - 0.62 + index * 0.24, cy - 0.38, 0.14, height, FAINT if index % 2 else SOFT, 0.06)
    level = cy - 0.38 + np.mean(heights)
    axis.plot([cx - 0.66, cx + 0.5], [level, level], color=BASE, linewidth=1.8,
              linestyle=(0, (2, 2)), zorder=5)
    axis.add_patch(Circle((cx + 0.62, level), 0.09, facecolor=STRONG, edgecolor="none", zorder=6))


def icon_bearing(axis, cx, cy):
    axis.add_patch(Circle((cx, cy), 0.5, facecolor="none", edgecolor=BASE, linewidth=5, zorder=4))
    axis.add_patch(Circle((cx, cy), 0.22, facecolor="none", edgecolor=BASE, linewidth=4, zorder=4))
    for angle in np.linspace(0, 2 * np.pi, 9)[:-1]:
        axis.add_patch(Circle((cx + 0.36 * np.cos(angle), cy + 0.36 * np.sin(angle)), 0.07,
                              facecolor=SOFT, edgecolor="none", zorder=4))


def icon_network(axis, cx, cy):
    layers = [(-0.55, (-0.35, 0, 0.35)), (0, (-0.45, -0.15, 0.15, 0.45)), (0.55, (-0.2, 0.2))]
    for (x1, ys1), (x2, ys2) in zip(layers, layers[1:]):
        for y1 in ys1:
            for y2 in ys2:
                axis.plot([cx + x1, cx + x2], [cy + y1, cy + y2], color=FAINT, linewidth=1, zorder=3)
    for x, ys in layers:
        for y in ys:
            axis.add_patch(Circle((cx + x, cy + y), 0.09, facecolor=BASE if x == 0.55 else SOFT,
                                  edgecolor="none", zorder=4))


def diagram(title: str, cards: list[tuple], name: str) -> None:
    style.apply()
    width = len(cards) * CARD_W + (len(cards) - 1) * GAP + 2 * MARGIN
    figure, axis = plt.subplots(figsize=(width / 1.45, HEIGHT / 1.45))
    axis.set_xlim(0, width)
    axis.set_ylim(0, HEIGHT)
    axis.set_aspect("equal")
    axis.axis("off")

    rounded(axis, 0, 0, width, HEIGHT, FRAME, CARD_RADIUS + CARD_Y, z=1)
    axis.text(width / 2, HEIGHT - 0.5, title, ha="center", va="center", fontsize=13,
              fontweight="medium", color=style.NEUTRAL["ink"], zorder=6)

    for index, (name_text, subtitle, draw) in enumerate(cards):
        x = MARGIN + index * (CARD_W + GAP)
        centre = x + CARD_W / 2
        rounded(axis, x, CARD_Y, CARD_W, CARD_H, CARD, CARD_RADIUS, z=2)
        draw(axis, centre, ICON_Y)
        axis.text(centre, NAME_Y, name_text, ha="center", va="center", fontsize=10,
                  fontweight="medium", color=style.NEUTRAL["ink"], zorder=6)
        axis.text(centre, SUBTITLE_Y, subtitle, ha="center", va="center", fontsize=7.4,
                  color=style.NEUTRAL["secondary"], zorder=6)
        if index:
            mid = x - GAP / 2
            axis.add_patch(Polygon([[mid - 0.07, MIDDLE - 0.12], [mid - 0.07, MIDDLE + 0.12], [mid + 0.09, MIDDLE]],
                                   facecolor=ARROW, edgecolor="none", zorder=3))

    DOCS.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        figure.savefig(DOCS / f"{name}.{extension}", pad_inches=0.05, facecolor=style.NEUTRAL["background"])
    plt.close(figure)


def architecture() -> None:
    diagram("Model Architecture", [
        ("Input", "Vibration window", icon_signal),
        ("Normalisation", "Centred and scaled", icon_normalise),
        ("Frequency filters", "Finding useful bands", icon_filters),
        ("Convolution", "Learning fault patterns", icon_network),
        ("Pooling", "Summarising over time", icon_pool),
        ("Classification", "Choosing a condition", icon_classes),
        ("Output", "Bearing condition", icon_bearing),
    ], "architecture")

if __name__ == "__main__":
    architecture()
