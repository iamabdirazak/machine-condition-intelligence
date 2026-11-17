"""Documentation diagrams: the model as a file, and its architecture.

The layout: a soft frame with a title, a symmetric row of raised
cards, each with an icon, a name and a short description, and small arrows
between them.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from PIL import Image

from src.paths import ROOT
from src.visualization import style

DOCS = ROOT / "docs"
STRONG, BASE, SOFT, MUTED, FAINT = (style.SHADES[k] for k in ("strong", "base", "soft", "muted", "faint"))
FRAME, CARD, ARROW = style.NEUTRAL["background"], style.NEUTRAL["surface"], style.NEUTRAL["tertiary"]

CARD_W, CARD_H, GAP = 2.6, 3.4, 0.41  # portrait cards, like the UAV figure
MARGIN = 0.4  # around the row of cards, equal to CARD_Y
CARD_Y = 0.4
CARD_RADIUS, FRAME_RADIUS = 0.27, 0.38  # in proportion to the UAV figure's cards and frame
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

    rounded(axis, 0, 0, width, HEIGHT, FRAME, FRAME_RADIUS, z=1)
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

    style.save(figure, DOCS, name, panel=False)


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


# The model as a file -------------------------------------------------------

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as Shape
from matplotlib.patches import PathPatch


def fill(axis, patch, colors, angle=90, z=3):
    """Fill a patch with a linear gradient; angle 90 runs bottom to top."""
    axis.add_patch(patch)
    patch.set_facecolor("none")
    patch.set_edgecolor("none")
    x0, y0, x1, y1 = patch.get_path().transformed(patch.get_patch_transform()).get_extents().bounds
    x1, y1 = x0 + x1, y0 + y1
    t = np.radians(angle)
    gx, gy = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))
    ramp = gx * np.cos(t) + gy * np.sin(t)
    ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min())
    image = axis.imshow(ramp, extent=(x0, x1, y0, y1), origin="lower", zorder=z,
                        cmap=LinearSegmentedColormap.from_list("", colors), interpolation="bicubic")
    image.set_clip_path(patch)


def corner(points, centre, radius, start, end):
    angles = np.linspace(np.radians(start), np.radians(end), 12)
    points += [(centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)) for a in angles]


def page(x, y, w, h, fold, radius):
    """A document outline with rounded corners and a folded top-right corner."""
    points = []
    corner(points, (x + radius, y + radius), radius, 180, 270)
    corner(points, (x + w - radius, y + radius), radius, 270, 360)
    points += [(x + w, y + h - fold), (x + w - fold, y + h)]
    corner(points, (x + radius, y + h - radius), radius, 90, 180)
    return PathPatch(Shape(points + [points[0]], closed=True))


def squircle(x, y, size, radius):
    return FancyBboxPatch((x, y), size, size, boxstyle=f"round,pad=0,rounding_size={radius}")


def onnx_mark() -> np.ndarray:
    """The ONNX mark from docs/onnx.png, without its wordmark, recoloured in the accent.

    Lines take the brightest shade. Each facet keeps its original shading, mapped
    from the deepest shade for the darkest facet to the base shade for the lightest.
    """
    source = np.asarray(Image.open(DOCS / "onnx.png").convert("RGBA")).astype(float) / 255
    mark = source[:575]                                                        # above the wordmark
    rows, columns = np.where(mark[..., 3] > 0.05)
    mark = mark[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    lightness, alpha = mark[..., :3].mean(axis=2), mark[..., 3]
    line = lightness < 0.5
    facets = LinearSegmentedColormap.from_list("", [MUTED, SOFT, BASE])
    shade = np.clip((lightness - 0.6) / 0.38, 0, 1)                           # facet greys span about 0.6 to 0.98
    image = np.empty_like(mark)
    image[..., :3] = np.where(line[..., None], matplotlib.colors.to_rgb(STRONG), facets(shade)[..., :3])
    image[..., 3] = np.where(line, alpha, alpha * 0.9)
    return image


def model_file() -> None:
    """The released model shown as what it is: a single ONNX file."""
    style.apply()
    width, height = 7 * CARD_W + 6 * GAP + 2 * MARGIN, 7.2
    figure, axis = plt.subplots(figsize=(width / 1.45, height / 1.45))
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_aspect("equal")
    axis.axis("off")
    rounded(axis, 0, 0, width, height, FRAME, FRAME_RADIUS, z=1)

    w, h, fold = 2.55, 3.4, 0.62                                              # document proportions, 3:4
    x, y = width / 2 - w / 2, 2.5
    cx = width / 2

    for spread, alpha in ((0.32, 0.04), (0.2, 0.06), (0.1, 0.09), (0.03, 0.14)):   # soft lift
        shadow = page(x - spread, y - spread - 0.12, w + 2 * spread, h + 2 * spread, fold + spread, 0.2 + spread)
        shadow.set(facecolor="black", edgecolor="none", alpha=alpha, zorder=2)
        axis.add_patch(shadow)

    sheet = page(x, y, w, h, fold, 0.2)
    fill(axis, sheet, ["#1D1D20", "#2A2A2E", "#38383D"], z=3)                # lit from above
    rim = page(x, y, w, h, fold, 0.2)
    rim.set(facecolor="none", edgecolor="#4A4A50", linewidth=1, zorder=4)
    axis.add_patch(rim)
    flap = PathPatch(Shape([(x + w - fold, y + h), (x + w - fold, y + h - fold + 0.1),
                            (x + w - fold + 0.1, y + h - fold), (x + w, y + h - fold),
                            (x + w - fold, y + h)], closed=True))
    fill(axis, flap, ["#3E3E44", "#5A5A61"], angle=45, z=4)

    mark = onnx_mark()
    size = 1.3                                                                 # about half the page width
    aspect = mark.shape[0] / mark.shape[1]
    label = 0.3                                                                # cap height of the label
    block = size * aspect + 0.36 + label                                       # mark, gap, label
    top = y + h * 0.47 + block / 2                                             # centred, just below the middle
    axis.imshow(mark, extent=(cx - size / 2, cx + size / 2, top - size * aspect, top),
                zorder=6, interpolation="lanczos")
    axis.text(cx, top - block + label / 2, "ONNX", ha="center", va="center", fontsize=15,
              fontweight="semibold", color="#C7C7CC", zorder=6)

    axis.text(width / 2, 1.7, "Machine Condition Model", ha="center", va="center", fontsize=14,
              fontweight="semibold", color=style.NEUTRAL["ink"], zorder=6)
    axis.text(width / 2, 1.25, "model-int8.onnx  \u00b7  21,604 parameters  \u00b7  41 KB", ha="center", va="center", fontsize=9,
              color=style.NEUTRAL["secondary"], zorder=6)

    style.save(figure, DOCS, "model", panel=False)

# The model at a glance -----------------------------------------------------

def specification() -> None:
    """The model's key facts as a row of tiles: what a reader checks first."""
    tiles = [("Input", "2048 samples", "One accelerometer at 12 kHz"),
             ("Output", "Bearing condition", "Normal or one of three faults"),
             ("Format", "ONNX", "8-bit 41 KB, 32-bit 88 KB"),
             ("Parameters", "21,604", "Convolutional model"),
             ("Runtime", "0.02 ms", "Per window, on one CPU thread")]
    style.apply()
    width = 7 * CARD_W + 6 * GAP + 2 * MARGIN
    tile_w = (width - 2 * MARGIN - (len(tiles) - 1) * GAP) / len(tiles)
    tile_h = 1.75
    height = tile_h + 2 * CARD_Y
    figure, axis = plt.subplots(figsize=(width / 1.45, height / 1.45))
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_aspect("equal")
    axis.axis("off")
    rounded(axis, 0, 0, width, height, FRAME, FRAME_RADIUS, z=1)

    for index, (label, value, detail) in enumerate(tiles):
        x = MARGIN + index * (tile_w + GAP)
        rounded(axis, x, CARD_Y, tile_w, tile_h, CARD, CARD_RADIUS, z=2)
        left, top = x + 0.32, CARD_Y + tile_h
        axis.text(left, top - 0.38, label, ha="left", va="center", fontsize=7.4,
                  color=style.NEUTRAL["secondary"], zorder=6)
        axis.text(left, top - 0.86, value, ha="left", va="center", fontsize=12.5,
                  fontweight="semibold", color=style.NEUTRAL["ink"], zorder=6)
        axis.text(left, top - 1.32, detail, ha="left", va="center", fontsize=7.4,
                  color=style.NEUTRAL["secondary"], zorder=6)

    style.save(figure, DOCS, "specification", panel=False)


if __name__ == "__main__":
    model_file()
    specification()
    architecture()
