"""Fault signatures: each condition as a waveform and as an envelope spectrum."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from src.acquisition import cwru
from src.contracts import Signal
from src.features.envelope import envelope_spectrum
from src.paths import FIGURES
from src.visualization import style

SHOWN_MS = 100
SHOWN_HZ = 600
HARMONICS = 3
MARKER_NAMES = {"inner-race": "BPFI", "outer-race": "BPFO", "ball": "BSF"}


def plot(signals: list[Signal], name: str = "fault-signatures") -> None:
    style.apply()
    figure, rows = plt.subplots(
        len(signals), 2, figsize=(9, 1.45 * len(signals)),
        sharex="col", gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.28, "hspace": 0.45},
    )

    figure.subplots_adjust(top=0.92)

    for (waveform, spectrum), signal in zip(rows, signals):
        color = style.COLORS[signal.condition]
        draw_waveform(waveform, signal, color)
        draw_spectrum(spectrum, signal, color)
        waveform.set_ylabel(style.LABELS[signal.condition], rotation=0, ha="right",
                            va="center", labelpad=14, fontsize=9.5,
                            color=style.NEUTRAL["ink"], fontweight="medium")

    rows[0][0].set_title("Waveform")
    rows[0][1].set_title("Envelope spectrum")
    rows[-1][0].set_xlabel("Time (ms)")
    rows[-1][1].set_xlabel("Frequency (Hz)")
    style.legend(figure, [("dash", style.NEUTRAL["secondary"], "Predicted fault frequency and harmonics")],
                 anchor=(0.9, 0.975))

    style.save(figure, FIGURES, name)


def draw_waveform(axis, signal: Signal, color: str) -> None:
    shown = int(SHOWN_MS / 1000 * signal.sample_rate)
    time = np.arange(shown) / signal.sample_rate * 1000
    axis.plot(time, signal.samples[:shown], color=color, linewidth=0.7)
    peak = float(np.abs(signal.samples[:shown]).max())
    axis.set_xlim(0, SHOWN_MS)
    axis.set_ylim(-peak * 1.1, peak * 1.1)
    axis.yaxis.set_major_locator(MaxNLocator(3, symmetric=True))


def draw_spectrum(axis, signal: Signal, color: str) -> None:
    frequencies, amplitudes = envelope_spectrum(signal.samples, signal.sample_rate)
    visible = frequencies <= SHOWN_HZ
    axis.fill_between(frequencies[visible], amplitudes[visible], color=color, alpha=0.12, linewidth=0)
    axis.plot(frequencies[visible], amplitudes[visible], color=color)
    axis.set_xlim(0, SHOWN_HZ)
    axis.set_ylim(bottom=0)
    axis.yaxis.set_major_locator(MaxNLocator(2))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2g"))

    order = cwru.FAULT_ORDERS.get(signal.condition)
    if not order:
        return
    for harmonic in range(1, HARMONICS + 1):
        axis.axvline(harmonic * order * signal.shaft_hz, color=color, alpha=0.45,
                     linestyle=(0, (2, 3)), linewidth=0.8, zorder=0)
    axis.text(order * signal.shaft_hz, 1.0, MARKER_NAMES[signal.condition],
              transform=axis.get_xaxis_transform(), ha="center", va="bottom",
              fontsize=7.5, color=color, fontweight="medium")


if __name__ == "__main__":
    plot([cwru.load(r) for r in cwru.RECORDINGS if r.load == 0])
