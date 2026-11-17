"""Hand-crafted features: what a vibration engineer would look at first.

Statistics describe the shape of the signal: how strong it is and how spiky.
Fault bands ask the physics directly: is there unusual energy at the exact
frequency this fault would produce, given the shaft speed?
"""

import numpy as np
from scipy.stats import kurtosis, skew

from src.contracts import Windows
from src.features.envelope import envelope_spectrum

HARMONICS = 2
BAND_HZ = 6  # about one frequency bin either side at 2048 samples and 12 kHz


def statistics(samples: np.ndarray) -> dict[str, np.ndarray]:
    rms = np.sqrt(np.mean(samples**2, axis=1))
    peak = np.max(np.abs(samples), axis=1)
    return {
        "rms": rms,
        "peak": peak,
        "crest factor": peak / rms,
        "impulse factor": peak / np.mean(np.abs(samples), axis=1),
        "kurtosis": kurtosis(samples, axis=1),
        "skewness": skew(samples, axis=1),
    }


def fault_bands(windows: Windows, sample_rate: int, orders: dict[str, float]) -> dict[str, np.ndarray]:
    """Peak envelope amplitude near each fault frequency, relative to the window's typical level."""
    frequencies, amplitudes = envelope_spectrum(windows.samples, sample_rate)
    typical = np.median(amplitudes, axis=1)
    shaft_hz = windows.rpms / 60
    bands = {}
    for fault, order in orders.items():
        strongest = np.zeros(len(windows))
        for harmonic in range(1, HARMONICS + 1):
            centre = harmonic * order * shaft_hz[:, None]
            near = np.abs(frequencies[None, :] - centre) <= BAND_HZ
            strongest += np.where(near, amplitudes, 0).max(axis=1)
        bands[f"{fault} band"] = strongest / typical
    return bands


def extract(windows: Windows, sample_rate: int, orders: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    features = statistics(windows.samples) | fault_bands(windows, sample_rate, orders)
    return np.column_stack(list(features.values())), list(features)
