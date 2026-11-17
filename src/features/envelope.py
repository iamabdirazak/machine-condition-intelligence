"""Envelope spectrum: how often a bearing is being struck, hidden under its resonance.

A defect produces a short impact every time a ball rolls over it. Each impact
rings the structure at a high resonant frequency. Band-passing around that
resonance, taking the envelope, and reading its spectrum recovers the impact
rate, which is the fault frequency.

Works on one signal or on a batch of windows (along the last axis).
"""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

RESONANCE_BAND = (2_000, 5_000)


def envelope_spectrum(
    samples: np.ndarray,
    sample_rate: int,
    band: tuple[float, float] = RESONANCE_BAND,
) -> tuple[np.ndarray, np.ndarray]:
    bandpass = butter(4, band, btype="bandpass", fs=sample_rate, output="sos")
    envelope = np.abs(hilbert(sosfiltfilt(bandpass, samples, axis=-1), axis=-1))
    envelope -= envelope.mean(axis=-1, keepdims=True)
    length = samples.shape[-1]
    frequencies = np.fft.rfftfreq(length, d=1 / sample_rate)
    amplitudes = np.abs(np.fft.rfft(envelope, axis=-1)) / length
    return frequencies, amplitudes
