"""Ways to make windows look more like the messy field: used to train, and to test robustness."""

from dataclasses import replace

import numpy as np

from src.contracts import Windows


def add_noise(windows: Windows, snr_db: float | np.ndarray, seed: int = 0) -> Windows:
    """White Gaussian noise at a given signal-to-noise ratio, set per window.

    0 dB means the noise is as strong as the vibration itself. Pass one value
    for every window, or one value per window.
    """
    power = np.mean(windows.samples**2, axis=1, keepdims=True)
    snr = np.reshape(np.asarray(snr_db, dtype=float), (-1, 1))
    scale = np.sqrt(power / 10 ** (snr / 10))
    noise = np.random.default_rng(seed).standard_normal(windows.samples.shape) * scale
    return replace(windows, samples=(windows.samples + noise).astype(np.float32))


def reshape_spectrum(samples: np.ndarray, rng: np.random.Generator,
                     knots: int = 8, max_db: float = 12.0) -> np.ndarray:
    """Pass each window through a random mechanical path.

    Vibration reaches a sensor through the machine's structure, which boosts
    some frequencies and damps others. A random smooth gain curve, and
    sometimes a low-pass roll-off, imitates a sensor mounted somewhere else.
    """
    length = samples.shape[1]
    position = np.linspace(0, 1, length // 2 + 1)
    gains = rng.uniform(-max_db, max_db, (len(samples), knots))
    curves = np.stack([np.interp(position, np.linspace(0, 1, knots), g) for g in gains])
    cutoffs = rng.uniform(0.3, 0.9, (len(samples), 1))
    rolled_off = rng.random((len(samples), 1)) < 0.5
    curves -= rolled_off * 40 * np.clip((position - cutoffs) / 0.1, 0, 1)
    shaped = np.fft.rfft(samples, axis=1) * 10 ** (curves / 20)
    return np.fft.irfft(shaped, length, axis=1).astype(np.float32)
