"""From continuous recordings to fixed-length windows a model can read."""

from dataclasses import replace

import numpy as np

from src.contracts import Signal, Windows

SECONDS = 10.0  # every recording contributes the same duration, so no class dominates
LENGTH = 2048  # samples per window: about 0.17 s, five or more shaft turns at 12 kHz


def trim(signal: Signal, seconds: float = SECONDS) -> Signal:
    return replace(signal, samples=signal.samples[: int(seconds * signal.sample_rate)])


def cut(signals: list[Signal], length: int = LENGTH, hop: int = LENGTH) -> Windows:
    """Slide a window along each signal. A hop shorter than the length makes windows overlap."""
    parts = []
    for signal in signals:
        starts = range(0, len(signal.samples) - length + 1, hop)
        parts.append((
            np.stack([signal.samples[start:start + length] for start in starts]),
            [signal.condition] * len(starts),
            [signal.load] * len(starts),
            [signal.rpm] * len(starts),
            [signal.source] * len(starts),
        ))
    samples, conditions, loads, rpms, sources = zip(*parts)
    return Windows(
        samples=np.concatenate(samples).astype(np.float32),
        conditions=np.concatenate(conditions),
        loads=np.concatenate(loads),
        rpms=np.concatenate(rpms),
        sources=np.concatenate(sources),
    )
