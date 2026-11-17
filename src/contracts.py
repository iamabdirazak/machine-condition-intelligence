"""The shapes of data that pass between stages."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Signal:
    """One continuous vibration recording from a single sensor."""

    samples: np.ndarray
    sample_rate: int
    rpm: int
    condition: str
    load: int
    source: str  # identifies the recording, so splits never mix it across train and test

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def shaft_hz(self) -> float:
        return self.rpm / 60


@dataclass(frozen=True)
class Windows:
    """Fixed-length windows cut from signals, with what each one came from."""

    samples: np.ndarray  # (count, length)
    conditions: np.ndarray  # (count,) condition name per window
    loads: np.ndarray  # (count,)
    rpms: np.ndarray  # (count,) shaft speed, needed to locate fault frequencies
    sources: np.ndarray  # (count,) recording each window was cut from

    def __len__(self) -> int:
        return len(self.samples)

    def where(self, mask: np.ndarray) -> "Windows":
        return Windows(self.samples[mask], self.conditions[mask], self.loads[mask],
                       self.rpms[mask], self.sources[mask])
