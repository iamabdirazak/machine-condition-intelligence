"""A factual summary of every recording, taken before anything is changed."""

import csv
from pathlib import Path

import numpy as np
from scipy.stats import kurtosis

from src.contracts import Signal
from src.paths import REPORTS

WINDOW = 2048  # samples per window, about 0.17 s at 12 kHz


def summarize(signal: Signal) -> dict:
    samples = signal.samples
    return {
        "condition": signal.condition,
        "load": signal.load,
        "seconds": round(signal.seconds, 1),
        "rpm": signal.rpm,
        "rms": round(float(np.sqrt(np.mean(samples**2))), 3),
        "kurtosis": round(float(kurtosis(samples)), 2),
        "windows": len(samples) // WINDOW,
    }


def audit(signals: list[Signal], into: Path = REPORTS / "audit.csv") -> list[dict]:
    rows = [summarize(signal) for signal in signals]
    into.parent.mkdir(parents=True, exist_ok=True)
    with into.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    from src.acquisition import cwru

    for row in audit(cwru.load_all()):
        print("  ".join(f"{value!s:<10}" for value in row.values()))
