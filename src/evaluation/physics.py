"""Does the model decide as the bearing's physics does?

A defect is struck once per pass of a ball, so its impacts repeat at a
frequency set by the bearing's geometry and the shaft speed. For every
window, the envelope spectrum is read at each fault's frequency and its
first three harmonics, against the spectrum's own noise floor. Physics names
the fault whose frequencies stand highest above the floor, or calls the
bearing normal if none stands clear of it. The released model's decision on
the same window is then compared with physics', with no training involved.
"""

import json

import numpy as np

from src.acquisition import cwru
from src.deployment.inference import Classifier
from src.features.envelope import envelope_spectrum
from src.paths import MODELS, REPORTS
from src.preparation.windowing import cut, trim

HARMONICS = 3
CLEAR = 4.0  # a fault's peaks must stand this many times above the noise floor


def physics_decision(samples: np.ndarray, rpm: float) -> tuple[str, dict[str, float]]:
    frequencies, amplitudes = envelope_spectrum(samples, cwru.SAMPLE_RATE)
    floor = np.median(amplitudes[(frequencies > 20) & (frequencies < 600)])
    resolution = frequencies[1]
    shaft = rpm / 60
    strength = {}
    for fault, order in cwru.FAULT_ORDERS.items():
        peaks = []
        for k in range(1, HARMONICS + 1):
            near = np.abs(frequencies - k * order * shaft) <= resolution
            peaks.append(amplitudes[near].max())
        strength[fault] = float(np.mean(peaks) / floor)
    best = max(strength, key=strength.get)
    return (best if strength[best] >= CLEAR else "normal"), strength


def main() -> None:
    windows = cut([trim(signal) for signal in cwru.load_all("DE")])
    model = Classifier(MODELS)
    physics, learned = [], []
    for samples, rpm in zip(windows.samples, windows.rpms):
        physics.append(physics_decision(samples, rpm)[0])
        learned.append(model(samples)[0])
    physics, learned, truth = np.array(physics), np.array(learned), np.asarray(windows.conditions)

    result = {
        "windows": int(len(truth)),
        "physics_matches_label": float((physics == truth).mean()),
        "model_matches_label": float((learned == truth).mean()),
        "model_agrees_with_physics": float((learned == physics).mean()),
        "by_condition": {
            condition: {
                "physics_matches_label": float((physics[truth == condition] == condition).mean()),
                "model_matches_label": float((learned[truth == condition] == condition).mean()),
                "model_agrees_with_physics": float((learned[truth == condition] == physics[truth == condition]).mean()),
            }
            for condition in ("normal", "inner-race", "ball", "outer-race")
        },
    }
    (REPORTS / "physics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
