"""Classifying vibration with a released model. Needs only NumPy and ONNX Runtime.

    classifier = Classifier(Path("models/2026-11-17-network"))
    condition, confidence = classifier(window)   # window: 2048 samples at 12 kHz
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

INPUT = "windows"


class Classifier:
    def __init__(self, release: Path, file: str = "model-int8.onnx"):
        card = json.loads((release / "model.json").read_text())
        self.classes = card["classes"]
        self.length = card["window"]["samples"]
        self.model = ort.InferenceSession(str(release / file), providers=["CPUExecutionProvider"])

    def __call__(self, window: np.ndarray) -> tuple[str, float]:
        if window.shape != (self.length,):
            raise ValueError(f"Window has shape {window.shape}; expected ({self.length},)")
        logits = self.model.run(None, {INPUT: window[None].astype(np.float32)})[0][0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        best = int(probabilities.argmax())
        return self.classes[best], float(probabilities[best])
