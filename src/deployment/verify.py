"""Proof that an exported model behaves like the original, and how fast it runs."""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from src.deployment.export import INPUT


def session(path: Path, threads: int = 1) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def outputs(path: Path, samples: np.ndarray) -> np.ndarray:
    return session(path).run(None, {INPUT: samples.astype(np.float32)})[0]


@torch.no_grad()
def parity(model: nn.Module, path: Path, samples: np.ndarray) -> dict:
    """How far the exported model's outputs are from the original's, and how often they agree."""
    original = model.eval()(torch.as_tensor(samples)).numpy()
    exported = outputs(path, samples)
    return {
        "largest_difference": float(np.max(np.abs(original - exported))),
        "same_decision": float(np.mean(original.argmax(1) == exported.argmax(1))),
    }


def latency(path: Path, length: int, runs: int = 300) -> float:
    """Median milliseconds to classify one window, on one CPU thread."""
    model = session(path)
    window = np.random.default_rng(0).standard_normal((1, length)).astype(np.float32)
    for _ in range(20):
        model.run(None, {INPUT: window})
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        model.run(None, {INPUT: window})
        timings.append((time.perf_counter() - start) * 1000)
    return float(np.median(timings))
