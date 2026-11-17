"""Noticing when incoming data no longer looks like the training data.

The network's embedding summarises each window. During training we record
its average and spread. For a batch of new windows, the drift score is how
far, on average, the batch's embedding has moved from that reference,
measured in reference spreads. A threshold set on the training data itself
turns the score into an alarm: adapt, or ask for help, before trusting the
predictions.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.contracts import Windows

BATCH = 32  # about 5.5 s of vibration


@dataclass(frozen=True)
class Reference:
    mean: np.ndarray
    spread: np.ndarray
    threshold: float


@torch.no_grad()
def embed(model: nn.Module, windows: Windows) -> np.ndarray:
    model.eval()
    return torch.cat([model.embed(p) for p in torch.as_tensor(windows.samples).split(512)]).numpy()


def score(reference_mean: np.ndarray, reference_spread: np.ndarray, embeddings: np.ndarray) -> float:
    return float(np.mean(np.abs(embeddings.mean(axis=0) - reference_mean) / reference_spread))


def scores(reference: Reference, model: nn.Module, windows: Windows, seed: int = 0) -> np.ndarray:
    """Drift score of each consecutive-in-random-order batch of windows."""
    embeddings = embed(model, windows)
    order = np.random.default_rng(seed).permutation(len(embeddings))
    batches = [order[i:i + BATCH] for i in range(0, len(order) - BATCH + 1, BATCH)]
    return np.array([score(reference.mean, reference.spread, embeddings[b]) for b in batches])


def calibrate(model: nn.Module, train: Windows, quantile: float = 0.99, trials: int = 500, seed: int = 0) -> Reference:
    """Reference statistics and an alarm threshold from the training data alone."""
    embeddings = embed(model, train)
    mean, spread = embeddings.mean(axis=0), embeddings.std(axis=0) + 1e-6
    rng = np.random.default_rng(seed)
    null = [score(mean, spread, embeddings[rng.choice(len(embeddings), BATCH, replace=False)]) for _ in range(trials)]
    return Reference(mean, spread, float(np.quantile(null, quantile)))
