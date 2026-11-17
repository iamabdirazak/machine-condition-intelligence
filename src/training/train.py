"""Training a network on windows, optionally with noise augmentation."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from src.contracts import Windows
from src.features.augmentation import add_noise, reshape_spectrum


@dataclass(frozen=True)
class Settings:
    epochs: int = 15
    batch: int = 64
    learning_rate: float = 1e-3
    augment: bool = True
    transfer: bool = False
    noise_range_db: tuple[float, float] = (-5.0, 20.0)
    augmented_share: float = 0.5
    seed: int = 0


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit(model: nn.Module, train: Windows, classes: list[str], settings: Settings = Settings()) -> nn.Module:
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    target = torch.as_tensor([classes.index(c) for c in train.conditions])
    place = device()
    model = model.to(place)
    optimiser = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    loss = nn.CrossEntropyLoss()

    for epoch in range(settings.epochs):
        samples = augmented(train, settings, rng, epoch) if settings.augment else train.samples
        inputs = torch.as_tensor(samples)
        model.train()
        for batch in torch.randperm(len(inputs)).split(settings.batch):
            optimiser.zero_grad()
            loss(model(inputs[batch].to(place)), target[batch].to(place)).backward()
            optimiser.step()
    return model.eval().cpu()


def augmented(train: Windows, settings: Settings, rng: np.random.Generator, epoch: int) -> np.ndarray:
    """A fresh copy each epoch: some windows clean, the rest noisy and, optionally, re-filtered."""
    low, high = settings.noise_range_db
    noisy = add_noise(train, rng.uniform(low, high, len(train)), seed=settings.seed + epoch).samples
    keep_clean = rng.random(len(train)) > settings.augmented_share
    samples = np.where(keep_clean[:, None], train.samples, noisy)
    if settings.transfer:
        reshaped = rng.random(len(train)) < settings.augmented_share
        samples[reshaped] = reshape_spectrum(samples[reshaped], rng)
    return samples


@torch.no_grad()
def predict(model: nn.Module, windows: Windows, classes: list[str], batch: int = 512) -> np.ndarray:
    model.eval()
    inputs = torch.as_tensor(windows.samples)
    indices = torch.cat([model(part).argmax(dim=1) for part in inputs.split(batch)])
    return np.asarray(classes)[indices.numpy()]
