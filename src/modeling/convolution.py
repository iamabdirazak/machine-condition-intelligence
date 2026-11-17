"""A small 1D convolutional network that reads raw vibration windows.

The first layer is wide and strided: it learns its own filter bank, the
learned counterpart of the band-pass filter chosen by hand for the envelope
spectrum. Three narrow blocks follow, then global pooling and a classifier.
Every block uses batch normalisation, which is what lets the network adapt to
a new operating condition later without retraining.
"""

import torch
from torch import nn


def block(inputs: int, outputs: int, kernel: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(inputs, outputs, kernel, stride=stride, padding=kernel // 2, bias=False),
        nn.BatchNorm1d(outputs),
        nn.ReLU(),
        nn.MaxPool1d(2),
    )


class ConditionNet(nn.Module):
    def __init__(self, classes: int, width: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            block(1, width, kernel=64, stride=8),
            block(width, 2 * width),
            block(2 * width, 4 * width),
            block(4 * width, 4 * width),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(4 * width, classes)

    def embed(self, windows: torch.Tensor) -> torch.Tensor:
        """What the network has understood about each window, before it decides."""
        centred = windows - windows.mean(dim=1, keepdim=True)
        normalised = centred / (centred.std(dim=1, keepdim=True) + 1e-6)
        return self.encoder(normalised.unsqueeze(1))

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """Windows of shape (batch, samples), each normalised to zero mean and unit variance."""
        return self.classifier(self.embed(windows))
