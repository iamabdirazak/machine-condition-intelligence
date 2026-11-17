"""Adapting a trained network to a new sensor or machine, without labels (AdaBN).

Batch normalisation stores the average and spread of the signals the network
saw in training. When the sensor changes, those statistics no longer match,
and every layer reads its inputs on the wrong scale. Re-estimating them from
a batch of new, unlabelled windows realigns the network. No labels, no
gradient steps, and only a few seconds of data.
"""

import copy

import torch
from torch import nn

from src.contracts import Windows


@torch.no_grad()
def adapt(model: nn.Module, windows: Windows, batch: int = 256) -> nn.Module:
    adapted = copy.deepcopy(model).train()
    for layer in adapted.modules():
        if isinstance(layer, nn.BatchNorm1d):
            layer.reset_running_stats()
            layer.momentum = None  # a plain average over every batch seen
    for part in torch.as_tensor(windows.samples).split(batch):
        adapted(part)
    return adapted.eval()
