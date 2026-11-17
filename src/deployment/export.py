"""From a trained network to ONNX: one portable file for every target, then a smaller int8 copy."""

from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch import nn

INPUT, OUTPUT = "windows", "logits"


def to_onnx(model: nn.Module, path: Path, length: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.eval(), torch.zeros(1, length), path,
        input_names=[INPUT], output_names=[OUTPUT],
        dynamic_axes={INPUT: {0: "batch"}, OUTPUT: {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    return path


class _Windows(CalibrationDataReader):
    """Feeds real windows to the quantiser so it can measure each layer's range."""

    def __init__(self, samples: np.ndarray, batch: int = 32):
        self.batches = iter(samples[i:i + batch] for i in range(0, len(samples), batch))

    def get_next(self):
        batch = next(self.batches, None)
        return None if batch is None else {INPUT: batch.astype(np.float32)}


def to_int8(source: Path, target: Path, calibration: np.ndarray) -> Path:
    """Weights and activations stored as 8-bit integers, calibrated on real vibration.

    The per-window normalisation that runs before the first convolution stays
    in full precision: a standard deviation computed in 8 bits loses too much.
    """
    prepared = target.with_name("prepared.onnx")
    quant_pre_process(str(source), str(prepared))
    quantize_static(
        str(prepared), str(target), _Windows(calibration),
        quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        nodes_to_exclude=before_first_convolution(prepared),
    )
    prepared.unlink()
    return target


def before_first_convolution(path: Path) -> list[str]:
    nodes = onnx.load(str(path)).graph.node
    first = next(i for i, node in enumerate(nodes) if node.op_type == "Conv")
    return [node.name for node in nodes[:first]]
