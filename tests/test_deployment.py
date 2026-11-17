"""An exported model must behave exactly like the network it came from."""

import json

import numpy as np
import pytest
import torch

from src.deployment import export, verify
from src.deployment.inference import Classifier
from src.modeling.convolution import ConditionNet

LENGTH = 2048
CLASSES = ["normal", "inner-race", "ball", "outer-race"]


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    torch.manual_seed(0)
    model = ConditionNet(len(CLASSES)).eval()
    folder = tmp_path_factory.mktemp("release")
    export.to_onnx(model, folder / "model.onnx", LENGTH)
    card = {"classes": CLASSES, "window": {"samples": LENGTH}}
    (folder / "model.json").write_text(json.dumps(card))
    return model, folder


def test_onnx_matches_pytorch(release):
    model, folder = release
    samples = np.random.default_rng(0).standard_normal((16, LENGTH)).astype(np.float32)
    result = verify.parity(model, folder / "model.onnx", samples)
    assert result["largest_difference"] < 1e-4
    assert result["same_decision"] == 1.0


def test_classifier_names_a_condition(release):
    _, folder = release
    condition, confidence = Classifier(folder, "model.onnx")(np.zeros(LENGTH, dtype=np.float32))
    assert condition in CLASSES
    assert 0.0 <= confidence <= 1.0


def test_classifier_rejects_wrong_length(release):
    _, folder = release
    with pytest.raises(ValueError):
        Classifier(folder, "model.onnx")(np.zeros(100, dtype=np.float32))
