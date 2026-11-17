"""Train the final model on every load, export it, check it, and write its model card.

Usage: python -m src.deployment.release
"""

import json
from datetime import date

import numpy as np
import torch

from src.acquisition import cwru
from src.deployment import export, verify
from src.evaluation.metrics import assess
from src.modeling.convolution import ConditionNet
from src.paths import MODELS, REPORTS
from src.preparation import splitting, windowing
from src.training import train

CLASSES = list(cwru.FIRST_ID)
SETTINGS = train.Settings(transfer=True)


def release() -> dict:
    folder = MODELS / f"{date.today().isoformat()}-network"
    signals = [windowing.trim(s) for s in cwru.load_all("DE")]
    training = windowing.cut(signals, hop=splitting.TRAIN_HOP)
    drive_end = windowing.cut(signals)
    checking = windowing.cut([windowing.trim(s) for s in cwru.load_all("FE")])

    model = train.fit(ConditionNet(len(CLASSES)), training, CLASSES, SETTINGS)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "weights.pt")

    full = export.to_onnx(model, folder / "model.onnx", windowing.LENGTH)
    calibration = training.samples[np.random.default_rng(0).choice(len(training), 512, replace=False)]
    small = export.to_int8(full, folder / "model-int8.onnx", calibration)

    def decisions(path):
        return np.asarray(CLASSES)[verify.outputs(path, checking.samples).argmax(1)]

    evaluation = json.loads((REPORTS / "convolution-transfer.json").read_text())
    card = {
        "name": "Machine condition intelligence",
        "task": "Bearing condition from drive-end vibration: " + ", ".join(CLASSES),
        "classes": CLASSES,
        "window": {"samples": windowing.LENGTH, "sample_rate": cwru.SAMPLE_RATE,
                   "seconds": round(windowing.LENGTH / cwru.SAMPLE_RATE, 3)},
        "training": {"data": "CWRU, drive-end sensor, 0.007-inch faults, loads 0–3 hp, 10 s per recording",
                     "settings": SETTINGS.__dict__ | {"noise_range_db": list(SETTINGS.noise_range_db)}},
        "parameters": sum(p.numel() for p in model.parameters()),
        "files": {
            name: {"kilobytes": round(path.stat().st_size / 1024, 1),
                   "milliseconds_per_window": round(verify.latency(path, windowing.LENGTH), 3)}
            for name, path in {"model.onnx": full, "model-int8.onnx": small}.items()
        },
        "parity": {
            sensor: {
                "model.onnx": verify.parity(model, full, windows.samples),
                "model-int8.onnx": verify.parity(model, small, windows.samples),
            }
            for sensor, windows in {"drive end": drive_end, "fan end": checking}.items()
        },
        "fan_end_sensor_without_adaptation": {
            "model.onnx": assess(checking.conditions, decisions(full), CLASSES)["macro_f1"],
            "model-int8.onnx": assess(checking.conditions, decisions(small), CLASSES)["macro_f1"],
        },
        "evaluation": {
            "leave_one_load_out": round(np.mean([f["macro_f1"] for f in evaluation["leave one load out"].values()]), 3),
            "single_load": round(np.mean([f["macro_f1"] for f in evaluation["single load"].values()]), 3),
            "noise_0_db": evaluation["noise"]["0 dB"]["macro_f1"],
            "sensor_shift": round(np.mean([f["macro_f1"] for f in evaluation["sensor shift"].values()]), 3),
            "sensor_shift_adapted": round(np.mean([f["macro_f1"] for f in evaluation["sensor shift, adapted"].values()]), 3),
        },
        "limits": [
            "One bearing type (SKF 6205) on one test rig; other machines need validation.",
            "One fault size (0.007 inch); larger faults were not used in training.",
            "A different sensor position lowers accuracy; adapt with unlabelled data before trusting it.",
            "Noise in training was white Gaussian; very strong coloured noise is a known weakness.",
        ],
    }
    (folder / "model.json").write_text(json.dumps(card, indent=2, default=float))
    return card


if __name__ == "__main__":
    print(json.dumps(release(), indent=2, default=float))
