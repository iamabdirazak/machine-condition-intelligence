"""Runs the workflow end to end for one approach, under four protocols.

Leave one load out   train on three loads, test on the fourth (a sanity check)
Single load          train on one load, test on the other three (load shift)
Noise                leave one load out again, with noise added to the test windows
Sensor shift         train on the drive-end sensor, test on the fan-end sensor,
                     before and after unlabelled adaptation (networks only)

Networks are also checked for drift detection: how often each kind of new data
raises the alarm. Usage: python -m src.run [approach]
"""

import json
import sys
from typing import Callable

import numpy as np

from src.acquisition import cwru
from src.contracts import Windows
from src.evaluation.metrics import assess
from src.features import handcrafted
from src.features.augmentation import add_noise
from src.modeling import baseline
from src.modeling.convolution import ConditionNet
from src.monitoring import drift
from src.monitoring.adapt import adapt
from src.paths import REPORTS
from src.preparation import splitting, windowing
from src.training import train

CLASSES = list(cwru.FIRST_ID)
LOADS = list(cwru.RPM)
SNRS = (20, 10, 5, 0, -5)
SEEDS = (0, 1, 2)  # the sensor shift varies between runs, so it is repeated


def features(windows: Windows) -> np.ndarray:
    return handcrafted.extract(windows, cwru.SAMPLE_RATE, cwru.FAULT_ORDERS)[0]


class Baseline:
    def __init__(self, windows: Windows, seed: int = 0):
        self.model = baseline.build(seed).fit(features(windows), windows.conditions)

    def __call__(self, windows: Windows) -> np.ndarray:
        return self.model.predict(features(windows))


class Network:
    def __init__(self, windows: Windows, settings: train.Settings):
        self.model = train.fit(ConditionNet(len(CLASSES)), windows, CLASSES, settings)
        self.reference = drift.calibrate(self.model, windows)

    def __call__(self, windows: Windows) -> np.ndarray:
        return train.predict(self.model, windows, CLASSES)

    def adapted(self, windows: Windows) -> "Network":
        copy = object.__new__(Network)
        copy.model, copy.reference = adapt(self.model, windows), self.reference
        return copy

    def alarm_rate(self, windows: Windows) -> float:
        return float(np.mean(drift.scores(self.reference, self.model, windows) > self.reference.threshold))


APPROACHES: dict[str, Callable[..., object]] = {
    "baseline": Baseline,
    "convolution-clean": lambda w, seed=0: Network(w, train.Settings(augment=False, seed=seed)),
    "convolution": lambda w, seed=0: Network(w, train.Settings(seed=seed)),
    "convolution-transfer": lambda w, seed=0: Network(w, train.Settings(transfer=True, seed=seed)),
}


def score(predict, test: Windows) -> dict:
    return assess(test.conditions, predict(test), CLASSES)


def average(results: list[dict]) -> dict:
    return {
        "macro_f1": round(float(np.mean([r["macro_f1"] for r in results])), 4),
        "recall": {c: round(float(np.mean([r["recall"][c] for r in results])), 4) for c in CLASSES},
    }


def run(approach: str = "baseline") -> dict:
    fit = APPROACHES[approach]
    sensors = {s: [windowing.trim(x) for x in cwru.load_all(s)] for s in cwru.SENSORS}
    signals, other = sensors["DE"], sensors["FE"]
    folds = {load: splitting.by_load(signals, load) for load in LOADS}
    shifted = {load: splitting.by_load(other, load)[1] for load in LOADS}
    predictors = {load: fit(train_set) for load, (train_set, _) in folds.items()}

    single = {}
    for load in LOADS:
        train_set = windowing.cut([s for s in signals if s.load == load], hop=splitting.TRAIN_HOP)
        test_set = windowing.cut([s for s in signals if s.load != load])
        single[f"{load} hp"] = score(fit(train_set), test_set)

    results = {
        "leave one load out": {f"{load} hp": score(predictors[load], test) for load, (_, test) in folds.items()},
        "single load": single,
        "noise": {
            f"{snr} dB": average([score(predictors[load], add_noise(test, snr)) for load, (_, test) in folds.items()])
            for snr in SNRS
        },
    }

    repeats = {load: [predictors[load]] + [fit(folds[load][0], seed) for seed in SEEDS[1:]] for load in LOADS}
    runs = {load: [score(p, shifted[load]) for p in repeats[load]] for load in LOADS}
    results["sensor shift"] = {f"{load} hp": average(runs[load]) for load in LOADS}
    spread = {"before": float(np.std([r["macro_f1"] for load in LOADS for r in runs[load]]))}

    if isinstance(predictors[0], Network):
        adapted = {load: [score(p.adapted(shifted[load]), shifted[load]) for p in repeats[load]] for load in LOADS}
        results["sensor shift, adapted"] = {f"{load} hp": average(adapted[load]) for load in LOADS}
        spread["after"] = float(np.std([r["macro_f1"] for load in LOADS for r in adapted[load]]))
        results["drift alarms"] = {
            scenario: round(float(np.mean([predictors[load].alarm_rate(data(load)) for load in LOADS])), 4)
            for scenario, data in {
                "new load": lambda load: folds[load][1],
                "noise 0 dB": lambda load: add_noise(folds[load][1], 0),
                "other sensor": lambda load: shifted[load],
            }.items()
        }

    results["sensor shift spread"] = {k: round(v, 4) for k, v in spread.items()}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{approach}.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    approach = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    for protocol, folds in run(approach).items():
        if protocol in ("drift alarms", "sensor shift spread"):
            for key, value in folds.items():
                print(f"{approach:<22} {protocol:<24} {key:<13} {value}")
            continue
        for key, result in folds.items():
            print(f"{approach:<22} {protocol:<24} {key:<13} macro-F1 {result['macro_f1']:.3f}")
