"""How good a set of predictions is, overall and for each condition."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def assess(truth: np.ndarray, predicted: np.ndarray, classes: list[str]) -> dict:
    recalls = recall_score(truth, predicted, labels=classes, average=None, zero_division=0)
    return {
        "macro_f1": round(float(f1_score(truth, predicted, labels=classes, average="macro")), 4),
        "accuracy": round(float(accuracy_score(truth, predicted)), 4),
        "recall": {c: round(float(r), 4) for c, r in zip(classes, recalls)},
        "confusion": confusion_matrix(truth, predicted, labels=classes).tolist(),
    }
