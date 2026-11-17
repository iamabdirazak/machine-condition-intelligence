"""Baseline: gradient-boosted trees on hand-crafted features.

Small, fast and interpretable. It sets the bar every learned model must beat.
"""

from sklearn.ensemble import HistGradientBoostingClassifier


def build(seed: int = 0) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
