"""Two ways to split: the honest one we use, and the leaky one we measure against.

Honest: hold out an entire motor load. No recording, and so no window or part
of a window, from the test load is ever seen in training. Training windows may
overlap to give the model more views; test windows never do.

Leaky: cut overlapping windows from everything and shuffle them into train and
test. Neighbouring windows share most of their samples, so the test set holds
near-copies of training data. Many published CWRU results are split this way.
"""

import numpy as np

from src.contracts import Signal, Windows
from src.preparation.windowing import LENGTH, cut

TRAIN_HOP = LENGTH // 4  # 75% overlap, training only


def by_load(signals: list[Signal], test_load: int) -> tuple[Windows, Windows]:
    train = [s for s in signals if s.load != test_load]
    test = [s for s in signals if s.load == test_load]
    return cut(train, hop=TRAIN_HOP), cut(test, hop=LENGTH)


def at_random(signals: list[Signal], test_fraction: float = 0.2, seed: int = 0) -> tuple[Windows, Windows]:
    windows = cut(signals, hop=TRAIN_HOP)
    shuffled = np.random.default_rng(seed).permutation(len(windows))
    is_test = np.zeros(len(windows), dtype=bool)
    is_test[shuffled[: int(test_fraction * len(windows))]] = True
    return windows.where(~is_test), windows.where(is_test)
