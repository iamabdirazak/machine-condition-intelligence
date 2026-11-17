"""The honest split must never share a recording between train and test."""

import numpy as np
import pytest

from src.contracts import Signal
from src.preparation import splitting


def synthetic(condition: str, load: int, seconds: float = 1.0) -> Signal:
    samples = np.random.default_rng(load).standard_normal(int(12_000 * seconds))
    return Signal(samples, 12_000, 1750, condition, load, f"{condition}-{load}hp")


SIGNALS = [synthetic(c, load) for c in ("normal", "ball") for load in range(4)]


@pytest.mark.parametrize("test_load", range(4))
def test_by_load_shares_no_recording(test_load):
    train, test = splitting.by_load(SIGNALS, test_load)
    assert set(train.sources).isdisjoint(test.sources)
    assert set(test.loads) == {test_load}
    assert test_load not in set(train.loads)


def test_at_random_does_share_recordings():
    train, test = splitting.at_random(SIGNALS)
    assert set(train.sources) & set(test.sources), "the leaky split is expected to leak"
