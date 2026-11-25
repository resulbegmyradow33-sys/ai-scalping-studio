"""RL comparison baselines: hold, ret_5 rule and seeded random."""

from __future__ import annotations

import numpy as np
import pytest

from scalping.config import Settings
from scalping.features.build import FeatureFrame
from scalping.rl.train import _hold_baseline, _random_baseline, _rule_baseline

WINDOW = 60


def frame(n: int = 300, *, slope: float = 0.0) -> FeatureFrame:
    ts = np.datetime64("2026-01-05T00:00", "us") + np.arange(n).astype("timedelta64[m]")
    price = 1000.0 + slope * np.arange(n, dtype=np.float64)
    ohlc = np.column_stack([price, price + 0.001, price - 0.001, price])
    features = np.zeros((n, 3), dtype=np.float32)
    features[:, 1] = slope
    return FeatureFrame(
        X=features,
        ts=ts.astype("datetime64[us]"),
        ohlc=ohlc,
        spread=np.full(n, 0.20),
        atr=np.full(n, 5.0),
        valid_from=60,
        columns=["a", "ret_5", "c"],
    )


def test_hold_and_flat_rule_stay_flat() -> None:
    flat = frame(slope=0.0)
    flat = flat.__class__(
        X=flat.X,
        ts=flat.ts,
        ohlc=flat.ohlc,
        spread=flat.spread,
        atr=flat.atr,
        valid_from=flat.valid_from,
        columns=["a", "ret_5", "c"],
    )

    assert _hold_baseline(flat, Settings()) == pytest.approx(0.0)
    # ret_5 is always zero on a flat frame, so the rule never buys.
    assert _rule_baseline(flat, Settings()) == pytest.approx(0.0)


def test_rule_baseline_follows_an_uptrend() -> None:
    profit = _rule_baseline(frame(slope=1.0), Settings())

    assert profit is not None
    assert profit > 0.0


def test_random_baseline_is_seeded_and_finite() -> None:
    first = _random_baseline(frame(), Settings(), seed=7)
    second = _random_baseline(frame(), Settings(), seed=7)

    assert first is not None
    assert first == pytest.approx(second)
    assert first < 0.0  # spread and commission make random entries lose


def test_missing_ret5_column_disables_the_rule_baseline() -> None:
    empty = frame()
    without = FeatureFrame(
        X=empty.X,
        ts=empty.ts,
        ohlc=empty.ohlc,
        spread=empty.spread,
        atr=empty.atr,
        valid_from=empty.valid_from,
        columns=["a", "b", "c"],
    )
    assert _rule_baseline(without, Settings()) is None
